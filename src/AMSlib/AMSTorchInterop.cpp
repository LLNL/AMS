#include "AMSTorchInterop.hpp"

#include <stdexcept>
#include <utility>

#include "wf/resource_manager.hpp"

namespace ams
{
namespace
{
AMSDType fromTorchDType(c10::ScalarType type)
{
  switch (type) {
    case torch::kFloat32:
      return AMS_SINGLE;
    case torch::kFloat64:
      return AMS_DOUBLE;
    case torch::kInt32:
      return AMS_INT32;
    case torch::kInt64:
      return AMS_INT64;
    default:
      throw std::invalid_argument("Unsupported Torch tensor dtype");
  }
}

c10::ScalarType toTorchDType(AMSDType type)
{
  switch (type) {
    case AMS_SINGLE:
      return torch::kFloat32;
    case AMS_DOUBLE:
      return torch::kFloat64;
    case AMS_INT32:
      return torch::kInt32;
    case AMS_INT64:
      return torch::kInt64;
    default:
      throw std::invalid_argument("Unsupported AMS tensor dtype");
  }
}

AMSResourceType fromTorchDevice(const torch::Tensor& tensor)
{
  if (tensor.device().is_cpu())
    return tensor.is_pinned() ? AMS_PINNED : AMS_HOST;
  if (tensor.device().is_cuda()) return AMS_DEVICE;
  throw std::invalid_argument("Unsupported Torch tensor device");
}

torch::Device toTorchDevice(AMSResourceType type)
{
  if (type == AMS_HOST || type == AMS_PINNED) return torch::Device(torch::kCPU);
  if (type == AMS_DEVICE) return torch::Device(torch::kCUDA);
  throw std::invalid_argument("Unsupported AMS tensor device");
}

SmallVector<AMSTensor::IntDimType> dims(c10::IntArrayRef values)
{
  SmallVector<AMSTensor::IntDimType> result;
  result.reserve(values.size());
  for (int64_t value : values)
    result.push_back(value);
  return result;
}

template <typename T>
AMSTensor makeView(torch::Tensor tensor,
                   ArrayRef<AMSTensor::IntDimType> shape,
                   ArrayRef<AMSTensor::IntDimType> strides,
                   AMSResourceType location,
                   AMSTensor::LifetimeToken owner)
{
  return AMSTensor::view<T>(
      tensor.data_ptr<T>(), shape, strides, location, std::move(owner));
}
}  // namespace

AMSTensor fromTorchView(torch::Tensor tensor)
{
  if (!tensor.defined())
    throw std::invalid_argument("Torch tensor is undefined");
  const AMSDType dtype = fromTorchDType(tensor.scalar_type());
  const AMSResourceType location = fromTorchDevice(tensor);
  auto shape = dims(tensor.sizes());
  auto strides = dims(tensor.strides());
  auto owner =
      std::static_pointer_cast<void>(std::make_shared<torch::Tensor>(tensor));
  switch (dtype) {
    case AMS_SINGLE:
      return makeView<float>(
          tensor, shape, strides, location, std::move(owner));
    case AMS_DOUBLE:
      return makeView<double>(
          tensor, shape, strides, location, std::move(owner));
    case AMS_INT32:
      return makeView<int32_t>(
          tensor, shape, strides, location, std::move(owner));
    case AMS_INT64:
      return makeView<int64_t>(
          tensor, shape, strides, location, std::move(owner));
    default:
      throw std::invalid_argument("Unsupported Torch tensor dtype");
  }
}

AMSTensor fromTorchCopy(const torch::Tensor& tensor)
{
  if (!tensor.defined())
    throw std::invalid_argument("Torch tensor is undefined");
  torch::Tensor source = tensor.detach().contiguous();
  const AMSDType dtype = fromTorchDType(source.scalar_type());
  const AMSResourceType location = fromTorchDevice(source);
  auto shape = dims(source.sizes());
  auto strides = dims(source.strides());
  AMSTensor result = [&]() {
    switch (dtype) {
      case AMS_SINGLE:
        return AMSTensor::create<float>(shape, strides, location);
      case AMS_DOUBLE:
        return AMSTensor::create<double>(shape, strides, location);
      case AMS_INT32:
        return AMSTensor::create<int32_t>(shape, strides, location);
      case AMS_INT64:
        return AMSTensor::create<int64_t>(shape, strides, location);
      default:
        throw std::invalid_argument("Unsupported Torch tensor dtype");
    }
  }();
  if (result.nbytes())
    internal::_raw_copy(source.data_ptr(),
                        location,
                        result.data_ptr(),
                        location,
                        result.nbytes());
  return result;
}

torch::Tensor toTorchView(AMSTensor& tensor)
{
  std::vector<int64_t> shape(tensor.shape().begin(), tensor.shape().end());
  std::vector<int64_t> strides(tensor.strides().begin(),
                               tensor.strides().end());
  auto options = torch::TensorOptions()
                     .dtype(toTorchDType(tensor.dtype()))
                     .device(toTorchDevice(tensor.location()));
  auto owner = tensor.lifetimeToken();
  return torch::from_blob(
      tensor.data_ptr(),
      shape,
      strides,
      [owner = std::move(owner)](void*) mutable { owner.reset(); },
      options);
}

torch::Tensor toTorchCopy(const AMSTensor& tensor)
{
  // Torch has no read-only tensor view, so keep this view private to the copy.
  std::vector<int64_t> shape(tensor.shape().begin(), tensor.shape().end());
  std::vector<int64_t> strides(tensor.strides().begin(),
                               tensor.strides().end());
  auto options = torch::TensorOptions()
                     .dtype(toTorchDType(tensor.dtype()))
                     .device(toTorchDevice(tensor.location()));
  auto owner = tensor.lifetimeToken();
  auto view = torch::from_blob(
      const_cast<void*>(tensor.data_ptr()),
      shape,
      strides,
      [owner = std::move(owner)](void*) mutable { owner.reset(); },
      options);
  return view.clone(torch::MemoryFormat::Contiguous);
}
}  // namespace ams
