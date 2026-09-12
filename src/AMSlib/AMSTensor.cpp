#include "AMSTensor.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "wf/resource_manager.hpp"

using namespace ams;

namespace
{
using Dim = AMSTensor::IntDimType;

size_t elementSize(AMSDType dtype)
{
  switch (dtype) {
    case AMS_SINGLE:
      return sizeof(float);
    case AMS_DOUBLE:
      return sizeof(double);
    case AMS_INT32:
      return sizeof(int32_t);
    case AMS_INT64:
      return sizeof(int64_t);
    default:
      throw std::invalid_argument("Unsupported AMSTensor dtype");
  }
}

size_t checkedAdd(size_t a, size_t b)
{
  if (b > std::numeric_limits<size_t>::max() - a)
    throw std::overflow_error("AMSTensor size addition overflow");
  return a + b;
}

size_t checkedMul(size_t a, size_t b)
{
  if (a && b > std::numeric_limits<size_t>::max() / a)
    throw std::overflow_error("AMSTensor size multiplication overflow");
  return a * b;
}

struct Metadata {
  Dim elements;
  size_t logicalBytes;
  size_t spanElements;
  size_t storageBytes;
  bool contiguous;
};

Metadata validateMetadata(ArrayRef<Dim> shape,
                          ArrayRef<Dim> strides,
                          AMSDType dtype,
                          AMSResourceType location)
{
  if (shape.size() != strides.size())
    throw std::invalid_argument("AMSTensor shape/stride ranks differ");
  if (location != AMS_HOST && location != AMS_DEVICE && location != AMS_PINNED)
    throw std::invalid_argument("Invalid AMSTensor memory resource");

  const size_t itemSize = elementSize(dtype);
  size_t elements = 1;
  bool empty = false;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] < 0)
      throw std::invalid_argument("AMSTensor dimensions cannot be negative");
    if (strides[i] <= 0)
      throw std::invalid_argument("AMSTensor strides must be positive");
    if (shape[i] == 0) empty = true;
    elements = checkedMul(elements, static_cast<size_t>(shape[i]));
  }
  if (empty) elements = 0;
  if (elements > static_cast<size_t>(std::numeric_limits<Dim>::max()))
    throw std::overflow_error("AMSTensor element count exceeds IntDimType");

  size_t span = elements == 0 ? 0 : 1;
  if (elements != 0) {
    std::vector<size_t> order;
    for (size_t i = 0; i < shape.size(); ++i)
      if (shape[i] > 1) order.push_back(i);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return strides[a] < strides[b];
    });
    size_t required = 1;
    for (size_t axis : order) {
      const size_t stride = static_cast<size_t>(strides[axis]);
      if (stride < required)
        throw std::invalid_argument(
            "AMSTensor strides describe overlapping storage");
      required =
          checkedAdd(required,
                     checkedMul(static_cast<size_t>(shape[axis] - 1), stride));
    }
    span = required;
  }

  bool contiguous = true;
  size_t expected = 1;
  for (size_t i = shape.size(); i-- > 0;) {
    if (shape[i] > 1 && static_cast<size_t>(strides[i]) != expected)
      contiguous = false;
    expected = checkedMul(expected, static_cast<size_t>(shape[i]));
  }
  if (empty) contiguous = true;

  return {static_cast<Dim>(elements),
          checkedMul(elements, itemSize),
          span,
          checkedMul(span, itemSize),
          contiguous};
}

SmallVector<Dim> contiguousStrides(ArrayRef<Dim> shape)
{
  SmallVector<Dim> result(shape.size(), 1);
  size_t stride = 1;
  for (size_t i = shape.size(); i-- > 0;) {
    if (stride > static_cast<size_t>(std::numeric_limits<Dim>::max()))
      throw std::overflow_error("AMSTensor stride exceeds IntDimType");
    result[i] = static_cast<Dim>(stride);
    stride = checkedMul(stride, static_cast<size_t>(shape[i]));
  }
  return result;
}

AMSTensor::LifetimeToken allocateStorage(size_t bytes,
                                         size_t alignment,
                                         AMSResourceType location,
                                         uint8_t*& data)
{
  auto allocator = ResourceManager::getInstance().getAllocator(location);
  const size_t allocationBytes = bytes == 0 ? alignment : bytes;
  void* ptr = allocator->allocate(allocationBytes, alignment);
  if (!ptr) throw std::bad_alloc();
  data = static_cast<uint8_t*>(ptr);
  return AMSTensor::LifetimeToken(ptr, [allocator](void* p) {
    if (p) allocator->deallocate(p);
  });
}

size_t logicalOffset(size_t linear, ArrayRef<Dim> shape, ArrayRef<Dim> strides)
{
  size_t offset = 0;
  for (size_t axis = shape.size(); axis-- > 0;) {
    const size_t dim = static_cast<size_t>(shape[axis]);
    const size_t index = dim == 0 ? 0 : linear % dim;
    if (dim != 0) linear /= dim;
    offset = checkedAdd(offset,
                        checkedMul(index, static_cast<size_t>(strides[axis])));
  }
  return offset;
}
}  // namespace

AMSTensor::AMSTensor(uint8_t* data,
                     ArrayRef<IntDimType> shapes,
                     ArrayRef<IntDimType> strides,
                     AMSDType dtype,
                     AMSResourceType location,
                     bool writable,
                     LifetimeToken lifetime)
    : _data(data),
      _shape(shapes),
      _strides(strides),
      _dType(dtype),
      _location(location),
      _writable(writable),
      _valid(true),
      _lifetime(std::move(lifetime))
{
  const Metadata metadata = validateMetadata(shapes, strides, dtype, location);
  _elements = metadata.elements;
  _element_size = static_cast<IntDimType>(elementSize(dtype));
  _bytes = metadata.logicalBytes;
  _storage_bytes = metadata.storageBytes;
  _contiguous = metadata.contiguous;
  if (!_data && _elements != 0)
    throw std::invalid_argument("Non-empty AMSTensor requires a data pointer");
  if (_data &&
      reinterpret_cast<uintptr_t>(_data) % static_cast<size_t>(_element_size) !=
          0)
    throw std::invalid_argument("AMSTensor data pointer is improperly aligned");
}

void AMSTensor::requireValid() const
{
  if (!_valid) throw std::logic_error("AMSTensor is moved-from");
}

void AMSTensor::requireDataType(AMSDType requested) const
{
  if (_dType != requested)
    throw std::invalid_argument("AMSTensor dtype mismatch");
}

void AMSTensor::resetMovedFrom() noexcept
{
  _data = nullptr;
  _elements = 0;
  _element_size = 0;
  _shape.clear();
  _strides.clear();
  _dType = AMS_UNKNOWN_TYPE;
  _location = AMS_UNKNOWN;
  _contiguous = false;
  _writable = false;
  _valid = false;
  _bytes = 0;
  _storage_bytes = 0;
  _lifetime.reset();
}

template <typename ScalarType>
AMSTensor AMSTensor::create(ArrayRef<IntDimType> shapes,
                            ArrayRef<IntDimType> strides,
                            AMSResourceType location)
{
  const Metadata metadata =
      validateMetadata(shapes, strides, dtypeFor<ScalarType>(), location);
  uint8_t* data = nullptr;
  auto owner = allocateStorage(metadata.storageBytes,
                               alignof(std::remove_cv_t<ScalarType>),
                               location,
                               data);
  return AMSTensor(data,
                   shapes,
                   strides,
                   dtypeFor<ScalarType>(),
                   location,
                   true,
                   std::move(owner));
}

template <typename ScalarType>
AMSTensor AMSTensor::view(ScalarType* data,
                          ArrayRef<IntDimType> shapes,
                          ArrayRef<IntDimType> strides,
                          AMSResourceType location)
{
  return view(data, shapes, strides, location, {});
}

template <typename ScalarType>
AMSTensor AMSTensor::view(ScalarType* data,
                          ArrayRef<IntDimType> shapes,
                          ArrayRef<IntDimType> strides,
                          AMSResourceType location,
                          LifetimeToken lifetime)
{
  using U = std::remove_cv_t<ScalarType>;
  return AMSTensor(reinterpret_cast<uint8_t*>(const_cast<U*>(data)),
                   shapes,
                   strides,
                   dtypeFor<U>(),
                   location,
                   !std::is_const_v<ScalarType>,
                   std::move(lifetime));
}

AMSTensor AMSTensor::view(AMSTensor& tensor)
{
  tensor.requireValid();
  return AMSTensor(tensor._data,
                   tensor._shape,
                   tensor._strides,
                   tensor._dType,
                   tensor._location,
                   tensor._writable,
                   tensor._lifetime);
}

AMSTensor AMSTensor::view(const AMSTensor& tensor)
{
  tensor.requireValid();
  return AMSTensor(tensor._data,
                   tensor._shape,
                   tensor._strides,
                   tensor._dType,
                   tensor._location,
                   false,
                   tensor._lifetime);
}

AMSTensor::AMSTensor(AMSTensor&& other) noexcept
    : _data(other._data),
      _elements(other._elements),
      _element_size(other._element_size),
      _shape(std::move(other._shape)),
      _strides(std::move(other._strides)),
      _dType(other._dType),
      _location(other._location),
      _contiguous(other._contiguous),
      _writable(other._writable),
      _valid(other._valid),
      _bytes(other._bytes),
      _storage_bytes(other._storage_bytes),
      _lifetime(std::move(other._lifetime))
{
  other.resetMovedFrom();
}

AMSTensor& AMSTensor::operator=(AMSTensor&& other) noexcept
{
  if (this != &other) {
    _data = other._data;
    _elements = other._elements;
    _element_size = other._element_size;
    _shape = std::move(other._shape);
    _strides = std::move(other._strides);
    _dType = other._dType;
    _location = other._location;
    _contiguous = other._contiguous;
    _writable = other._writable;
    _valid = other._valid;
    _bytes = other._bytes;
    _storage_bytes = other._storage_bytes;
    _lifetime = std::move(other._lifetime);
    other.resetMovedFrom();
  }
  return *this;
}

AMSTensor AMSTensor::transpose(IntDimType axis1, IntDimType axis2)
{
  requireValid();
  if (axis1 < 0 || axis2 < 0 || static_cast<size_t>(axis1) >= _shape.size() ||
      static_cast<size_t>(axis2) >= _shape.size())
    throw std::out_of_range("AMSTensor transpose axis is out of range");
  auto shape = _shape;
  auto strides = _strides;
  std::swap(shape[axis1], shape[axis2]);
  std::swap(strides[axis1], strides[axis2]);
  return AMSTensor(
      _data, shape, strides, _dType, _location, _writable, _lifetime);
}

AMSTensor AMSTensor::transpose(IntDimType axis1, IntDimType axis2) const
{
  requireValid();
  if (axis1 < 0 || axis2 < 0 || static_cast<size_t>(axis1) >= _shape.size() ||
      static_cast<size_t>(axis2) >= _shape.size())
    throw std::out_of_range("AMSTensor transpose axis is out of range");
  auto shape = _shape;
  auto strides = _strides;
  std::swap(shape[axis1], shape[axis2]);
  std::swap(strides[axis1], strides[axis2]);
  return AMSTensor(_data, shape, strides, _dType, _location, false, _lifetime);
}

AMSTensor AMSTensor::clone() const
{
  requireValid();
  auto strides = contiguousStrides(_shape);
  AMSTensor result = [&]() {
    switch (_dType) {
      case AMS_SINGLE:
        return create<float>(_shape, strides, _location);
      case AMS_DOUBLE:
        return create<double>(_shape, strides, _location);
      case AMS_INT32:
        return create<int32_t>(_shape, strides, _location);
      case AMS_INT64:
        return create<int64_t>(_shape, strides, _location);
      default:
        throw std::invalid_argument("Unsupported AMSTensor dtype");
    }
  }();
  if (_elements == 0) return result;
  if (_contiguous) {
    internal::_raw_copy(_data, _location, result._data, _location, _bytes);
  } else {
    for (size_t i = 0; i < static_cast<size_t>(_elements); ++i) {
      const size_t src = checkedMul(logicalOffset(i, _shape, _strides),
                                    static_cast<size_t>(_element_size));
      const size_t dst = checkedMul(i, static_cast<size_t>(_element_size));
      internal::_raw_copy(_data + src,
                          _location,
                          result._data + dst,
                          _location,
                          static_cast<size_t>(_element_size));
    }
  }
  return result;
}

AMSTensor AMSTensor::concat(ArrayRef<AMSTensor> tensors, AMSDType inputDType)
{
  if (tensors.empty())
    throw std::invalid_argument("AMSTensor::concat requires input tensors");
  elementSize(inputDType);
  const AMSTensor& first = tensors[0];
  first.requireValid();
  if (first._shape.empty())
    throw std::invalid_argument("Cannot concatenate scalar tensors");
  const size_t rank = first._shape.size();
  size_t last = 0;
  for (const auto& tensor : tensors) {
    tensor.requireValid();
    if (tensor._dType != inputDType)
      throw std::invalid_argument("AMSTensor::concat dtype mismatch");
    if (tensor._location != first._location)
      throw std::invalid_argument("AMSTensor::concat location mismatch");
    if (tensor._shape.size() != rank)
      throw std::invalid_argument("AMSTensor::concat rank mismatch");
    for (size_t axis = 0; axis + 1 < rank; ++axis)
      if (tensor._shape[axis] != first._shape[axis])
        throw std::invalid_argument("AMSTensor::concat shape mismatch");
    last = checkedAdd(last, static_cast<size_t>(tensor._shape.back()));
  }
  if (last > static_cast<size_t>(std::numeric_limits<Dim>::max()))
    throw std::overflow_error("AMSTensor::concat dimension overflow");
  SmallVector<Dim> shape(first._shape.begin(), first._shape.end());
  shape.back() = static_cast<Dim>(last);
  auto strides = contiguousStrides(shape);
  AMSTensor result = [&]() {
    switch (inputDType) {
      case AMS_SINGLE:
        return create<float>(shape, strides, first._location);
      case AMS_DOUBLE:
        return create<double>(shape, strides, first._location);
      case AMS_INT32:
        return create<int32_t>(shape, strides, first._location);
      case AMS_INT64:
        return create<int64_t>(shape, strides, first._location);
      default:
        throw std::invalid_argument("Unsupported AMSTensor dtype");
    }
  }();
  size_t rows = 1;
  for (size_t axis = 0; axis + 1 < rank; ++axis)
    rows = checkedMul(rows, static_cast<size_t>(shape[axis]));
  size_t dstLinear = 0;
  for (size_t row = 0; row < rows; ++row) {
    for (const auto& tensor : tensors) {
      const size_t width = static_cast<size_t>(tensor._shape.back());
      for (size_t col = 0; col < width; ++col) {
        const size_t logical = checkedAdd(checkedMul(row, width), col);
        const size_t src =
            checkedMul(logicalOffset(logical, tensor._shape, tensor._strides),
                       static_cast<size_t>(tensor._element_size));
        const size_t dst =
            checkedMul(dstLinear++, static_cast<size_t>(tensor._element_size));
        internal::_raw_copy(tensor._data + src,
                            tensor._location,
                            result._data + dst,
                            result._location,
                            static_cast<size_t>(tensor._element_size));
      }
    }
  }
  return result;
}

#define AMS_INSTANTIATE(T)                                          \
  template AMSTensor AMSTensor::create<T>(ArrayRef<IntDimType>,     \
                                          ArrayRef<IntDimType>,     \
                                          AMSResourceType);         \
  template AMSTensor AMSTensor::view<T>(T*,                         \
                                        ArrayRef<IntDimType>,       \
                                        ArrayRef<IntDimType>,       \
                                        AMSResourceType);           \
  template AMSTensor AMSTensor::view<const T>(const T*,             \
                                              ArrayRef<IntDimType>, \
                                              ArrayRef<IntDimType>, \
                                              AMSResourceType);     \
  template AMSTensor AMSTensor::view<T>(T*,                         \
                                        ArrayRef<IntDimType>,       \
                                        ArrayRef<IntDimType>,       \
                                        AMSResourceType,            \
                                        LifetimeToken);             \
  template AMSTensor AMSTensor::view<const T>(const T*,             \
                                              ArrayRef<IntDimType>, \
                                              ArrayRef<IntDimType>, \
                                              AMSResourceType,      \
                                              LifetimeToken)

AMS_INSTANTIATE(float);
AMS_INSTANTIATE(double);
AMS_INSTANTIATE(int32_t);
AMS_INSTANTIATE(int64_t);
#undef AMS_INSTANTIATE
