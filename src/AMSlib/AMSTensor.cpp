#include "AMSTensor.hpp"

#include <stdexcept>

#include "AMS.h"
#include "ArrayRef.hpp"
#include "SmallVector.hpp"
#include "include/AMSTensor.hpp"
#include "wf/resource_manager.hpp"
#include "wf/utils.hpp"

using namespace ams;

/**
   * @brief Computes the number of elements in the tensor given its shape.
   * @param[in] shapes The shape of the tensor as an array reference.
   * @return The total number of elements in the tensor.
   */
template <typename T>
static inline AMSTensor::IntDimType computeNumElements(ams::ArrayRef<T> shapes)
{
  return std::accumulate(shapes.begin(),
                         shapes.end(),
                         1,
                         std::multiplies<AMSTensor::IntDimType>());
}

bool AMSTensor::isContiguous(ams::ArrayRef<AMSTensor::IntDimType> shape,
                         ams::ArrayRef<AMSTensor::IntDimType> strides) const
{
  const size_t ndim = shape.size();
  if (ndim == 0) return true;
  if (strides[ndim - 1] != 1) return false;
  for (int i = ndim - 2; i >= 0; --i) {
    if (strides[i] != strides[i + 1] * shape[i + 1])
      return false;
  }
  return true;
}

namespace
{
template <typename T>
constexpr AMSDType scalar_to_ams_dtype()
{
  using U = std::remove_cv_t<T>;
  if constexpr (std::is_same_v<U, float>) {
    return AMS_SINGLE;
  } else if constexpr (std::is_same_v<U, double>) {
    return AMS_DOUBLE;
  } else if constexpr (std::is_same_v<U, int32_t>) {
    return AMS_INT32;
  } else if constexpr (std::is_same_v<U, int64_t>) {
    return AMS_INT64;
  } else {
    static_assert(!sizeof(T), "Unsupported AMS scalar type");
  }
}
}  // namespace

AMSTensor::AMSTensor(uint8_t* data,
                     ams::ArrayRef<AMSTensor::IntDimType> shapes,
                     ams::ArrayRef<AMSTensor::IntDimType> strides,
                     AMSDType dType,
                     AMSResourceType location,
                     bool view)
    : _data(data),
      _element_size(dtype_to_size(dType)),
      _shape(shapes),
      _strides(strides),
      _dType(dType),
      _location(location),
      _owned(!view)
{
  _elements = computeNumElements(shapes);
  _bytes = _elements * _element_size;
  _contiguous = isContiguous(shapes, strides);
  if (!_data) {
    throw std::runtime_error("Generating tensor with Null Pointer AMSTensor.");
  }
}

template <typename ScalarType>
AMSTensor AMSTensor::create(ams::ArrayRef<AMSTensor::IntDimType> shapes,
                            ams::ArrayRef<AMSTensor::IntDimType> strides,
                            AMSResourceType location)
{
  auto numElements = computeNumElements(shapes);
  auto& rm = ams::ResourceManager::getInstance();
  using U = std::remove_cv_t<ScalarType>;
  auto allocationElements = numElements == 0 ? 1 : numElements;
  U* data = rm.allocate<U>(allocationElements, location, sizeof(U));
  return AMSTensor(reinterpret_cast<uint8_t*>(data),
                   shapes,
                   strides,
                   scalar_to_ams_dtype<U>(),
                   location);
}

template <typename ScalarType>
AMSTensor AMSTensor::view(ScalarType* data,
                          ams::ArrayRef<AMSTensor::IntDimType> shapes,
                          ams::ArrayRef<AMSTensor::IntDimType> strides,
                          AMSResourceType location)
{
  using U = std::remove_cv_t<ScalarType>;
  return AMSTensor(reinterpret_cast<uint8_t*>(const_cast<U*>(data)),
                   shapes,
                   strides,
                   scalar_to_ams_dtype<U>(),
                   location,
                   true);
}

AMSTensor AMSTensor::view(AMSTensor& tensor)
{
  if (tensor._dType == AMS_DOUBLE)
    return AMSTensor::view((double*)tensor._data,
                           tensor._shape,
                           tensor._strides,
                           tensor._location);
  else if (tensor._dType == AMS_SINGLE)
    return AMSTensor::view((float*)tensor._data,
                           tensor._shape,
                           tensor._strides,
                           tensor._location);
  else if (tensor._dType == AMS_INT32)
    return AMSTensor::view((int32_t*)tensor._data,
                           tensor._shape,
                           tensor._strides,
                           tensor._location);
  else if (tensor._dType == AMS_INT64)
    return AMSTensor::view((int64_t*)tensor._data,
                           tensor._shape,
                           tensor._strides,
                           tensor._location);
  throw std::runtime_error(
      "Creating view through copying constructor has incorrect dtype");
}

AMSTensor::~AMSTensor()
{
  // Only release whenwe own the pointer
  if (_owned && _data) {
    auto& rm = ams::ResourceManager::getInstance();
    rm.deallocate(_data, _location);
    _data = nullptr;
    _owned = false;
  }
}

AMSTensor::AMSTensor(AMSTensor&& other) noexcept
    : _data(other._data),
      _elements(other._elements),
      _element_size(other._element_size),
      _shape(std::move(other._shape)),
      _strides(std::move(other._strides)),
      _dType(other._dType),
      _location(other._location),
      _owned(other._owned),
      _contiguous(other._contiguous),
      _bytes(other._bytes)
{
  other._data = nullptr;
  other._owned = false;
}

AMSTensor& AMSTensor::operator=(AMSTensor&& other) noexcept
{
  if (this != &other) {
    // Free existing resources if we own them
    if (_owned && _data) {
      auto& rm = ams::ResourceManager::getInstance();
      rm.deallocate(_data, _location);
    }
    // Steal resources from `other`
    _data = other._data;
    _elements = other._elements;
    _element_size = other._element_size;
    _shape = std::move(other._shape);
    _strides = std::move(other._strides);
    _dType = other._dType;
    _location = other._location;
    _owned = other._owned;
    _contiguous = other._contiguous;
    _bytes = other._bytes;

    other._data = nullptr;
    other._owned = false;
  }
  return *this;
}

AMSTensor AMSTensor::transpose(AMSTensor::IntDimType axis1,
                               AMSTensor::IntDimType axis2) const
{
  // Ensure the axes are within bounds
  if (axis1 >= _shape.size() || axis2 >= _shape.size()) {
    throw std::out_of_range("Transpose axes are out of bounds");
  }

  // Create new shape and strides for the transposed tensor
  auto newShape = _shape;
  auto newStrides = _strides;

  // Swap the specified axes in both shape and strides
  std::swap(newShape[axis1], newShape[axis2]);
  std::swap(newStrides[axis1], newStrides[axis2]);

  // Create a new tensor with the same data, new shape, and strides
  if (dtype() == AMSDType::AMS_DOUBLE)
    return view((double*)_data, newShape, newStrides, _location);
  else if (dtype() == AMSDType::AMS_SINGLE)
    return view((float*)_data, newShape, newStrides, _location);
  else if (dtype() == AMSDType::AMS_INT32)
    return view((int32_t*)_data, newShape, newStrides, _location);
  else if (dtype() == AMSDType::AMS_INT64)
    return view((int64_t*)_data, newShape, newStrides, _location);
  // NOTE: Use defensive programming here and just crash. We can fix a better interface later
  // for error handling.
  throw std::runtime_error("Unknow data type in transpose\n");
}

AMSTensor AMSTensor::clone() const
{
  auto& rm = ams::ResourceManager::getInstance();
  const size_t ndim = _shape.size();

  uint8_t* dstData =
      rm.allocate<uint8_t>(static_cast<size_t>(_elements) * _element_size,
                           _location);

  // Compute contiguous strides (C style) for the destination
  ams::SmallVector<IntDimType> dstStrides(ndim);
  if (ndim > 0) {
    dstStrides[ndim-1] = 1;
    for (int i = static_cast<int>(ndim) - 2; i >= 0; --i)
      dstStrides[i] = dstStrides[i+1] * _shape[i+1];
  }

  if (_contiguous) {
    ams::internal::_raw_copy(static_cast<void*>(_data),
                             _location,
                             static_cast<void*>(dstData),
                             _location,
                             static_cast<size_t>(_elements) * _element_size);
  } else {
    // Slow path: element-wise copy for non-contiguous tensors.
    // We iterate over every element using an N-dimensional index,
    // compute the source offset from the original strides and the
    // destination offset from the contiguous strides, then copy
    // one element at a time.

    ams::SmallVector<IntDimType> idx(ndim, 0);
    for (IntDimType e = 0; e < _elements; ++e) {
      // Compute source and destination byte offsets
      IntDimType srcOffset = 0;
      IntDimType dstOffset = 0;
      for (size_t d = 0; d < ndim; ++d) {
        srcOffset += idx[d] * _strides[d];
        dstOffset += idx[d] * dstStrides[d];
      }

      ams::internal::_raw_copy(
          static_cast<void*>(_data + srcOffset * _element_size),
          _location,
          static_cast<void*>(dstData + dstOffset * _element_size),
          _location,
          static_cast<size_t>(_element_size));

      // Increment the N-dimensional index (rightmost dimension first)
      for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
        if (++idx[d] < _shape[d]) break;
        idx[d] = 0;
      }
    }
  }

  // Construct the new owning tensor using the private constructor
  return AMSTensor(dstData, _shape, dstStrides, _dType, _location, false);
}

AMSTensor AMSTensor::concat(ArrayRef<AMSTensor> tensors, AMSDType inputDType)
{
  if (tensors.size() == 1) {
    // Single tensor: just return a view
    return AMSTensor::view(const_cast<AMSTensor&>(tensors[0]));
  }

  // Compute concatenated shape: all dims same except last which sums
  auto firstShape = tensors[0].shape();
  size_t ndim = firstShape.size();
  size_t lastDimTotal = 0;
  for (auto& t : tensors) {
    lastDimTotal += t.shape()[ndim-1];
  }

  ams::SmallVector<AMSTensor::IntDimType> newShape(firstShape.begin(), firstShape.end());
  newShape[ndim-1] = static_cast<AMSTensor::IntDimType>(lastDimTotal);

  // Compute contiguous strides for the concatenated tensor
  ams::SmallVector<AMSTensor::IntDimType> newStrides(ndim);
  newStrides[ndim - 1] = 1;
  for (int i = static_cast<int>(ndim) - 2; i >= 0; --i) {
    newStrides[i] = newStrides[i+1] * newShape[i + 1];
  }

  size_t elemSize = dtype_to_size(inputDType);
  size_t totalElements = 1;
  for (auto s : newShape) totalElements *= s;
  size_t totalBytes = totalElements * elemSize;

  auto& rm = ams::ResourceManager::getInstance();
  uint8_t* buffer = rm.allocate<uint8_t>(totalBytes, AMSResourceType::AMS_HOST);

  // Copy data row by row: for each row, copy each tensor's last-dim slice
  size_t numRows = 1;
  for (size_t i = 0; i < ndim - 1; ++i) numRows *= firstShape[i];

  size_t dstOffset = 0;
  for (size_t row = 0; row < numRows; ++row) {
    for (auto& t : tensors) {
      size_t sliceBytes = t.shape()[ndim - 1] * elemSize;
      std::memcpy(buffer + dstOffset,
                  static_cast<const uint8_t*>(t.data_ptr()) + row * sliceBytes,
                  sliceBytes);
      dstOffset += sliceBytes;
    }
  }

  // Create owning tensor from the buffer
  // TODO: improve error handling
  if (inputDType == AMSDType::AMS_SINGLE)
    return AMSTensor::view(reinterpret_cast<float*>(buffer), newShape, newStrides, AMSResourceType::AMS_HOST);
  else if (inputDType == AMSDType::AMS_DOUBLE)
    return AMSTensor::view(reinterpret_cast<double*>(buffer), newShape, newStrides, AMSResourceType::AMS_HOST);
  else if (inputDType == AMSDType::AMS_INT32)
    return AMSTensor::view(reinterpret_cast<int32_t*>(buffer), newShape, newStrides, AMSResourceType::AMS_HOST);
  else if (inputDType == AMSDType::AMS_INT64)
    return AMSTensor::view(reinterpret_cast<int64_t*>(buffer), newShape, newStrides, AMSResourceType::AMS_HOST);
  throw std::runtime_error("Unsupported dtype in concat");
}

template AMSTensor AMSTensor::create<float>(ams::ArrayRef<IntDimType>,
                                            ams::ArrayRef<IntDimType>,
                                            AMSResourceType);
template AMSTensor AMSTensor::create<double>(ams::ArrayRef<IntDimType>,
                                             ams::ArrayRef<IntDimType>,
                                             AMSResourceType);
template AMSTensor AMSTensor::create<int32_t>(ams::ArrayRef<IntDimType>,
                                              ams::ArrayRef<IntDimType>,
                                              AMSResourceType);
template AMSTensor AMSTensor::create<int64_t>(ams::ArrayRef<IntDimType>,
                                              ams::ArrayRef<IntDimType>,
                                              AMSResourceType);

template AMSTensor AMSTensor::view<float>(float*,
                                          ams::ArrayRef<IntDimType>,
                                          ams::ArrayRef<IntDimType>,
                                          AMSResourceType);
template AMSTensor AMSTensor::view<double>(double*,
                                           ams::ArrayRef<IntDimType>,
                                           ams::ArrayRef<IntDimType>,
                                           AMSResourceType);
template AMSTensor AMSTensor::view<int32_t>(int32_t*,
                                            ams::ArrayRef<IntDimType>,
                                            ams::ArrayRef<IntDimType>,
                                            AMSResourceType);
template AMSTensor AMSTensor::view<int64_t>(int64_t*,
                                            ams::ArrayRef<IntDimType>,
                                            ams::ArrayRef<IntDimType>,
                                            AMSResourceType);

template AMSTensor AMSTensor::view<const float>(const float*,
                                                ams::ArrayRef<IntDimType>,
                                                ams::ArrayRef<IntDimType>,
                                                AMSResourceType);
template AMSTensor AMSTensor::view<const double>(const double*,
                                                 ams::ArrayRef<IntDimType>,
                                                 ams::ArrayRef<IntDimType>,
                                                 AMSResourceType);
template AMSTensor AMSTensor::view<const int32_t>(const int32_t*,
                                                  ams::ArrayRef<IntDimType>,
                                                  ams::ArrayRef<IntDimType>,
                                                  AMSResourceType);
template AMSTensor AMSTensor::view<const int64_t>(const int64_t*,
                                                  ams::ArrayRef<IntDimType>,
                                                  ams::ArrayRef<IntDimType>,
                                                  AMSResourceType);
