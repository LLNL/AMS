/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <H5Ipublic.h>
#include <H5Tpublic.h>
#include <H5public.h>


#include <experimental/filesystem>
#include <stdexcept>
#include <sstream>
#include <vector>
 
#include "AMSTensor.hpp"
#include "ArrayRef.hpp"
#include "wf/basedb.hpp"


using namespace ams::db;
using namespace ams;

namespace
{
namespace fs = std::experimental::filesystem;
}

static std::string SmallVectorToString(ams::MutableArrayRef<hsize_t> shape)
{
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    oss << shape[i];
    if (i < shape.size() - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}

static std::string tensorSizeToString(ArrayRef<AMSTensor::IntDimType> shape)
{
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    oss << shape[i];
    if (i < shape.size() - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}

static std::string amsDTypeToString(AMSDType dtype)
{
  switch (dtype) {
    case AMSDType::AMS_SINGLE: return "float32";
    case AMSDType::AMS_DOUBLE: return "float64";
    case AMSDType::AMS_INT32:  return "int32";
    case AMSDType::AMS_INT64:  return "int64";
    default:                   return "unknown dtype";
  }
}

static hid_t amsDTypeToHDF5Type(AMSDType dtype)
{
  switch (dtype) {
    case AMSDType::AMS_SINGLE: return H5T_NATIVE_FLOAT;
    case AMSDType::AMS_DOUBLE: return H5T_NATIVE_DOUBLE;
    case AMSDType::AMS_INT32:  return H5T_NATIVE_INT;
    case AMSDType::AMS_INT64:  return H5T_NATIVE_LONG;
    default:                   return H5T_NO_CLASS;
  }
}

hid_t hdf5DB::getDataSet(hid_t group,
                         std::string dName,
                         SmallVector<hsize_t>& currentShape,
                         ArrayRef<AMSTensor::IntDimType> Shape,
                         hid_t dataType,
                         const size_t Chunk)
{
  const int nDims = Shape.size();
  currentShape.resize(nDims);
  currentShape.assign(nDims, 0);
  // We always start from 0
  hsize_t dims = 0;
  hid_t dset = -1;

  int exists = H5Lexists(group, dName.c_str(), H5P_DEFAULT);

  if (exists > 0) {
    dset = H5Dopen(group, dName.c_str(), H5P_DEFAULT);
    HDF5_ERROR(dset);
    // We are assuming symmetrical data sets a.t.m
    hid_t dspace = H5Dget_space(dset);
    const int file_ndims = H5Sget_simple_extent_ndims(dspace);
    if (file_ndims != nDims) {
      throw std::runtime_error(
          "File system file with current tensor shape to not match");
    }
    hsize_t dims[nDims];
    H5Sget_simple_extent_dims(dspace, dims, NULL);
    currentShape[0] = dims[0];
    return dset;
  }

  // We will extend the data-set size, so we use unlimited option
  hsize_t max_dims[Shape.size()];
  hsize_t initial_shape[Shape.size()];
  for (int i = 0; i < Shape.size(); i++) {
    max_dims[i] = static_cast<hsize_t>(Shape[i]);
    initial_shape[i] = 0;
  }
  max_dims[0] = H5S_UNLIMITED;
  hid_t fileSpace = H5Screate_simple(nDims, initial_shape, max_dims);
  HDF5_ERROR(fileSpace);

  hid_t pList = H5Pcreate(H5P_DATASET_CREATE);
  HDF5_ERROR(pList);

  herr_t ec = H5Pset_layout(pList, H5D_CHUNKED);
  HDF5_ERROR(ec);

  // cDims impacts performance considerably.
  // TODO: Align this with the caching mechanism for this option to work
  // out.
  max_dims[0] = Chunk;
  H5Pset_chunk(pList, nDims, max_dims);
  dset = H5Dcreate(group,
                   dName.c_str(),
                   dataType,
                   fileSpace,
                   H5P_DEFAULT,
                   pList,
                   H5P_DEFAULT);
  HDF5_ERROR(dset);
  H5Sclose(fileSpace);
  H5Pclose(pList);
  return dset;
}


void hdf5DB::createDataSets(ArrayRef<AMSTensor::IntDimType> InShapes, ArrayRef<AMSTensor::IntDimType> OutShapes)
{
  HDIset = getDataSet(HFile, "input_data", currentInputShape, InShapes, HDType);

  HDOset =
      getDataSet(HFile, "output_data", currentOutputShape, OutShapes, HDType);
}

void hdf5DB::writeDataToDataset(ams::MutableArrayRef<hsize_t> currentShape,
                                hid_t& dset,
                                const AMSTensor& tensor_data)
{
  herr_t status;

  // // Ensure tensor is contiguous
  // torch::Tensor tensor_contiguous = tensor_data.contiguous();

  // TODO: make sure it is contiguous
  std::vector<hsize_t> tensor_dims(tensor_data.sizes().begin(),
                                   tensor_data.sizes().end());

  int rank = tensor_dims.size();

  // Initialize currentShape if it's empty (e.g., first write or reopening an existing file)
  if (currentShape.empty()) {
    hid_t fileSpace = H5Dget_space(dset);
    if (fileSpace < 0) {
      throw std::runtime_error("Failed to get dataspace from dataset.");
    }
    if (H5Sget_simple_extent_dims(fileSpace, currentShape.data(), NULL) < 0) {
      H5Sclose(fileSpace);
      throw std::runtime_error("Failed to retrieve dataset dimensions.");
    }
    H5Sclose(fileSpace);
  }

  // Create a memory representation for the data to be stored
  hid_t memSpace = H5Screate_simple(rank, tensor_dims.data(), NULL);
  if (memSpace < 0) {
    throw std::runtime_error("Failed to create memory dataspace.");
  }

  // Prepare the dataset for new data
  ams::SmallVector<hsize_t> newShape(tensor_dims.begin(), tensor_dims.end());

  newShape[0] += currentShape[0];  // Update the first dimension

  status = H5Dset_extent(dset, newShape.data());
  if (status < 0) {
    throw std::runtime_error("Failed to extend dataset's dimensions.");
  }


  // Refresh fileSpace after extending
  hid_t fileSpace = H5Dget_space(dset);
  if (fileSpace < 0) {
    throw std::runtime_error(
        "Failed to get refreshed dataspace after extending dataset.");
  }

  // Debugging: Check dimensions of fileSpace
  std::vector<hsize_t> file_dims(rank);
  H5Sget_simple_extent_dims(fileSpace, file_dims.data(), NULL);

  // Select hyperslab
  herr_t err = H5Sselect_hyperslab(fileSpace,
                                   H5S_SELECT_SET,
                                   currentShape.data(),
                                   NULL,
                                   tensor_dims.data(),
                                   NULL);
  if (err < 0) {
    H5Sclose(fileSpace);
    H5Sclose(memSpace);
    throw std::runtime_error("Failed to select hyperslab.");
  }

  // Write the tensor data to the dataset
  status = H5Dwrite(dset,
                    HDType,
                    memSpace,
                    fileSpace,
                    H5P_DEFAULT,
                    tensor_data.data_ptr());
  if (status < 0) {
    throw std::runtime_error("Failed to write data to dataset.");
  }

  // Update currentShape
  currentShape[0] = newShape[0];

  // Close HDF5 objects
  H5Sclose(memSpace);
  H5Sclose(fileSpace);
}


void hdf5DB::_store(const AMSTensor& inputs, const AMSTensor& outputs)
{
  AMS_DBG(DB,
          "DB of type {} stores input/output tensors of  shapes {}, "
          "{}",
          type(),
          tensorSizeToString(inputs.sizes()),
          tensorSizeToString(outputs.sizes()));

  if (HDIset == -1 || HDOset == -1) {
    createDataSets(inputs.sizes(), outputs.sizes());
  }

  writeDataToDataset(currentInputShape, HDIset, inputs);
  writeDataToDataset(currentOutputShape, HDOset, outputs);
  AMS_DBG(DB,
          "DB (file:{}) next elements to be stored at Input:{} Output: {}",
          fn,
          SmallVectorToString(currentOutputShape),
          SmallVectorToString(currentInputShape));
}


hdf5DB::hdf5DB(std::string path, std::string domain_name, uint64_t rId)
    : FileDB(path, domain_name, ".h5", rId), HDOset(-1), HDIset(-1), HDType(-1)
{
  std::error_code ec;
  bool exists = fs::exists(this->fn);
  this->checkError(ec);

  if (exists)
    HFile = H5Fopen(this->fn.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  else {
    HFile = H5Fcreate(this->fn.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t dims[1] = {domain_name.size()};
    hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
    hid_t dataset_id = H5Dcreate(HFile,
                                 "domain_name",
                                 H5T_NATIVE_CHAR,
                                 dataspace_id,
                                 H5P_DEFAULT,
                                 H5P_DEFAULT,
                                 H5P_DEFAULT);
    H5Dwrite(dataset_id,
             H5T_NATIVE_CHAR,
             H5S_ALL,
             H5S_ALL,
             H5P_DEFAULT,
             domain_name.c_str());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
  }
  HDF5_ERROR(HFile);
  HDType = -1;
}


hdf5DB::~hdf5DB()
{
  AMS_DBG(DB, "Closing File: {} {}", type(), this->fn)
  // HDF5 Automatically closes all opened fds at exit of application.
  herr_t err = H5Fclose(HFile);
  // NOTE: WE need to investigate this further. I recall that older HDF5 version may not need to close
  // explicitly their handlers. Cause they keep an internal state.
  HDF5_ERROR(err);
}

void hdf5DB::store(ArrayRef<AMSTensor> Inputs,
                   ArrayRef<AMSTensor> Outputs)
{

  // auto tOptions = torch::TensorOptions()
  //                     .dtype(torch::kFloat32)
  //                     .device(c10::DeviceType::CPU);

  // c10::SmallVector<torch::Tensor> ConvertedInputs(Inputs.begin(), Inputs.end());
  // c10::SmallVector<torch::Tensor> ConvertedOutputs(Outputs.begin(),
  //                                                  Outputs.end());

  // auto inputs =
  //     torch::cat(ConvertedInputs, Inputs[0].sizes().size() - 1).to(tOptions);
  // auto outputs =
  //     torch::cat(ConvertedOutputs, Outputs[0].sizes().size() - 1).to(tOptions);

  // TODO: handle error in better fashion here
  if (Inputs.size() == 0 || Outputs.size() == 0) {
    throw std::invalid_argument("store() requires non-empty input and output tensors");
  }

  // TODO: Check every tensors type constentcy
  AMSDType inputDType = Inputs[0].dtype();
  AMSDType outputDType = Outputs[0].dtype();

  if (inputDType != outputDType) {
    throw std::invalid_argument(
        "Storing into HDF5 database requires all tensors to have the same "
        "datatype. Now they have: " +
        amsDTypeToString(inputDType) + " and " +
        amsDTypeToString(outputDType));
  }

  if (HDType == -1) {
    HDType = amsDTypeToHDF5Type(inputDType);
  }

  if (HDType == -1 || HDType == H5T_NO_CLASS)
    throw std::invalid_argument(
        "Data base can not deduce the data type of the tensors" +
        amsDTypeToString(inputDType) + " and " +
        amsDTypeToString(outputDType));

  auto inputs = AMSTensor::concat(Inputs, inputDType);
  auto outputs = AMSTensor::concat(Outputs, outputDType);

  _store(inputs, outputs);
}
