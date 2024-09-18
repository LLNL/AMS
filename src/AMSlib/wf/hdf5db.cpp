/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "wf/basedb.hpp"

using namespace ams::db;

hid_t hdf5DB::getDataSet(hid_t group,
                         std::string dName,
                         hid_t dataType,
                         const size_t Chunk)
{
  // Our datasets a.t.m are 1-D vectors
  const int nDims = 1;
  // We always start from 0
  hsize_t dims = 0;
  hid_t dset = -1;

  int exists = H5Lexists(group, dName.c_str(), H5P_DEFAULT);

  if (exists > 0) {
    dset = H5Dopen(group, dName.c_str(), H5P_DEFAULT);
    HDF5_ERROR(dset);
    // We are assuming symmetrical data sets a.t.m
    if (totalElements == 0) {
      hid_t dspace = H5Dget_space(dset);
      const int ndims = H5Sget_simple_extent_ndims(dspace);
      hsize_t dims[ndims];
      H5Sget_simple_extent_dims(dspace, dims, NULL);
      totalElements = dims[0];
    }
    return dset;
  } else {
    // We will extend the data-set size, so we use unlimited option
    hsize_t maxDims = H5S_UNLIMITED;
    hid_t fileSpace = H5Screate_simple(nDims, &dims, &maxDims);
    HDF5_ERROR(fileSpace);

    hid_t pList = H5Pcreate(H5P_DATASET_CREATE);
    HDF5_ERROR(pList);

    herr_t ec = H5Pset_layout(pList, H5D_CHUNKED);
    HDF5_ERROR(ec);

    // cDims impacts performance considerably.
    // TODO: Align this with the caching mechanism for this option to work
    // out.
    hsize_t cDims = Chunk;
    H5Pset_chunk(pList, nDims, &cDims);
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
  }
  return dset;
}


void hdf5DB::createDataSets(size_t numElements,
                            const size_t numIn,
                            const size_t numOut)
{
  for (int i = 0; i < numIn; i++) {
    hid_t dSet =
        getDataSet(HFile, std::string("input_") + std::to_string(i), HDType);
    HDIsets.push_back(dSet);
  }

  for (int i = 0; i < numOut; i++) {
    hid_t dSet =
        getDataSet(HFile, std::string("output_") + std::to_string(i), HDType);
    HDOsets.push_back(dSet);
  }

  if (storePredicate()) {
    pSet = getDataSet(HFile, "predicate", H5T_NATIVE_HBOOL);
  }
}

template <typename TypeValue>
void hdf5DB::writeDataToDataset(std::vector<hid_t>& dsets,
                                std::vector<TypeValue*>& data,
                                size_t numElements)
{
  int index = 0;
  for (auto* I : data) {
    writeVecToDataset(dsets[index++],
                      static_cast<void*>(I),
                      numElements,
                      HDType);
  }
}

void hdf5DB::writeVecToDataset(hid_t dSet,
                               void* data,
                               size_t elements,
                               hid_t DType)
{
  const int nDims = 1;
  hsize_t dims = elements;
  hsize_t start;
  hsize_t count;
  hid_t memSpace = H5Screate_simple(nDims, &dims, NULL);
  HDF5_ERROR(memSpace);

  dims = totalElements + elements;
  H5Dset_extent(dSet, &dims);

  hid_t fileSpace = H5Dget_space(dSet);
  HDF5_ERROR(fileSpace);

  // Data set starts at offset totalElements
  start = totalElements;
  // And we append additional elements
  count = elements;
  // Select hyperslab
  herr_t err = H5Sselect_hyperslab(
      fileSpace, H5S_SELECT_SET, &start, NULL, &count, NULL);
  HDF5_ERROR(err);

  H5Dwrite(dSet, DType, memSpace, fileSpace, H5P_DEFAULT, data);
  H5Sclose(fileSpace);
}


template <typename TypeValue>
void hdf5DB::_store(size_t num_elements,
                    std::vector<TypeValue*>& inputs,
                    std::vector<TypeValue*>& outputs,
                    bool* predicate)
{
  CALIPER(CALI_MARK_BEGIN("HDF5_STORE");)
  if (isDouble<TypeValue>::default_value())
    HDType = H5T_NATIVE_DOUBLE;
  else
    HDType = H5T_NATIVE_FLOAT;


  CFATAL(HDF5DB,
         storePredicate() && predicate == nullptr,
         "DB Configured to store predicates, predicate is not provided")


  DBG(DB,
      "DB of type %s stores %ld elements of input/output dimensions (%lu, "
      "%lu)",
      type().c_str(),
      num_elements,
      inputs.size(),
      outputs.size())
  const size_t num_in = inputs.size();
  const size_t num_out = outputs.size();

  if (HDIsets.empty()) {
    createDataSets(num_elements, num_in, num_out);
  }

  CFATAL(HDF5DB,
         (HDIsets.size() != num_in || HDOsets.size() != num_out),
         "The data dimensionality is different than the one in the "
         "DB")

  writeDataToDataset(HDIsets, inputs, num_elements);
  writeDataToDataset(HDOsets, outputs, num_elements);

  if (storePredicate() && predicate != nullptr) {
    writeVecToDataset(pSet,
                      static_cast<void*>(predicate),
                      num_elements,
                      H5T_NATIVE_HBOOL);
  }

  totalElements += num_elements;
  CALIPER(CALI_MARK_END("HDF5_STORE");)
}


hdf5DB::hdf5DB(std::string path, std::string fn, uint64_t rId, bool predicate)
    : FileDB(path, fn, predicate ? ".debug.h5" : ".h5", rId),
      predicateStore(predicate)
{
  std::error_code ec;
  bool exists = fs::exists(this->fn);
  this->checkError(ec);

  if (exists)
    HFile = H5Fopen(this->fn.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  else
    HFile = H5Fcreate(this->fn.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
  HDF5_ERROR(HFile);
  totalElements = 0;
  HDType = -1;
}


hdf5DB::~hdf5DB()
{
  DBG(DB, "Closing File: %s %s", type().c_str(), this->fn.c_str())
  // HDF5 Automatically closes all opened fds at exit of application.
  //    herr_t err = H5Fclose(HFile);
  //    HDF5_ERROR(err);
}

void hdf5DB::store(size_t num_elements,
                   std::vector<float*>& inputs,
                   std::vector<float*>& outputs,
                   bool* predicate)
{
  if (HDType == -1) {
    HDType = H5T_NATIVE_FLOAT;
  }

  CFATAL(HDF5DB,
         HDType != H5T_NATIVE_FLOAT,
         "Database %s initialized to work on 'float' received different "
         "datatypes",
         fn.c_str());

  _store(num_elements, inputs, outputs, predicate);
}


void hdf5DB::store(size_t num_elements,
                   std::vector<double*>& inputs,
                   std::vector<double*>& outputs,
                   bool* predicate)
{
  if (HDType == -1) {
    HDType = H5T_NATIVE_DOUBLE;
  }
  _store(num_elements, inputs, outputs, predicate);
}
