/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <AMS.h>

#include <climits>
#include <cstring>
#include <iostream>
#include <ml/hdcache.hpp>
#include <umpire/ResourceManager.hpp>
#include <umpire/Umpire.hpp>
#include <vector>
#include <wf/resource_manager.hpp>

template <typename T>
std::vector<const T *> generate_vectors(const int num_clusters,
                                        int elements,
                                        int dims)
{
  std::vector<const T *> v_data;
  auto &rm = ams::ResourceManager::getInstance();
  // This are fixed to mimic the way the faiss was generated
  // The code below generates data values that are either within
  // the distance of the faiss index or just outside of it.
  const T distance = 10.0;
  const T offset = 5.0;
  for (int i = 0; i < dims; i++) {
    T *data =
        rm.allocate<T>(num_clusters * elements, AMSResourceType::AMS_HOST);
    for (int j = 0; j < elements; j++) {
      // Generate a value for every cluster center
      for (int k = 0; k < num_clusters; k++) {
        T tmp = ((T)rand()) / INT_MAX;
        tmp += (k + 1) * num_clusters;
        if ((j % 2) == 0) {
          tmp += offset;
        }
        data[j * num_clusters + k] = tmp;
      }
    }
    v_data.push_back(data);
  }
  return std::move(v_data);
}

template <typename T>
void print_vectors(std::vector<T *> &vec, int num_elements, int num_clusters)
{
  for (int i = 0; i < num_elements; i++) {
    for (int c = 0; c < num_clusters; c++) {
      for (auto v : vec) {
        std::cout << v[i * num_clusters + c] << ":";
      }
      std::cout << "\n";
    }
  }
}


bool validate(const int num_clusters, const int elements, bool *predicates)
{
  bool res = true;
  for (int j = 0; j < elements; j++) {
    // Generate a value for every cluster center
    for (int k = 0; k < num_clusters; k++) {
      if (j % 2 == 0 && predicates[j * num_clusters + k] == true) {
        res = false;
      } else if (j % 2 == 1 && predicates[j * num_clusters + k] == false) {
        res = false;
      }
    }
  }
  return res;
}

template <typename T>
bool do_faiss(std::shared_ptr<HDCache<T>> &index,
              AMSResourceType resource,
              int nClusters,
              int nDims,
              int nElements,
              float threshold)
{

  std::vector<const T *> orig_data =
      generate_vectors<T>(nClusters, nElements, nDims);
  std::vector<const T *> data = orig_data;
  auto &rm = ams::ResourceManager::getInstance();

  bool *predicates = rm.allocate<bool>(nClusters * nElements, resource);

  if (resource == AMSResourceType::AMS_DEVICE) {
    for (int i = 0; i < orig_data.size(); i++) {
      T *d_data = rm.allocate<T>(nClusters * nElements, resource);
      rm.copy(const_cast<T *>(orig_data[i]),
              AMSResourceType::AMS_HOST,
              d_data,
              AMSResourceType::AMS_DEVICE,
              nClusters * nElements);
      data[i] = d_data;
    }
  }


  index->evaluate(nClusters * nElements, data, predicates);

  bool *h_predicates = predicates;

  if (resource == AMSResourceType::AMS_DEVICE) {
    h_predicates =
        rm.allocate<bool>(nClusters * nElements, AMSResourceType::AMS_HOST);
    rm.copy(predicates,
            AMSResourceType::AMS_DEVICE,
            h_predicates,
            AMSResourceType::AMS_HOST,
            nClusters * nElements);
    for (auto d : data) {
      rm.deallocate(const_cast<T *>(d), AMSResourceType::AMS_DEVICE);
    }
    rm.deallocate(predicates, AMSResourceType::AMS_DEVICE);
  }


  for (auto h_d : orig_data)
    rm.deallocate(const_cast<T *>(h_d), AMSResourceType::AMS_HOST);

  bool res = validate(nClusters, nElements, h_predicates);

  rm.deallocate(h_predicates, AMSResourceType::AMS_HOST);
  return res;
}


int main(int argc, char *argv[])
{
  using namespace ams;

  if (argc < 8) {
    std::cerr << "Wrong CLI\n";
    std::cerr << argv[0]
              << " 'use device' 'path to faiss' 'data type (double|float)' "
                 "'UQPolicy (0:Mean, 1:Max)' 'Num Clusters' 'Threshold' "
                 "'number of dimensions' 'num elements'";
    abort();
  }
  auto &rm = umpire::ResourceManager::getInstance();
  int use_device = std::atoi(argv[1]);
  char *faiss_path = argv[2];
  char *data_type = argv[3];
  AMSUQPolicy uq_policy = static_cast<AMSUQPolicy>(std::atoi(argv[4]));
  int nClusters = std::atoi(argv[5]);
  float threshold = std::atoi(argv[6]);
  int nDims = std::atoi(argv[7]);
  int nElements = std::atoi(argv[8]);

  AMSResourceType resource = AMSResourceType::AMS_HOST;
  if (use_device == 1) resource = AMSResourceType::AMS_DEVICE;

  auto &ams_rm = ResourceManager::getInstance();
  ams_rm.init();

  if (std::strcmp("double", data_type) == 0) {
    std::shared_ptr<HDCache<double>> cache = HDCache<double>::getInstance(
        faiss_path, resource, uq_policy, 10, threshold);
    bool result =
        do_faiss(cache, resource, nClusters, nDims, nElements, threshold);
    cache.reset();
    return !result;
  } else if (std::strcmp("single", data_type) == 0) {
    std::shared_ptr<HDCache<float>> cache = HDCache<float>::getInstance(
        faiss_path, resource, uq_policy, 10, threshold);
    bool result =
        do_faiss(cache, resource, nClusters, nDims, nElements, threshold);
    cache.reset();
    return !result;
  }


  return 0;
}
