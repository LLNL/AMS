/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <AMS.h>

#include <cstring>
#include <iostream>
#include <vector>
#include <wf/data_handler.hpp>
#include <wf/resource_manager.hpp>

#define SIZE (32 * 1024 + 1)

void initPredicate(bool* ptr, double* data, int size)
{
  for (int i = 0; i < size; i++) {
    ptr[i] = i % 2 == 0;
    data[i] = i;
  }
}

int verify(double* dense, int size, int flag)
{
  for (int i = 0; i < size; i++) {
    if (dense[i] != (i * 2 + (!flag))) {
      std::cout << i << " Expected " << i * 2 << " gotten " << dense[i] << "\n";
      return 1;
    }
  }
  return 0;
}

int verify(bool* pred, double* d1, double* d2, int size, int flag)
{
  for (int i = 0; i < size; i++) {
    if ((pred[i] == flag) && d1[i] != d2[i]) {
      std::cout << pred[i] << " dense " << d1[i] << " sparse " << d2[i] << "\n";
      return 1;
    }
  }
  return 0;
}

int main(int argc, char* argv[])
{
  using namespace ams;
  using data_handler = DataHandler<double>;
  const size_t size = SIZE;
  int device = std::atoi(argv[1]);
  AMSSetupAllocator(AMSResourceType::HOST);
  if (device == 0) {
    AMSResourceType resource = AMSResourceType::HOST;
    bool* predicate = ams::ResourceManager::allocate<bool>(SIZE, resource);
    double* dense = ams::ResourceManager::allocate<double>(SIZE, resource);
    double* sparse = ams::ResourceManager::allocate<double>(SIZE, resource);
    double* rsparse = ams::ResourceManager::allocate<double>(SIZE, resource);

    initPredicate(predicate, sparse, SIZE);
    std::vector<const double*> s_data({const_cast<const double*>(sparse)});
    std::vector<double*> sr_data({rsparse});
    std::vector<double*> d_data({dense});
    int elements;

    for (int flag = 0; flag < 2; flag++) {
      elements = data_handler::pack(resource, predicate, size, s_data, d_data, flag);

      if (elements != (SIZE + flag) / 2) {
        std::cout << "Did not compute dense number correctly " << elements
                  << "\n";
        return 1;
      }

      if (verify(dense, elements, flag)) {
        std::cout << "Dense elements do not have the correct values\n";
        return 1;
      }

      data_handler::unpack(resource, predicate, size, d_data, sr_data, flag);

      if (verify(predicate, sparse, rsparse, size, flag)) {
        std::cout << "Unpacking packed data does not match initial values\n";
        return 1;
      }
    }

    ResourceManager::deallocate(predicate, AMSResourceType::HOST);
    ResourceManager::deallocate(dense, AMSResourceType::HOST);
    ResourceManager::deallocate(sparse, AMSResourceType::HOST);
    ResourceManager::deallocate(rsparse, AMSResourceType::HOST);
  } else if (device == 1) {
    AMSResourceType resource = AMSResourceType::DEVICE;
    bool* h_predicate =
        ams::ResourceManager::allocate<bool>(SIZE, AMSResourceType::HOST);
    double* h_dense =
        ams::ResourceManager::allocate<double>(SIZE, AMSResourceType::HOST);
    double* h_sparse =
        ams::ResourceManager::allocate<double>(SIZE, AMSResourceType::HOST);
    double* h_rsparse =
        ams::ResourceManager::allocate<double>(SIZE, AMSResourceType::HOST);

    initPredicate(h_predicate, h_sparse, SIZE);

    bool* predicate = ams::ResourceManager::allocate<bool>(SIZE, resource);
    double* dense = ams::ResourceManager::allocate<double>(SIZE, resource);
    double* sparse = ams::ResourceManager::allocate<double>(SIZE, resource);
    double* rsparse = ams::ResourceManager::allocate<double>(SIZE, resource);
    int* reindex = ams::ResourceManager::allocate<int>(SIZE, resource);

    ResourceManager::copy(h_predicate, predicate);
    ResourceManager::copy(h_sparse, sparse);

    std::vector<const double*> s_data({const_cast<const double*>(sparse)});
    std::vector<double*> sr_data({rsparse});
    std::vector<double*> d_data({dense});

    for (int flag = 0; flag < 2; flag++) {
      int elements;
      elements = data_handler::pack(resource, predicate, size, s_data, d_data, flag);

      if (elements != (SIZE + flag) / 2) {
        std::cout << "Did not compute dense number correctly(" << elements
                  << ")\n";
        return 1;
      }

      ams::ResourceManager::copy(dense, h_dense);

      if (verify(h_dense, elements, flag)) {
        std::cout << "Dense elements do not have the correct values\n";
        return 1;
      }

      data_handler::unpack(resource, predicate, size, d_data, sr_data, flag);

      ams::ResourceManager::copy(rsparse, h_rsparse);

      if (verify(h_predicate, h_sparse, h_rsparse, size, flag)) {
        //      for ( int k = 0; k < SIZE; k++){
        //        std::cout << k << " " << h_sparse[k] << " " << h_rsparse[k] <<
        //        "\n";
        //      }
        std::cout << "Unpacking packed data does not match initial values\n";
        return 1;
      }
    }

    ams::ResourceManager::deallocate(predicate, AMSResourceType::DEVICE);
    ams::ResourceManager::deallocate(h_predicate, AMSResourceType::HOST);
    ams::ResourceManager::deallocate(dense, AMSResourceType::DEVICE);
    ams::ResourceManager::deallocate(h_dense, AMSResourceType::HOST);
    ams::ResourceManager::deallocate(sparse, AMSResourceType::DEVICE);
    ams::ResourceManager::deallocate(h_sparse, AMSResourceType::HOST);
    ams::ResourceManager::deallocate(rsparse, AMSResourceType::DEVICE);
    ams::ResourceManager::deallocate(h_rsparse, AMSResourceType::HOST);
    ams::ResourceManager::deallocate(reindex, AMSResourceType::DEVICE);
  }

  return 0;
}
