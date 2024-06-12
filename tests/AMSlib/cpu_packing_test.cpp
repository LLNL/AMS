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
  auto& rm = ams::ResourceManager::getInstance();
  rm.init();
  if (device == 0) {
    AMSResourceType resource = AMSResourceType::AMS_HOST;
    bool* predicate = rm.allocate<bool>(SIZE, resource);
    double* dense = rm.allocate<double>(SIZE, resource);
    double* sparse = rm.allocate<double>(SIZE, resource);
    double* rsparse = rm.allocate<double>(SIZE, resource);

    initPredicate(predicate, sparse, SIZE);
    std::vector<const double*> s_data({const_cast<const double*>(sparse)});
    std::vector<double*> sr_data({rsparse});
    std::vector<double*> d_data({dense});
    int elements;

    for (int flag = 0; flag < 2; flag++) {
      elements =
          data_handler::pack(resource, predicate, size, s_data, d_data, flag);

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

    rm.deallocate(predicate, AMSResourceType::AMS_HOST);
    rm.deallocate(dense, AMSResourceType::AMS_HOST);
    rm.deallocate(sparse, AMSResourceType::AMS_HOST);
    rm.deallocate(rsparse, AMSResourceType::AMS_HOST);
  } else if (device == 1) {
    AMSResourceType resource = AMSResourceType::AMS_DEVICE;
    bool* h_predicate = rm.allocate<bool>(SIZE, AMSResourceType::AMS_HOST);
    double* h_dense = rm.allocate<double>(SIZE, AMSResourceType::AMS_HOST);
    double* h_sparse = rm.allocate<double>(SIZE, AMSResourceType::AMS_HOST);
    double* h_rsparse = rm.allocate<double>(SIZE, AMSResourceType::AMS_HOST);

    initPredicate(h_predicate, h_sparse, SIZE);

    bool* predicate = rm.allocate<bool>(SIZE, resource);
    double* dense = rm.allocate<double>(SIZE, resource);
    double* sparse = rm.allocate<double>(SIZE, resource);
    double* rsparse = rm.allocate<double>(SIZE, resource);
    int* reindex = rm.allocate<int>(SIZE, resource);

    rm.copy(h_predicate,
            AMSResourceType::AMS_HOST,
            predicate,
            AMSResourceType::AMS_DEVICE,
            SIZE);
    rm.copy(h_sparse,
            AMSResourceType::AMS_HOST,
            sparse,
            AMSResourceType::AMS_DEVICE,
            SIZE);

    std::vector<const double*> s_data({const_cast<const double*>(sparse)});
    std::vector<double*> sr_data({rsparse});
    std::vector<double*> d_data({dense});

    for (int flag = 0; flag < 2; flag++) {
      int elements;
      elements =
          data_handler::pack(resource, predicate, size, s_data, d_data, flag);

      if (elements != (SIZE + flag) / 2) {
        std::cout << "Did not compute dense number correctly(" << elements
                  << ")\n";
        return 1;
      }

      rm.copy(dense,
              AMSResourceType::AMS_DEVICE,
              h_dense,
              AMSResourceType::AMS_HOST,
              elements);

      if (verify(h_dense, elements, flag)) {
        std::cout << "Dense elements do not have the correct values\n";
        return 1;
      }

      data_handler::unpack(resource, predicate, size, d_data, sr_data, flag);

      rm.copy(rsparse,
              AMSResourceType::AMS_DEVICE,
              h_rsparse,
              AMSResourceType::AMS_HOST,
              size);

      if (verify(h_predicate, h_sparse, h_rsparse, size, flag)) {
        //      for ( int k = 0; k < SIZE; k++){
        //        std::cout << k << " " << h_sparse[k] << " " << h_rsparse[k] <<
        //        "\n";
        //      }
        std::cout << "Unpacking packed data does not match initial values\n";
        return 1;
      }
    }

    rm.deallocate(predicate, AMSResourceType::AMS_DEVICE);
    rm.deallocate(h_predicate, AMSResourceType::AMS_HOST);
    rm.deallocate(dense, AMSResourceType::AMS_DEVICE);
    rm.deallocate(h_dense, AMSResourceType::AMS_HOST);
    rm.deallocate(sparse, AMSResourceType::AMS_DEVICE);
    rm.deallocate(h_sparse, AMSResourceType::AMS_HOST);
    rm.deallocate(rsparse, AMSResourceType::AMS_DEVICE);
    rm.deallocate(h_rsparse, AMSResourceType::AMS_HOST);
    rm.deallocate(reindex, AMSResourceType::AMS_DEVICE);
  }

  return 0;
}
