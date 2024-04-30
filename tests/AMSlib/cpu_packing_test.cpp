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

#define SIZE (3281)

void initPredicate(bool* ptr, std::vector<const double*>& data, int size)
{
  for (auto d_ptr : data){
    double* d = const_cast<double*> (d_ptr);
    for (int i = 0; i < size; i++) {
      ptr[i] = i % 2 == 0;
      d[i] = i;
    }
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
  int dims = std::atoi(argv[2]);
  auto& rm = ams::ResourceManager::getInstance();
  rm.init();
  std::vector<const double*> s_data;
  std::vector<double*> sr_data;
  std::vector<double*> d_data;
  REPORT_MEM_USAGE(Start, "Start")

  AMSResourceType resource = AMSResourceType::HOST;
  if (device == 1)
    resource = AMSResourceType::DEVICE;

  for (int i = 0; i < dims; i++){
    double* dense = rm.allocate<double>(SIZE, resource);
    double* sparse = rm.allocate<double>(SIZE, resource);
    double* rsparse = rm.allocate<double>(SIZE, resource);
    s_data.push_back(const_cast<const double*>(sparse));
    sr_data.push_back(rsparse);
    d_data.push_back(dense);
  }

  bool* predicate = rm.allocate<bool>(SIZE, resource);

  if (device == 0) {
    initPredicate(predicate, s_data, SIZE);
    int elements;

    for (int flag = 0; flag < 2; flag++) {
      elements =
          data_handler::pack(resource, predicate, size, s_data, d_data, flag);

      if (elements != (SIZE + flag) / 2) {
        std::cout << "Did not compute dense number correctly " << elements
                  << "\n";
        return 1;
      }

      for ( auto d : d_data ){
        if (verify(d, elements, flag)) {
          std::cout << "Dense elements do not have the correct values\n";
          return 1;
        }
      }

      data_handler::unpack(resource, predicate, size, d_data, sr_data, flag);

      for ( int i = 0 ; i < dims; i++){
        if (verify(predicate, const_cast<double*>(s_data[i]), sr_data[i], size, flag)) {
          std::cout << "Unpacking packed data does not match initial values\n";
          return 1;
        }
      }
    }

  } else if (device == 1) {
    bool* h_predicate =
        rm.allocate<bool>(SIZE, AMSResourceType::HOST);
    double* h_dense =
        rm.allocate<double>(SIZE, AMSResourceType::HOST);
    double* h_sparse =
        rm.allocate<double>(SIZE, AMSResourceType::HOST);
    double* h_rsparse =
        rm.allocate<double>(SIZE, AMSResourceType::HOST);
    
    std::vector<const double*> tmp({const_cast<const double*>(h_sparse)});
    initPredicate(h_predicate, tmp, SIZE);

    bool* predicate = rm.allocate<bool>(SIZE, resource);
    double* dense = rm.allocate<double>(SIZE, resource);
    double* sparse = rm.allocate<double>(SIZE, resource);
    double* rsparse = rm.allocate<double>(SIZE, resource);

    rm.copy(h_predicate, predicate);
    for ( auto s: s_data ){
      rm.copy(h_sparse, const_cast<double*>(s));
    }

    for (int flag = 0; flag < 2; flag++) {
      int elements;
      elements =
          data_handler::pack(resource, predicate, size, s_data, d_data, flag);

      if (elements != (SIZE + flag) / 2) {
        std::cout << "Did not compute dense number correctly(" << elements
                  << ")\n";
        return 1;
      }
  
      for ( auto d : d_data ){
        rm.copy(d, h_dense);
        if (verify(h_dense, elements, flag)) {
          std::cout << "Dense elements do not have the correct values\n";
          return 1;
        }
      }

      data_handler::unpack(resource, predicate, size, d_data, sr_data, flag);

      for ( int i = 0; i < dims; i++){
        rm.copy(sr_data[i], h_rsparse);
        rm.copy(const_cast<double*>(s_data[i]), h_sparse);
        if (verify(h_predicate, h_sparse, h_rsparse, size, flag)) {
           std::cout << "Unpacking packed data does not match initial values\n";
          return 1;
        }
      }
    }

    rm.deallocate(h_predicate, AMSResourceType::HOST);
    rm.deallocate(h_dense, AMSResourceType::HOST);
    rm.deallocate(h_sparse, AMSResourceType::HOST);
    rm.deallocate(h_rsparse, AMSResourceType::HOST);
  }

  rm.deallocate(predicate, resource);
  for (int i = 0; i < dims; i++){
    rm.deallocate(const_cast<double*>(s_data[i]), resource);
    rm.deallocate(sr_data[i], resource);
    rm.deallocate(d_data[i], resource);
  }
  s_data.clear();
  sr_data.clear();
  d_data.clear();
  cudaDeviceReset();
  REPORT_MEM_USAGE(Finish, "Start")


  return 0;
}
