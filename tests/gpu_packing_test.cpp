#include <cstring>
#include <iostream>
#include <umpire/Umpire.hpp>
#include <vector>
#include "wf/utilities.hpp"
#include "utils/data_handler.hpp"

#define SIZE (32*1024 +3)

using namespace AMS::utilities;

void initPredicate(bool *ptr, double *data, int size){
  for (int i = 0 ; i < size  ; i++){
    ptr[i] = i%2 == 0;
    data[i] = i;
  }
}

int verify(double *dense, int size){
  for (int i = 0; i < size; i++){
    if ( dense[i] != i*2 ){
      return 1;
    }
  }
  return 0;
}

int verify(bool *pred, double *d1, double *d2, int size){
  for (int i = 0; i < size; i++){
    if (pred[i] && d1[i] != d2[i] ){
      std::cout<< i << " " << d1[i] << " " << d2[i] << "\n";
      return 1;
    }
  }
  return 0;
}

int main(int argc, char* argv[]) {
    using data_handler = DataHandler<double>;
    auto& rm = umpire::ResourceManager::getInstance();
    auto host_alloc_name = AMS::utilities::getHostAllocatorName();
    auto device_alloc_name = AMS::utilities::getDeviceAllocatorName();
    const size_t size = SIZE;
    int use_reindex  = std::atoi(argv[1]);

    rm.makeAllocator<umpire::strategy::QuickPool, true>(host_alloc_name, rm.getAllocator("HOST"));
    rm.makeAllocator<umpire::strategy::QuickPool, true>(device_alloc_name,
                                                        rm.getAllocator("DEVICE"));
    setDefaultDataAllocator(AMSDevice::DEVICE);

    bool* h_predicate = static_cast<bool*>(
            AMS::utilities::allocate(sizeof(bool) * SIZE, AMSDevice::HOST));
    double* h_dense = static_cast<double*>(AMS::utilities::allocate(sizeof(double)*SIZE, AMSDevice::HOST));
    double* h_sparse = static_cast<double*>(AMS::utilities::allocate(sizeof(double)*SIZE, AMSDevice::HOST));
    double* h_rsparse = static_cast<double*>(AMS::utilities::allocate(sizeof(double)*SIZE, AMSDevice::HOST));

    initPredicate(h_predicate, h_sparse, SIZE);

    bool* predicate = static_cast<bool*>(
            AMS::utilities::allocate(sizeof(bool) * SIZE));
    double* dense = static_cast<double*>(AMS::utilities::allocate(sizeof(double)*SIZE));
    double* sparse = static_cast<double*>(AMS::utilities::allocate(sizeof(double)*SIZE));
    double* rsparse = static_cast<double*>(AMS::utilities::allocate(sizeof(double)*SIZE));
    int* reindex = static_cast<int*>(AMS::utilities::allocate(sizeof(int)*SIZE));

    rm.copy(predicate, h_predicate);
    rm.copy(sparse, h_sparse);

    std::vector<double *> s_data({sparse});
    std::vector<double *> sr_data({rsparse});
    std::vector<double *> d_data({dense});

    int elements;
    if ( use_reindex )
      elements = data_handler::pack(predicate, reindex, size, s_data, d_data);
    else
      elements = data_handler::pack(predicate, size, s_data, d_data);

    if (elements != (SIZE+1) / 2 ){
      std::cout << "Did not compute dense number correctly(" << elements <<")\n";
      return 1;
    }

    rm.copy(h_dense, dense);
    if ( verify(h_dense, elements) ){
      std::cout << "Dense elements do not have the correct values\n";
      return 1;
    }
    
    if ( use_reindex )
      data_handler::unpack(reindex, elements, d_data, sr_data);
    else
      data_handler::unpack(predicate, size, d_data, sr_data);
    
    rm.copy(h_rsparse, rsparse);
    if ( verify ( h_predicate, h_sparse, h_rsparse, size ) ){
      std::cout<< "Unpacking packed data does not match initial values\n";
      return 1;
    }

    deallocate(predicate);
    deallocate(h_predicate, AMSDevice::HOST);
    deallocate(dense);
    deallocate(h_dense, AMSDevice::HOST);
    deallocate(sparse);
    deallocate(h_sparse, AMSDevice::HOST);
    deallocate(rsparse);
    deallocate(h_rsparse, AMSDevice::HOST);
    deallocate(reindex);

    return 0;
}
