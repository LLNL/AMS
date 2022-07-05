#include <cstring>
#include <iostream>
#include <umpire/Umpire.hpp>
#include <vector>
#include "utils/allocator.hpp"
#include "utils/utils_data.hpp"

#define SIZE (32*1024 +3)

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
    using namespace ams;
    using data_handler = DataHandler<double>;
    auto& rm = umpire::ResourceManager::getInstance();
    const size_t size = SIZE;
    int use_reindex  = std::atoi(argv[1]);

    ams::ResourceManager::setup(true);

    bool* h_predicate = 
            ams::ResourceManager::allocate<bool>(SIZE, ResourceManager::ResourceType::HOST);
    double* h_dense = ams::ResourceManager::allocate<double>(SIZE, ResourceManager::ResourceType::HOST);
    double* h_sparse = ams::ResourceManager::allocate<double>(SIZE, ResourceManager::ResourceType::HOST);
    double* h_rsparse = ams::ResourceManager::allocate<double>(SIZE, ResourceManager::ResourceType::HOST);

    initPredicate(h_predicate, h_sparse, SIZE);

    bool* predicate = ams::ResourceManager::allocate<bool>(SIZE);
    double* dense = ams::ResourceManager::allocate<double>(SIZE);
    double* sparse = ams::ResourceManager::allocate<double>(SIZE);
    double* rsparse = ams::ResourceManager::allocate<double>(SIZE);
    int* reindex = ams::ResourceManager::allocate<int>(SIZE);

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

    ams::ResourceManager::deallocate(predicate);
    ams::ResourceManager::deallocate(h_predicate, ResourceManager::ResourceType::HOST);
    ams::ResourceManager::deallocate(dense);
    ams::ResourceManager::deallocate(h_dense, ResourceManager::ResourceType::HOST);
    ams::ResourceManager::deallocate(sparse);
    ams::ResourceManager::deallocate(h_sparse, ResourceManager::ResourceType::HOST);
    ams::ResourceManager::deallocate(rsparse);
    ams::ResourceManager::deallocate(h_rsparse, ResourceManager::ResourceType::HOST);
    ams::ResourceManager::deallocate(reindex);

    return 0;
}
