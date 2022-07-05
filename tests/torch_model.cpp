#include <cstring>
#include <iostream>
#include <umpire/Umpire.hpp>
#include <vector>
#include "utils/allocator.hpp"
#include "utils/utils_data.hpp"
#include "ml/surrogate.hpp"

#define SIZE (32*1024 +3)

template <typename T>
void inference(char *path, int device){
  SurrogateModel<T> model(path, !device);
  std::vector<T*> inputs;
  std::vector<T*> outputs;
  for ( int i = 0; i < 2; i++ )
    inputs.push_back(ams::ResourceManager::allocate<T>(SIZE));

  for ( int i = 0 ; i < 4; i++)
    outputs.push_back(ams::ResourceManager::allocate<T>(SIZE));

  model.Eval(SIZE, inputs, outputs);

  for ( int i = 0; i < 2; i++)
    ams::ResourceManager::deallocate(inputs[i]);

  for ( int i = 0; i < 4; i++)
    ams::ResourceManager::deallocate(outputs[i]);
}

int main(int argc, char* argv[]) {
    using namespace ams;
    using data_handler = DataHandler<double>;
    auto& rm = umpire::ResourceManager::getInstance();
    int use_device = std::atoi(argv[1]);
    char *model_path = argv[2];
    char *data_type = argv[3];

    ams::ResourceManager::setup(use_device);
    if (strcmp("float", data_type) )
      inference<float>(model_path, use_device); 
    else if ( strcmp("double", data_type) )
      inference<double>(model_path, use_device); 

    return 0;
}
