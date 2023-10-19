/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <AMS.h>

#include <iostream>
#include <ml/surrogate.hpp>
#include <umpire/ResourceManager.hpp>
#include <umpire/Umpire.hpp>
#include <vector>
#include <wf/data_handler.hpp>
#include <wf/resource_manager.hpp>

#define SIZE (32L * 1024L + 3L)

template <typename T>
void inference(char *path, int device, AMSResourceType resource)
{
  using namespace ams;
  SurrogateModel<T> model(path, !device);

  std::vector<const T *> inputs;
  std::vector<T *> outputs;

  for (int i = 0; i < 2; i++)
    inputs.push_back(ams::ResourceManager::allocate<T>(SIZE, resource));

  for (int i = 0; i < 4; i++)
    outputs.push_back(ams::ResourceManager::allocate<T>(SIZE, resource));

  model.evaluate(
      SIZE, inputs.size(), outputs.size(), inputs.data(), outputs.data());


  for (int i = 0; i < 2; i++)
    ResourceManager::deallocate(const_cast<T*>(inputs[i]), resource);

  for (int i = 0; i < 4; i++)
    ResourceManager::deallocate(outputs[i], resource);
}

int main(int argc, char *argv[])
{
  using namespace ams;
  using data_handler = DataHandler<double>;
  auto &rm = umpire::ResourceManager::getInstance();
  int use_device = std::atoi(argv[1]);
  char *model_path = argv[2];
  char *data_type = argv[3];

  AMSSetupAllocator(AMSResourceType::HOST);
  AMSResourceType resource = AMSResourceType::HOST;
  if (use_device == 1) {
    AMSSetupAllocator(AMSResourceType::DEVICE);
    AMSSetDefaultAllocator(AMSResourceType::DEVICE);
    resource = AMSResourceType::DEVICE;
  }

  inference<double>(model_path, use_device, resource);

  return 0;
}
