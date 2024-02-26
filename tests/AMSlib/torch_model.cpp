/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <AMS.h>

#include <cstring>
#include <iostream>
#include <ml/surrogate.hpp>
#include <umpire/ResourceManager.hpp>
#include <umpire/Umpire.hpp>
#include <vector>
#include <wf/resource_manager.hpp>

#define SIZE (32L * 1024L + 3L)

template <typename T>
void inference(SurrogateModel<T> &model, AMSResourceType resource)
{
  using namespace ams;

  std::vector<const T *> inputs;
  std::vector<T *> outputs;
  auto &ams_rm = ams::ResourceManager::getInstance();

  for (int i = 0; i < 2; i++)
    inputs.push_back(ams_rm.allocate<T>(SIZE, resource));

  for (int i = 0; i < 4; i++)
    outputs.push_back(ams_rm.allocate<T>(SIZE, resource));

  model.evaluate(
      SIZE, inputs.size(), outputs.size(), inputs.data(), outputs.data());


  for (int i = 0; i < 2; i++)
    ams_rm.deallocate(const_cast<T *>(inputs[i]), resource);

  for (int i = 0; i < 4; i++)
    ams_rm.deallocate(outputs[i], resource);
}

int main(int argc, char *argv[])
{
  using namespace ams;
  auto &ams_rm = ams::ResourceManager::getInstance();
  int use_device = std::atoi(argv[1]);
  char *model_path = argv[2];
  char *data_type = argv[3];

  AMSResourceType resource = AMSResourceType::HOST;
  if (use_device == 1) {
    resource = AMSResourceType::DEVICE;
  }

  ams_rm.init();

  if (std::strcmp("double", data_type) == 0) {
    std::shared_ptr<SurrogateModel<double>> model =
        SurrogateModel<double>::getInstance(model_path, resource);
    assert(model->is_double());
    inference<double>(*model, resource);
  } else if (std::strcmp("single", data_type) == 0) {
    std::shared_ptr<SurrogateModel<float>> model =
        SurrogateModel<float>::getInstance(model_path, resource);
    assert(!model->is_double());
    inference<float>(*model, resource);
  }

  return 0;
}
