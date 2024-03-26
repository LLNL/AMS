#include <AMS.h>

#include <cstring>
#include <iostream>
#include <ml/surrogate.hpp>
#include <ml/uq.hpp>
#include <umpire/ResourceManager.hpp>
#include <umpire/Umpire.hpp>
#include <vector>
#include <wf/resource_manager.hpp>

#define SIZE (32L * 1024L + 3L)

template <typename T>
void model(UQ<T> &model,
           AMSResourceType resource,
           int num_inputs,
           int num_outputs)
{
  std::vector<const T *> inputs;
  std::vector<T *> outputs;
  auto &ams_rm = ams::ResourceManager::getInstance();

  for (int i = 0; i < num_inputs; i++)
    inputs.push_back(ams_rm.allocate<T>(SIZE, resource));

  for (int i = 0; i < num_outputs; i++)
    outputs.push_back(ams_rm.allocate<T>(SIZE, resource));

  bool *predicates = ams_rm.allocate<bool>(SIZE, resource);

  std::cout << "We are calling evaluate\n";
  model.evaluate(SIZE, inputs, outputs, predicates);


  for (int i = 0; i < num_inputs; i++)
    ams_rm.deallocate(const_cast<T *>(inputs[i]), resource);

  for (int i = 0; i < num_outputs; i++)
    ams_rm.deallocate(outputs[i], resource);

  ams_rm.deallocate(predicates, resource);
}


int main(int argc, char *argv[])
{
  using namespace ams;
  auto &ams_rm = ResourceManager::getInstance();
  int use_device = std::atoi(argv[1]);
  char *model_path = argv[2];
  char *data_type = argv[3];
  int num_inputs = std::atoi(argv[4]);
  int num_outputs = std::atoi(argv[5]);
  const AMSUQPolicy uq_policy = static_cast<AMSUQPolicy>(std::atoi(argv[6]));
  float threshold = std::atof(argv[7]);

  std::cout << "Executing on device " << use_device << "\n";

  AMSResourceType resource = AMSResourceType::HOST;
  if (use_device == 1) {
    resource = AMSResourceType::DEVICE;
  }

  ams_rm.init();


  if (std::strcmp("double", data_type) == 0) {
    UQ<double> UQModel(resource, uq_policy, nullptr, -1, model_path, threshold);
    model(UQModel, resource, num_inputs, num_outputs);
  } else if (std::strcmp("single", data_type) == 0) {
    UQ<float> UQModel(resource, uq_policy, nullptr, -1, model_path, threshold);
    model(UQModel, resource, num_inputs, num_outputs);
  }

  return 0;
}
