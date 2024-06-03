#include <AMS.h>
#include <ATen/core/interned_strings.h>
#include <c10/core/TensorOptions.h>
#include <torch/types.h>

#include <cstring>
#include <iostream>
#include <ml/surrogate.hpp>
#include <umpire/ResourceManager.hpp>
#include <umpire/Umpire.hpp>
#include <vector>
#include <wf/resource_manager.hpp>

#define SIZE (32L)

template <typename T>
bool inference(SurrogateModel<T> &model,
               AMSResourceType resource,
               std::string update_path)
{
  using namespace ams;

  std::vector<const T *> inputs;
  std::vector<T *> outputs;
  auto &ams_rm = ams::ResourceManager::getInstance();

  for (int i = 0; i < 2; i++)
    inputs.push_back(ams_rm.allocate<T>(SIZE, resource));

  for (int i = 0; i < 4 * 2; i++)
    outputs.push_back(ams_rm.allocate<T>(SIZE, resource));

  for (int repeat = 0; repeat < 2; repeat++) {
    model.evaluate(
        SIZE, inputs.size(), 4, inputs.data(), &(outputs.data()[repeat * 4]));
    if (repeat == 0) model.update(update_path);
  }

  // Verify
  bool errors = false;
  for (int i = 0; i < 4; i++) {
    T *first_model_out = outputs[i];
    T *second_model_out = outputs[i + 4];
    if (resource == AMSResourceType::AMS_DEVICE) {
      first_model_out = ams_rm.allocate<T>(SIZE, AMSResourceType::AMS_HOST);
      second_model_out = ams_rm.allocate<T>(SIZE, AMSResourceType::AMS_HOST);
      ams_rm.copy(outputs[i],
                  resource,
                  first_model_out,
                  AMSResourceType::AMS_HOST,
                  SIZE);
      ams_rm.copy(outputs[i + 4],
                  resource,
                  second_model_out,
                  AMSResourceType::AMS_HOST,
                  SIZE);
    }

    for (int j = 0; j < SIZE; j++) {
      if (first_model_out[j] != 1.0) {
        errors = true;
        std::cout << "One Model " << first_model_out << " " << j << " "
                  << first_model_out[j] << "\n";
      }
      if (second_model_out[j] != 0.0) {
        std::cout << "Zero Model " << second_model_out << " " << j << " "
                  << second_model_out[j] << "\n";
        errors = true;
      }
    }

    if (resource == AMSResourceType::AMS_DEVICE) {
      ams_rm.deallocate(first_model_out, AMSResourceType::AMS_HOST);
      ams_rm.deallocate(second_model_out, AMSResourceType::AMS_HOST);
    }
  }

  for (int i = 0; i < 2; i++)
    ams_rm.deallocate(const_cast<T *>(inputs[i]), resource);

  for (int i = 0; i < 4 * 2; i++)
    ams_rm.deallocate(outputs[i], resource);

  return errors;
}


int main(int argc, char *argv[])
{
  using namespace ams;
  auto &ams_rm = ams::ResourceManager::getInstance();
  int use_device = std::atoi(argv[1]);
  std::string data_type(argv[2]);
  std::string zero_model(argv[3]);
  std::string one_model(argv[4]);

  AMSResourceType resource = AMSResourceType::AMS_HOST;
  if (use_device == 1) {
    resource = AMSResourceType::AMS_DEVICE;
  }


  ams_rm.init();
  int ret = 0;
  if (data_type.compare("double") == 0) {
    std::shared_ptr<SurrogateModel<double>> model =
        SurrogateModel<double>::getInstance(one_model, resource);
    assert(model->is_double());
    ret = inference<double>(*model, resource, zero_model);
  } else if (data_type.compare("single") == 0) {
    std::shared_ptr<SurrogateModel<float>> model =
        SurrogateModel<float>::getInstance(one_model, resource);
    assert(!model->is_double());
    ret = inference<float>(*model, resource, zero_model);
  }
  std::cout << "Zero Model is " << zero_model << "\n";
  std::cout << "One Model is " << one_model << "\n";
  return ret;
}
