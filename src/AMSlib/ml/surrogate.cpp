#include "surrogate.hpp"

#include <c10/core/ScalarTypeToTypeMeta.h>
#include <torch/script.h>

#include <experimental/filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "AMS.h"
#include "wf/debug.h"
#include "wf/utils.hpp"

using namespace ams;
static std::string getDTypeAsString(torch::Dtype dtype)
{
  if (dtype == torch::kFloat32) return "float32";
  if (dtype == torch::kFloat64) return "float64";
  if (dtype == torch::kInt32) return "int32";
  if (dtype == torch::kInt64) return "int64";
  if (dtype == torch::kBool) return "bool";
  if (dtype == torch::kUInt8) return "uint8";
  if (dtype == torch::kInt8) return "int8";

  // Add other types as needed
  return "unknown";
}

static std::string getAMSDTypeAsString(AMSDType dType)
{
  if (dType == AMS_SINGLE)
    return "float32";
  else if (dType == AMS_DOUBLE)
    return "float64";
  return "unknown";
}

static std::string getAMSResourceTypeAsString(AMSResourceType res)
{
  if (res == ams::AMS_DEVICE)
    return "device";
  else if (res == ams::AMS_HOST)
    return "host";
  return "unknown-device";
}


SurrogateModel::InstanceMap& SurrogateModel::instances()
{
  // Torch owns process-global state; avoid destroying cached modules during
  // C++ static teardown after Torch/C10 globals may already be gone.
  static auto* Instances = new InstanceMap();
  return *Instances;
}


void SurrogateModel::clearCache() { instances().clear(); }


SurrogateModel::SurrogateModel(std::string& model_path)
    : _model_path(model_path)
{

  std::experimental::filesystem::path Path(model_path);
  std::error_code ec;

  if (!std::experimental::filesystem::exists(Path, ec)) {
    AMS_FATAL(Surrogate,
              "Path to Surrogate Model {} Does not "
              "exist",
              model_path)
  }

  try {
    module = torch::jit::load(model_path);
  } catch (const c10::Error& e) {
    AMS_FATAL(Surrogate, "Error opening {}", model_path);
  }

  auto method_ptr = module.find_method("get_ams_info");
  if (!method_ptr) {
    AMS_FATAL(Surrogate,
              "The Surrogate {} is not a valid "
              "AMSModel",
              model_path);
  }

  torch::IValue meta_ivalue = module.run_method("get_ams_info");
  auto meta_dict = meta_ivalue.toGenericDict();

  for (const auto& item : meta_dict) {
    std::string key = item.key().toStringRef();
    std::string value = item.value().toStringRef();
    if (key == "ams_type") {
      std::tie(model_dtype, torch_dtype) = convertModelDataType(value);
    } else if (key == "ams_device") {
      std::tie(model_device, torch_device) = convertModelResourceType(value);
    }
  }

  AMS_CFATAL(SurrogateModel,
             model_dtype == ams::AMS_UNKNOWN_TYPE ||
                 model_device == ams::AMSResourceType::AMS_UNKNOWN,
             "Model has unknown datatype or device");

  AMS_DBG(SurrogateModel,
          "Loaded model with type {} on device {}",
          getAMSDTypeAsString(model_dtype),
          getAMSResourceTypeAsString(model_device));
}


std::tuple<ams::AMSDType, torch::Dtype> SurrogateModel::getModelDataType() const
{
  return std::make_tuple(model_dtype, torch_dtype);
}

std::tuple<AMSResourceType, torch::DeviceType> SurrogateModel::
    getModelResourceType() const
{
  return std::make_tuple(model_device, torch_device);
}

std::tuple<AMSResourceType, torch::DeviceType> SurrogateModel::
    convertModelResourceType(std::string& value)
{

  if (value == "cpu") {
    return std::make_tuple(AMS_HOST, c10::DeviceType::CPU);
  } else if (value == "cuda") {
    return std::make_tuple(AMS_DEVICE, c10::DeviceType::CUDA);
  } else if (value == "hip") {
    return std::make_tuple(AMS_DEVICE, c10::DeviceType::CUDA);
  }
  // If no parameters or buffers are found, default to unknown
  AMS_FATAL(Surrogate,
            "Cannot determine device type of model "
            "{}",
            value);
  return std::make_tuple(AMS_UNKNOWN,
                         c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);
}

std::tuple<AMSDType, torch::Dtype> SurrogateModel::convertModelDataType(
    std::string& type)
{
  AMSDType dParamType = AMSDType::AMS_DOUBLE;
  torch::Dtype torchType = at::kDouble;
  if (type == "float32") {
    return std::make_tuple(AMSDType::AMS_SINGLE, at ::kFloat);
  } else if (type == "float64") {
    return std::make_tuple(AMSDType::AMS_DOUBLE, at ::kDouble);
  }

  AMS_FATAL(Surrogate, "unknown data type of model {}", type);

  return std::make_tuple(dParamType, torchType);
}

std::tuple<torch::Tensor, torch::Tensor> SurrogateModel::_evaluate(
    torch::Tensor& inputs,
    float threshold)
{
  if (inputs.dtype() != torch_dtype) {
    throw std::runtime_error(
        "Received inputs of wrong dType. Model "
        "is expecting " +
        getDTypeAsString(torch::typeMetaToScalarType(inputs.dtype())) +
        " and model is " + getDTypeAsString(torch_dtype));
  }
  c10::InferenceMode guard(true);
  auto out = module.forward({inputs});
  AMS_DBG(Surrogate, "Addess of module is {}", static_cast<void*>(&module));

  at::Tensor prediction =
      out.toTuple()->elements()[0].toTensor().set_requires_grad(false).detach();
  at::Tensor uncertainty =
      out.toTuple()->elements()[1].toTensor().set_requires_grad(false).detach();

  // Handle per-output uncertainty by taking the maximum across output dimensions
  // If uncertainty has more than 1 dimension and the last dimension > 1,
  // reduce it to per-sample uncertainty by taking max across outputs
  if (uncertainty.dim() > 1 && uncertainty.sizes()[uncertainty.dim() - 1] > 1) {
    AMS_DBG(Surrogate,
            "UQ model returned per-output uncertainty {}, reducing to "
            "per-sample by taking max",
            shapeToString(uncertainty));
    uncertainty = std::get<0>(uncertainty.max(/*dim=*/-1, /*keepdim=*/true));
  }

  auto predicate = uncertainty < threshold;
  return std::make_tuple(std::move(prediction), std::move(predicate));
}


std::tuple<torch::Tensor, torch::Tensor> SurrogateModel::evaluate(
    ams::MutableArrayRef<at::Tensor> Inputs,
    float threshold,
    torch::Tensor* packedWorkspace,
    bool* reusedWorkspace)
{
  if (reusedWorkspace) *reusedWorkspace = false;
  if (Inputs.size() == 0) {
    throw std::invalid_argument(
        "Input Vector should always contain at "
        "least one tensor");
  }

  torch::DeviceType InputDevice = Inputs[0].device().type();
  torch::Dtype InputDType = torch::typeMetaToScalarType(Inputs[0].dtype());
  auto CAxis = Inputs[0].sizes().size() - 1;
  if (Inputs[0].dim() == 0)
    throw std::invalid_argument("Surrogate inputs must have rank at least one");

  // Verify input/device matching
  for (auto& In : Inputs) {
    if (InputDevice != In.device().type()) {
      throw std::invalid_argument(
          "Unsupported feature, application "
          "domain tensors are on different "
          "devices\n");
    }
    if (InputDType != torch::typeMetaToScalarType(In.dtype())) {
      throw std::invalid_argument(
          "Unsupported feature, application "
          "domain tensors have different data "
          "types\n");
    }
    if (In.dim() != Inputs[0].dim())
      throw std::invalid_argument("Surrogate input ranks differ");
    for (int64_t axis = 0; axis < In.dim() - 1; ++axis)
      if (In.size(axis) != Inputs[0].size(axis))
        throw std::invalid_argument(
            "Surrogate input shapes differ outside the feature axis");
  }

  torch::Tensor ITensor;
  const bool direct = Inputs.size() == 1 && InputDevice == torch_device &&
                      InputDType == torch_dtype;
  if (direct) {
    ITensor = Inputs[0];
  } else {
    std::vector<int64_t> packedShape(Inputs[0].sizes().begin(),
                                     Inputs[0].sizes().end());
    int64_t features = 0;
    for (const auto& input : Inputs)
      features += input.size(CAxis);
    packedShape[CAxis] = features;
    torch::Tensor localWorkspace;
    torch::Tensor& workspace =
        packedWorkspace ? *packedWorkspace : localWorkspace;
    auto options =
        torch::TensorOptions().dtype(torch_dtype).device(torch_device);
    if (!workspace.defined() || workspace.scalar_type() != torch_dtype ||
        workspace.device().type() != torch_device) {
      workspace = torch::empty(packedShape, options);
    } else {
      void* oldPointer = workspace.data_ptr();
      workspace.resize_(packedShape);
      if (reusedWorkspace)
        *reusedWorkspace = workspace.data_ptr() == oldPointer;
    }
    int64_t offset = 0;
    for (const auto& input : Inputs) {
      const int64_t width = input.size(CAxis);
      workspace.narrow(CAxis, offset, width).copy_(input);
      offset += width;
    }
    ITensor = workspace;
  }
  AMS_DBG(Surrogate, "Input concatenated tensor is {}", shapeToString(ITensor));

  auto [OTensor, Predicate] = _evaluate(ITensor, threshold);
  if (InputDevice != torch_device) {
    OTensor = OTensor.to(InputDevice);
    Predicate = Predicate.to(InputDevice);
  }
  return std::make_tuple(std::move(OTensor), std::move(Predicate));
}
