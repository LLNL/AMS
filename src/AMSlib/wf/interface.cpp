
#include <stdexcept>
#include <string>
#include <vector>

#include "AMS.h"
#include "AMSTensor.hpp"
#include "wf/workflow.hpp"

using namespace ams;

#if defined(__AMS_ENABLE_TORCH__)
#include <ATen/ops/from_blob.h>
#include <c10/core/DeviceType.h>
#include <c10/util/SmallVector.h>
#include <torch/torch.h>

static AMSResourceType torchDeviceToAMSDevice(c10::DeviceType dType)
{
  switch (dType) {
    case c10::DeviceType::CUDA:
      return AMSResourceType::AMS_DEVICE;
    case c10::DeviceType::HIP:
      return AMSResourceType::AMS_DEVICE;
    case c10::DeviceType::CPU:
      return AMSResourceType::AMS_HOST;
    default:
      return AMSResourceType::AMS_UNKNOWN;
  }
  return AMSResourceType::AMS_UNKNOWN;
}

static AMSDType torchDTypeToAMSType(torch::Dtype dtype)
{
  static const std::unordered_map<torch::Dtype, AMSDType> dtypeMap = {
      {torch::kFloat32, AMSDType::AMS_SINGLE},
      {torch::kFloat, AMSDType::AMS_SINGLE},  // Alias for float32
      {torch::kFloat64, AMSDType::AMS_DOUBLE},
      {torch::kDouble, AMSDType::AMS_DOUBLE},  // Alias for float64
      {torch::kInt32, AMSDType::AMS_INT32},
      {torch::kInt64, AMSDType::AMS_INT64},
      {torch::kBool, AMSDType::AMS_UNKNOWN_TYPE},
      {torch::kUInt8, AMSDType::AMS_UNKNOWN_TYPE},
      {torch::kInt8, AMSDType::AMS_UNKNOWN_TYPE},
      {torch::kHalf, AMSDType::AMS_UNKNOWN_TYPE},
      {torch::kBFloat16, AMSDType::AMS_UNKNOWN_TYPE}};

  return dtypeMap.count(dtype) ? dtypeMap.at(dtype)
                               : AMSDType::AMS_UNKNOWN_TYPE;
}

static c10::DeviceType amsToTorchDevice(const ams::AMSResourceType resource)
{
  if (resource == ams::AMSResourceType::AMS_HOST)
    return c10::DeviceType::CPU;
  else if (resource == ams::AMSResourceType::AMS_DEVICE)
#if defined(__AMS_ENABLE_CUDA__)
    return c10::DeviceType::CUDA;
#elif defined(__AMS_ENABLE_HIP__)
    return c10::DeviceType::CUDA;
#endif

  throw std::runtime_error("Unknown ams resource type");
  return c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES;
}

static c10::ScalarType amsToTorchDType(const ams::AMSDType dType)
{
  if (dType == ams::AMSDType::AMS_SINGLE)
    return torch::kFloat32;
  else if (dType == ams::AMSDType::AMS_DOUBLE)
    return torch::kFloat64;
  else if (dType == ams::AMSDType::AMS_INT32)
    return torch::kInt32;
  else if (dType == ams::AMSDType::AMS_INT64)
    return torch::kInt64;

  throw std::runtime_error("Unknown ams data type");
  return torch::kHalf;
}

// single tensor
static ams::AMSTensor torchToAMSTensorView(torch::Tensor& tensor)
{
  auto dType = torchDTypeToAMSType(tensor.scalar_type());
  auto rType = torchDeviceToAMSDevice(tensor.device().type());

  auto shapes = ams::ArrayRef(tensor.sizes().begin(), tensor.sizes().size());
  auto strides =
      ams::ArrayRef(tensor.strides().begin(), tensor.strides().size());

  switch (dType) {
    case AMSDType::AMS_SINGLE:
      return AMSTensor::view(tensor.data_ptr<float>(), shapes, strides, rType);

    case AMSDType::AMS_DOUBLE:
      return AMSTensor::view(tensor.data_ptr<double>(), shapes, strides, rType);

    case AMSDType::AMS_INT32:
      return AMSTensor::view(tensor.data_ptr<int32_t>(),
                             shapes,
                             strides,
                             rType);

ams::SmallVector<ams::AMSTensor> torchToAMSTensors(
    ams::MutableArrayRef<torch::Tensor> tensorVector)
{
  ams::SmallVector<ams::AMSTensor> ams_tensors;
  for (auto& tensor : tensorVector) {
    ams_tensors.push_back(torchToAMSTensorView(tensor));
  }
  return ams_tensors;
}

static ams::SmallVector<torch::Tensor> amsToTorchTensors(
    const ams::SmallVector<ams::AMSTensor>& amsTensorVector)
{
  ams::SmallVector<torch::Tensor> ams_tensors;
  for (auto& tensor : amsTensorVector) {
    // We should be able to completely remove these conversion by using some template "magic."
    // I will leave these for later though
    auto dType = amsToTorchDType(tensor.dtype());
    auto deviceType = amsToTorchDevice(tensor.location());
    // In both cases, I am effectively only forwarding the pointer of begin/end to ams.
    // this is a cheap operating. It should boil down to: shapes.start = tensor.sizes.start, shapes.end = tensor.sizes.end;
    c10::SmallVector<long> shapes(tensor.shape().begin(), tensor.shape().end());
    c10::SmallVector<long> strides(tensor.strides().begin(),
                                   tensor.strides().end());
    ams_tensors.push_back(torch::from_blob(
        tensor.data_ptr(),
        shapes,
        strides,
        torch::TensorOptions().dtype(dType).device(deviceType)));
  }
  return std::move(ams_tensors);
}

void callApplication(ams::DomainLambda CallBack,
                     ams::MutableArrayRef<torch::Tensor> Ins,
                     ams::MutableArrayRef<torch::Tensor> InOuts,
                     ams::MutableArrayRef<torch::Tensor> Outs)
{
  auto AMSIns = torchToAMSTensors(Ins);
  auto AMSInOuts = torchToAMSTensors(InOuts);
  auto AMSOuts = torchToAMSTensors(Outs);
  CallBack(AMSIns, AMSInOuts, AMSOuts);
  return;
}

void callAMS(ams::AMSWorkflow* executor,
             DomainLambda Physics,
             const ams::SmallVector<ams::AMSTensor>& ins,
             ams::SmallVector<ams::AMSTensor>& inouts,
             ams::SmallVector<ams::AMSTensor>& outs)
{
  ams::SmallVector<torch::Tensor> tins = amsToTorchTensors(ins);
  ams::SmallVector<torch::Tensor> tinouts = amsToTorchTensors(inouts);
  ams::SmallVector<torch::Tensor> touts = amsToTorchTensors(outs);

  executor->evaluate(Physics, tins, tinouts, touts);
}

// ============================================================================
// Graph surrogate execution (in ams namespace for friend access)
// ============================================================================

namespace ams
{

bool tryGraphSurrogate(AMSWorkflow* executor,
                       const AMSHomogeneousGraph& graph,
                       AMSHomogeneousGraphFields& outputs)
{
  // Check if model is available
  if (!executor || !executor->MLModel) {
    return false;
  }

  try {
    // Convert AMS graph → Torch Dict[str, Tensor]
    auto torch_graph =
        amsToTorchHomogeneousGraph(graph,
                                   executor->MLModel->torch_device,
                                   executor->MLModel->torch_dtype);

    // Call model forward pass
    std::vector<torch::jit::IValue> inputs = {torch::jit::IValue(torch_graph)};
    auto result = executor->MLModel->module.forward(inputs);

    auto dict = result.toGenericDict();
    outputs.node_fields.clear();
    outputs.edge_fields.clear();
    outputs.global_fields.clear();
    const int64_t num_nodes = graph.node_features.shape()[0];
    const int64_t num_edges = graph.edge_index.shape()[1];

    for (const auto& item : dict) {
      const std::string key = item.key().toStringRef();
      const auto parts = splitKey(key, ':');
      if (parts.size() != 2 || parts[0].empty() || parts[1].empty()) {
        throw std::runtime_error("Malformed homogeneous graph output key '" +
                                 key +
                                 "'. Expected 'node:<field>', 'edge:<field>', "
                                 "or "
                                 "'global:<field>'.");
      }

      torch::Tensor tensor = item.value().toTensor();
      if (parts[0] == "node") {
        requireOutputFirstDim(tensor, num_nodes, key, "node");
        outputs.node_fields.insert(parts[1], torchToAMSTensorCopy(tensor));
      } else if (parts[0] == "edge") {
        requireOutputFirstDim(tensor, num_edges, key, "edge");
        outputs.edge_fields.insert(parts[1], torchToAMSTensorCopy(tensor));
      } else if (parts[0] == "global") {
        requireGlobalOutputShape(tensor, key);
        outputs.global_fields.insert(parts[1], torchToAMSTensorCopy(tensor));
      } else {
        throw std::runtime_error("Malformed homogeneous graph output key '" +
                                 key +
                                 "'. Expected entity prefix 'node', 'edge', or "
                                 "'global'.");
      }
    }

    return true;
  } catch (const std::exception& e) {
    throw std::runtime_error(
        std::string("Homogeneous graph surrogate failed: ") + e.what());
  }
}

bool tryGraphSurrogate(AMSWorkflow* executor,
                       const AMSHeterogeneousGraph& graph,
                       AMSHeterogeneousGraphFields& outputs)
{
  // Check if model is available
  if (!executor || !executor->MLModel) {
    return false;
  }

  try {
    // Convert AMS graph → Torch GenericDict
    auto torch_graph = amsToTorchHeterogeneousGraph(graph);

    // Call model forward pass
    std::vector<torch::jit::IValue> inputs = {torch::jit::IValue(torch_graph)};
    auto result = executor->MLModel->module.forward(inputs);

    auto dict = result.toGenericDict();
    outputs.node_stores.clear();
    outputs.edge_stores.clear();
    outputs.global_store.clear();
    for (const auto& item : dict) {
      const std::string key = item.key().toStringRef();
      const auto parts = splitKey(key, ':');
      torch::Tensor tensor = item.value().toTensor();

      if (parts.size() == 3 && parts[0] == "node" && !parts[1].empty() &&
          !parts[2].empty()) {
        const auto* store = graph.findNodeStore(parts[1]);
        if (!store || store->empty()) {
          throw std::runtime_error("Heterogeneous graph output key '" + key +
                                   "' references an unknown or empty node "
                                   "store.");
        }
        const auto& reference_tensor = store->begin()->second;
        if (reference_tensor.shape().size() < 1) {
          throw std::runtime_error("Heterogeneous graph output key '" + key +
                                   "' cannot infer node count from a scalar "
                                   "input field.");
        }
        const int64_t num_nodes = reference_tensor.shape()[0];
        requireOutputFirstDim(tensor, num_nodes, key, "node");
        outputs.getOrCreateNodeStore(parts[1]).insert(parts[2],
                                                      torchToAMSTensorCopy(
                                                          tensor));
      } else if (parts.size() == 3 && parts[0] == "edge" && !parts[1].empty() &&
                 !parts[2].empty()) {
        EdgeType edge_type = edgeTypeFromString(parts[1]);
        const auto* store = graph.findEdgeStore(edge_type);
        if (!store) {
          throw std::runtime_error("Heterogeneous graph output key '" + key +
                                   "' references an unknown edge store.");
        }
        const AMSTensor* edge_index = findTensor(*store, "edge_index");
        if (!edge_index || edge_index->shape().size() != 2) {
          throw std::runtime_error("Heterogeneous graph edge output key '" +
                                   key +
                                   "' requires an input edge_index tensor with "
                                   "shape [2, E].");
        }
        requireOutputFirstDim(tensor, edge_index->shape()[1], key, "edge");
        outputs.getOrCreateEdgeStore(edge_type).insert(parts[2],
                                                       torchToAMSTensorCopy(
                                                           tensor));
      } else if (parts.size() == 2 && parts[0] == "global" &&
                 !parts[1].empty()) {
        requireGlobalOutputShape(tensor, key);
        outputs.global_store.insert(parts[1], torchToAMSTensorCopy(tensor));
      } else {
        throw std::runtime_error("Malformed heterogeneous graph output key '" +
                                 key +
                                 "'. Expected 'node:<node_type>:<field>', "
                                 "'edge:<src>__<rel>__<dst>:<field>', or "
                                 "'global:<field>'.");
      }
    }

    return true;
  } catch (const std::exception& e) {
    throw std::runtime_error(
        std::string("Heterogeneous graph surrogate failed: ") + e.what());
  }
}

}  // namespace ams

// ============================================================================
// Graph-based callAMS overloads
// ============================================================================

void callAMS(ams::AMSWorkflow* executor,
             DomainLambda Physics,
             const ams::SmallVector<ams::AMSTensor>& ins,
             ams::SmallVector<ams::AMSTensor>& inouts,
             ams::SmallVector<ams::AMSTensor>& outs)
{
  // Delegate to public evaluate method (mirrors tensor pattern)
  executor->evaluate(Physics, graph_input, outputs);
}

void callAMS(ams::AMSWorkflow* executor,
             ams::HeterogeneousGraphDomainFn Physics,
             const ams::AMSHeterogeneousGraph& graph_input,
             ams::AMSHeterogeneousGraphFields& outputs)
{
  // Delegate to public evaluate method (mirrors tensor pattern)
  executor->evaluate(Physics, graph_input, outputs);
}

#else

void callAMS(ams::AMSWorkflow* executor,
             DomainLambda Physics,
             const ams::SmallVector<ams::AMSTensor>& ins,
             ams::SmallVector<ams::AMSTensor>& inouts,
             ams::SmallVector<ams::AMSTensor>& outs)
{
  // In training mode, we can directlty use AMSTensor, no conversion needed
  executor->evaluate(Physics, ins, inouts, outs);
}

#endif // __AMS_ENABLE_TORCH__
