#include <ATen/ops/from_blob.h>
#include <c10/core/DeviceType.h>
#include <c10/util/SmallVector.h>
#include <torch/torch.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "AMS.h"
#include "AMSTensor.hpp"
#include "wf/workflow.hpp"

using namespace ams;

// dtype/device helpers
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

    case AMSDType::AMS_INT64:
      return AMSTensor::view(tensor.data_ptr<int64_t>(),
                             shapes,
                             strides,
                             rType);

    default:
      throw std::runtime_error("torchToAMSTensorView: unsupported Torch dtype");
  }
}

static ams::AMSTensor torchToAMSTensorCopy(const torch::Tensor& tensor)
{
  torch::Tensor src = tensor.detach();
  if (!src.is_contiguous()) {
    src = src.contiguous();
  }

  auto dType = torchDTypeToAMSType(src.scalar_type());
  auto rType = torchDeviceToAMSDevice(src.device().type());
  if (rType == AMSResourceType::AMS_UNKNOWN) {
    throw std::runtime_error("torchToAMSTensorCopy: unsupported Torch device");
  }

  ams::SmallVector<ams::AMSTensor::IntDimType> shapes;
  ams::SmallVector<ams::AMSTensor::IntDimType> strides;
  for (const auto dim : src.sizes()) {
    shapes.push_back(static_cast<ams::AMSTensor::IntDimType>(dim));
  }
  for (const auto stride : src.strides()) {
    strides.push_back(static_cast<ams::AMSTensor::IntDimType>(stride));
  }

  auto& rm = ams::ResourceManager::getInstance();
  switch (dType) {
    case AMSDType::AMS_SINGLE: {
      auto out = AMSTensor::create<float>(shapes, strides, rType);
      rm.copy(
          src.data_ptr<float>(), rType, out.data<float>(), rType, src.numel());
      return out;
    }
    case AMSDType::AMS_DOUBLE: {
      auto out = AMSTensor::create<double>(shapes, strides, rType);
      rm.copy(src.data_ptr<double>(),
              rType,
              out.data<double>(),
              rType,
              src.numel());
      return out;
    }
    case AMSDType::AMS_INT32: {
      auto out = AMSTensor::create<int32_t>(shapes, strides, rType);
      rm.copy(src.data_ptr<int32_t>(),
              rType,
              out.data<int32_t>(),
              rType,
              src.numel());
      return out;
    }
    case AMSDType::AMS_INT64: {
      auto out = AMSTensor::create<int64_t>(shapes, strides, rType);
      rm.copy(src.data_ptr<int64_t>(),
              rType,
              out.data<int64_t>(),
              rType,
              src.numel());
      return out;
    }
    default:
      throw std::runtime_error("torchToAMSTensorCopy: unsupported Torch dtype");
  }
}

static torch::Tensor amsToTorchTensorView(const ams::AMSTensor& tensor)
{
  auto dType = amsToTorchDType(tensor.dType());
  auto deviceType = amsToTorchDevice(tensor.location());

  c10::SmallVector<long> shapes(tensor.shape().begin(), tensor.shape().end());
  c10::SmallVector<long> strides(tensor.strides().begin(),
                                 tensor.strides().end());

  return torch::from_blob(tensor.raw_data(),
                          shapes,
                          strides,
                          torch::TensorOptions().dtype(dType).device(
                              deviceType));
}

// flat containers
static ams::SmallVector<ams::AMSTensor> torchToAMSTensors(
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
  ams::SmallVector<torch::Tensor> torch_tensors;
  for (const auto& tensor : amsTensorVector) {
    torch_tensors.push_back(amsToTorchTensorView(tensor));
  }
  return torch_tensors;
}

static ams::AMSTensorMap torchDictToAMSTensorMap(
    const c10::Dict<std::string, torch::Tensor>& dict)
{
  ams::AMSTensorMap out;

  for (const auto& item : dict) {
    const std::string name = item.key();
    torch::Tensor tensor = item.value();

    out.emplace(name, torchToAMSTensorView(tensor));
  }

  return out;
}

static c10::Dict<std::string, torch::Tensor> amsTensorMapToTorchDict(
    const ams::AMSTensorMap& store)
{
  c10::Dict<std::string, torch::Tensor> out;

  for (const auto& [name, tensor] : store) {
    out.insert(name, amsToTorchTensorView(tensor));
  }

  return out;
}

static torch::Tensor amsTensorToTorchModelInput(const ams::AMSTensor& tensor,
                                                c10::DeviceType model_device,
                                                torch::Dtype model_dtype,
                                                bool preserve_dtype)
{
  torch::Tensor out = amsToTorchTensorView(tensor);
  torch::Dtype dtype = preserve_dtype ? out.scalar_type() : model_dtype;
  if (out.device().type() != model_device || out.scalar_type() != dtype) {
    out = out.to(model_device, dtype);
  }
  return out;
}

static void requireOutputFirstDim(const torch::Tensor& tensor,
                                  int64_t expected,
                                  const std::string& key,
                                  const std::string& entity)
{
  if (tensor.dim() < 1) {
    throw std::runtime_error("Graph surrogate output '" + key + "' for " +
                             entity + " fields must have rank at least 1.");
  }
  if (tensor.sizes()[0] != expected) {
    throw std::runtime_error("Graph surrogate output '" + key + "' for " +
                             entity + " fields has first dimension " +
                             std::to_string(tensor.sizes()[0]) + ", expected " +
                             std::to_string(expected) + ".");
  }
}

static void requireGlobalOutputShape(const torch::Tensor& tensor,
                                     const std::string& key)
{
  if (tensor.dim() != 2 || tensor.sizes()[0] != 1) {
    throw std::runtime_error("Graph surrogate output '" + key +
                             "' for global fields must have shape [1, F].");
  }
}

static std::vector<std::string> splitKey(const std::string& key, char delim)
{
  std::vector<std::string> parts;
  std::size_t start = 0;
  while (true) {
    std::size_t pos = key.find(delim, start);
    if (pos == std::string::npos) {
      parts.push_back(key.substr(start));
      break;
    }
    parts.push_back(key.substr(start, pos - start));
    start = pos + 1;
  }
  return parts;
}

// key helpers
static c10::Dict<std::string, torch::Tensor> toStringTensorDict(
    const c10::IValue& value)
{
  c10::Dict<std::string, torch::Tensor> out;

  auto generic = value.toGenericDict();
  for (const auto& kv : generic) {
    out.insert(kv.key().toStringRef(), kv.value().toTensor());
  }

  return out;
}

static c10::impl::GenericDict toStringIValueDict(const c10::IValue& value)
{
  c10::impl::GenericDict out(c10::StringType::get(), c10::AnyType::get());

  auto generic = value.toGenericDict();
  for (const auto& kv : generic) {
    out.insert(kv.key().toStringRef(), kv.value());
  }

  return out;
}

// homogeneous graphs
static c10::Dict<std::string, torch::Tensor> amsToTorchHomogeneousGraph(
    const ams::AMSHomogeneousGraph& g,
    c10::DeviceType model_device,
    torch::Dtype model_dtype)
{
  g.validate();

  c10::Dict<std::string, torch::Tensor> out;
  out.insert("node_features",
             amsTensorToTorchModelInput(
                 g.node_features, model_device, model_dtype, false));
  out.insert("edge_index",
             amsTensorToTorchModelInput(
                 g.edge_index, model_device, model_dtype, true));
  out.insert("edge_features",
             amsTensorToTorchModelInput(
                 g.edge_features, model_device, model_dtype, false));
  if (g.global_features.shape()[0] != 0) {
    out.insert("global_features",
               amsTensorToTorchModelInput(
                   g.global_features, model_device, model_dtype, false));
  }
  return out;
}

// heterogeneous graphs
static std::unordered_map<std::string, ams::AMSTensorMap>
torchDictToAMSNodeStores(const c10::impl::GenericDict& dict)
{
  std::unordered_map<std::string, ams::AMSTensorMap> out;

  for (const auto& item : dict) {
    out.emplace(std::string(item.key().toStringRef()),
                torchDictToAMSTensorMap(toStringTensorDict(item.value())));
  }

  return out;
}

static std::unordered_map<ams::EdgeType, ams::AMSTensorMap, ams::EdgeTypeHash>
torchDictToAMSEdgeStores(const c10::impl::GenericDict& dict)
{
  std::unordered_map<ams::EdgeType, ams::AMSTensorMap, ams::EdgeTypeHash> out;

  for (const auto& item : dict) {
    ams::EdgeType edge_type =
        edgeTypeFromString(std::string(item.key().toStringRef()));

    out.emplace(std::move(edge_type),
                torchDictToAMSTensorMap(toStringTensorDict(item.value())));
  }

  return out;
}

static ams::AMSHeterogeneousGraph torchToAMSHeterogeneousGraph(
    const c10::IValue& value)
{
  auto g = value.toGenericDict();

  ams::AMSHeterogeneousGraph out;

  c10::IValue nodes_ivalue;
  c10::IValue edges_ivalue;
  c10::IValue global_ivalue;

  bool has_nodes = false;
  bool has_edges = false;
  bool has_global = false;

  for (const auto& kv : g) {
    const auto key = kv.key().toStringRef();
    if (key == "node_stores") {
      nodes_ivalue = kv.value();
      has_nodes = true;
    } else if (key == "edge_stores") {
      edges_ivalue = kv.value();
      has_edges = true;
    } else if (key == "global_store") {
      global_ivalue = kv.value();
      has_global = true;
    }
  }

  if (!has_nodes) {
    throw std::runtime_error(
        "torchToAMSHeterogeneousGraph: missing node_stores");
  }
  if (!has_edges) {
    throw std::runtime_error(
        "torchToAMSHeterogeneousGraph: missing edge_stores");
  }
  if (!has_global) {
    throw std::runtime_error(
        "torchToAMSHeterogeneousGraph: missing global_store");
  }

  out.node_stores = torchDictToAMSNodeStores(toStringIValueDict(nodes_ivalue));
  out.edge_stores = torchDictToAMSEdgeStores(toStringIValueDict(edges_ivalue));
  out.global_store = torchDictToAMSTensorMap(toStringTensorDict(global_ivalue));

  return out;
}

static c10::Dict<std::string, c10::Dict<std::string, torch::Tensor>>
amsNodeStoresToTorchDict(
    const std::unordered_map<std::string, ams::AMSTensorMap>& node_stores)
{
  c10::Dict<std::string, c10::Dict<std::string, torch::Tensor>> out;

  for (const auto& [store_name, store] : node_stores) {
    out.insert(store_name, amsTensorMapToTorchDict(store));
  }

  return out;
}

static c10::Dict<std::string, c10::Dict<std::string, torch::Tensor>>
amsEdgeStoresToTorchDict(
    const std::unordered_map<ams::EdgeType,
                             ams::AMSTensorMap,
                             ams::EdgeTypeHash>& edge_stores)
{
  c10::Dict<std::string, c10::Dict<std::string, torch::Tensor>> out;

  for (const auto& [edge_type, store] : edge_stores) {
    out.insert(ams::edgeTypeToString(edge_type),
               amsTensorMapToTorchDict(store));
  }

  return out;
}

static c10::impl::GenericDict amsToTorchHeterogeneousGraph(
    const ams::AMSHeterogeneousGraph& g)
{
  c10::impl::GenericDict out(c10::StringType::get(), c10::AnyType::get());

  out.insert("node_stores", amsNodeStoresToTorchDict(g.node_stores));
  out.insert("edge_stores", amsEdgeStoresToTorchDict(g.edge_stores));
  out.insert("global_store", amsTensorMapToTorchDict(g.global_store));

  return out;
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
             ams::HomogeneousGraphDomainFn Physics,
             const ams::AMSHomogeneousGraph& graph_input,
             ams::AMSHomogeneousGraphFields& outputs)
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
