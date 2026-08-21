#include "AMSGraph.hpp"

#include <stdexcept>
#include <string>

namespace ams
{
namespace
{

static std::size_t hashCombine(std::size_t seed, std::size_t value) noexcept
{
  seed ^= value + 0x9e3779b9u + (seed << 6) + (seed >> 2);
  return seed;
}

static bool isFloatingDType(AMSDType dtype) noexcept
{
  return dtype == AMS_SINGLE || dtype == AMS_DOUBLE;
}

static bool isIntegerDType(AMSDType dtype) noexcept
{
  return dtype == AMS_INT64 || dtype == AMS_INT32;
}

static std::string dtypeName(AMSDType dtype)
{
  switch (dtype) {
    case AMS_SINGLE:
      return "float32";
    case AMS_DOUBLE:
      return "float64";
    case AMS_INT32:
      return "int32";
    case AMS_INT64:
      return "int64";
    default:
      return "unknown";
  }
}

static void requireRank(const AMSTensor& tensor,
                        std::size_t rank,
                        const std::string& name)
{
  if (tensor.shape().size() != rank) {
    throw std::runtime_error("AMSHomogeneousGraph " + name + " must be rank " +
                             std::to_string(rank) + ".");
  }
}

static void requireFloating(const AMSTensor& tensor, const std::string& name)
{
  if (!isFloatingDType(tensor.dtype())) {
    throw std::runtime_error("AMSHomogeneousGraph " + name +
                             " must be floating point, got " +
                             dtypeName(tensor.dtype()) + ".");
  }
}

static AMSTensor makeEmptyGlobalFeatures(const AMSTensor& node_features)
{
  using Dim = AMSTensor::IntDimType;
  const Dim shape[] = {0};
  const Dim strides[] = {1};

  switch (node_features.dtype()) {
    case AMS_SINGLE:
      return AMSTensor::create<float>(shape, strides, node_features.location());
    case AMS_DOUBLE:
      return AMSTensor::create<double>(shape,
                                       strides,
                                       node_features.location());
    default:
      throw std::runtime_error(
          "AMSHomogeneousGraph cannot create empty global_features from "
          "non-floating node_features dtype " +
          dtypeName(node_features.dtype()) + ".");
  }
}

}  // namespace

bool AMSTensorFieldMap::contains(const std::string& name) const
{
  return fields_.find(name) != fields_.end();
}

AMSTensor* AMSTensorFieldMap::find(const std::string& name) noexcept
{
  auto it = fields_.find(name);
  if (it == fields_.end()) {
    return nullptr;
  }
  return &it->second;
}

const AMSTensor* AMSTensorFieldMap::find(const std::string& name) const noexcept
{
  auto it = fields_.find(name);
  if (it == fields_.end()) {
    return nullptr;
  }
  return &it->second;
}

AMSTensor& AMSTensorFieldMap::at(const std::string& name)
{
  return fields_.at(name);
}

const AMSTensor& AMSTensorFieldMap::at(const std::string& name) const
{
  return fields_.at(name);
}

AMSTensor& AMSTensorFieldMap::insert(std::string name, AMSTensor tensor)
{
  if (fields_.find(name) != fields_.end()) {
    throw std::runtime_error("AMSTensorFieldMap::insert: field '" + name +
                             "' already exists");
  }
  auto [it, inserted] = fields_.emplace(std::move(name), std::move(tensor));
  (void)inserted;
  return it->second;
}

AMSTensor& AMSTensorFieldMap::set(std::string name, AMSTensor tensor)
{
  auto it = fields_.find(name);
  if (it != fields_.end()) {
    fields_.erase(it);
  }
  auto [new_it, inserted] = fields_.emplace(std::move(name), std::move(tensor));
  (void)inserted;
  return new_it->second;
}

AMSHomogeneousGraph::AMSHomogeneousGraph(AMSTensor node_features_,
                                         AMSTensor edge_index_,
                                         AMSTensor edge_features_)
    : node_features(std::move(node_features_)),
      edge_index(std::move(edge_index_)),
      edge_features(std::move(edge_features_)),
      global_features(makeEmptyGlobalFeatures(node_features))
{
  validate();
}

AMSHomogeneousGraph::AMSHomogeneousGraph(AMSTensor node_features_,
                                         AMSTensor edge_index_,
                                         AMSTensor edge_features_,
                                         AMSTensor global_features_)
    : node_features(std::move(node_features_)),
      edge_index(std::move(edge_index_)),
      edge_features(std::move(edge_features_)),
      global_features(std::move(global_features_))
{
  validate();
}

void AMSHomogeneousGraph::validate() const
{
  requireRank(node_features, 2, "node_features");
  requireFloating(node_features, "node_features");

  requireRank(edge_index, 2, "edge_index");
  if (!isIntegerDType(edge_index.dtype())) {
    throw std::runtime_error(
        "AMSHomogeneousGraph edge_index must have integer dtype "
        "(int64 preferred, int32 supported), got " +
        dtypeName(edge_index.dtype()) + ".");
  }
  if (edge_index.shape()[0] != 2) {
    throw std::runtime_error(
        "AMSHomogeneousGraph edge_index must have shape [2, E].");
  }

  requireRank(edge_features, 2, "edge_features");
  requireFloating(edge_features, "edge_features");
  if (edge_features.shape()[0] != edge_index.shape()[1]) {
    throw std::runtime_error(
        "AMSHomogeneousGraph edge_features first dimension must match "
        "number of edges.");
  }

  requireRank(global_features, 1, "global_features");
  requireFloating(global_features, "global_features");
}

EdgeType::EdgeType(std::string src_, std::string rel_, std::string dst_)
    : src(std::move(src_)), rel(std::move(rel_)), dst(std::move(dst_))
{
}

bool EdgeType::operator==(const EdgeType& other) const noexcept
{
  return src == other.src && rel == other.rel && dst == other.dst;
}

std::size_t EdgeTypeHash::operator()(const EdgeType& e) const noexcept
{
  std::size_t seed = std::hash<std::string>{}(e.src);
  seed = hashCombine(seed, std::hash<std::string>{}(e.rel));
  seed = hashCombine(seed, std::hash<std::string>{}(e.dst));
  return seed;
}

std::string edgeTypeToString(const EdgeType& e)
{
  return e.src + "__" + e.rel + "__" + e.dst;
}

ams::EdgeType edgeTypeFromString(const std::string& key)
{
  const std::string delim = "__";

  auto p1 = key.find(delim);
  if (p1 == std::string::npos) {
    throw std::runtime_error("edgeTypeFromString: missing first delimiter");
  }

  auto p2 = key.find(delim, p1 + delim.size());
  if (p2 == std::string::npos) {
    throw std::runtime_error("edgeTypeFromString: missing second delimiter");
  }

  if (key.find(delim, p2 + delim.size()) != std::string::npos) {
    throw std::runtime_error("edgeTypeFromString: too many delimiters");
  }

  std::string src = key.substr(0, p1);
  std::string rel = key.substr(p1 + delim.size(), p2 - (p1 + delim.size()));
  std::string dst = key.substr(p2 + delim.size());

  if (src.empty() || rel.empty() || dst.empty()) {
    throw std::runtime_error("edgeTypeFromString: empty src/rel/dst component");
  }

  return ams::EdgeType{std::move(src), std::move(rel), std::move(dst)};
}

bool containsTensor(const AMSTensorMap& store, const std::string& name)
{
  return store.find(name) != store.end();
}

AMSTensor* findTensor(AMSTensorMap& store, const std::string& name) noexcept
{
  auto it = store.find(name);
  if (it == store.end()) {
    return nullptr;
  }
  return &it->second;
}

const AMSTensor* findTensor(const AMSTensorMap& store,
                            const std::string& name) noexcept
{
  auto it = store.find(name);
  if (it == store.end()) {
    return nullptr;
  }
  return &it->second;
}

void insertTensor(AMSTensorMap& store, std::string name, AMSTensor&& tensor)
{
  auto [it, inserted] = store.emplace(std::move(name), std::move(tensor));
  if (!inserted) {
    throw std::runtime_error("insertTensor: tensor name already exists");
  }
}

void insertOrAssignTensor(AMSTensorMap& store,
                          std::string name,
                          AMSTensor&& tensor)
{
  auto it = store.find(name);
  if (it == store.end()) {
    store.emplace(std::move(name), std::move(tensor));
  } else {
    it->second = std::move(tensor);
  }
}

bool AMSHeterogeneousGraph::containsNodeStore(const std::string& name) const
{
  return node_stores.find(name) != node_stores.end();
}

bool AMSHeterogeneousGraph::containsEdgeStore(const EdgeType& edge_type) const
{
  return edge_stores.find(edge_type) != edge_stores.end();
}

AMSTensorMap& AMSHeterogeneousGraph::getOrCreateNodeStore(std::string name)
{
  auto [it, inserted] =
      node_stores.try_emplace(std::move(name), AMSTensorMap{});
  (void)inserted;
  return it->second;
}

AMSTensorMap& AMSHeterogeneousGraph::getOrCreateEdgeStore(EdgeType edge_type)
{
  auto [it, inserted] =
      edge_stores.try_emplace(std::move(edge_type), AMSTensorMap{});
  (void)inserted;
  return it->second;
}

AMSTensorMap* AMSHeterogeneousGraph::findNodeStore(
    const std::string& name) noexcept
{
  auto it = node_stores.find(name);
  if (it == node_stores.end()) {
    return nullptr;
  }
  return &it->second;
}

const AMSTensorMap* AMSHeterogeneousGraph::findNodeStore(
    const std::string& name) const noexcept
{
  auto it = node_stores.find(name);
  if (it == node_stores.end()) {
    return nullptr;
  }
  return &it->second;
}

AMSTensorMap* AMSHeterogeneousGraph::findEdgeStore(
    const EdgeType& edge_type) noexcept
{
  auto it = edge_stores.find(edge_type);
  if (it == edge_stores.end()) {
    return nullptr;
  }
  return &it->second;
}

const AMSTensorMap* AMSHeterogeneousGraph::findEdgeStore(
    const EdgeType& edge_type) const noexcept
{
  auto it = edge_stores.find(edge_type);
  if (it == edge_stores.end()) {
    return nullptr;
  }
  return &it->second;
}

bool AMSHeterogeneousGraphFields::containsNodeStore(
    const std::string& name) const
{
  return node_stores.find(name) != node_stores.end();
}

bool AMSHeterogeneousGraphFields::containsEdgeStore(
    const EdgeType& edge_type) const
{
  return edge_stores.find(edge_type) != edge_stores.end();
}

AMSTensorFieldMap& AMSHeterogeneousGraphFields::getOrCreateNodeStore(
    std::string name)
{
  auto [it, inserted] =
      node_stores.try_emplace(std::move(name), AMSTensorFieldMap{});
  (void)inserted;
  return it->second;
}

AMSTensorFieldMap& AMSHeterogeneousGraphFields::getOrCreateEdgeStore(
    EdgeType edge_type)
{
  auto [it, inserted] =
      edge_stores.try_emplace(std::move(edge_type), AMSTensorFieldMap{});
  (void)inserted;
  return it->second;
}

AMSTensorFieldMap* AMSHeterogeneousGraphFields::findNodeStore(
    const std::string& name) noexcept
{
  auto it = node_stores.find(name);
  if (it == node_stores.end()) {
    return nullptr;
  }
  return &it->second;
}

const AMSTensorFieldMap* AMSHeterogeneousGraphFields::findNodeStore(
    const std::string& name) const noexcept
{
  auto it = node_stores.find(name);
  if (it == node_stores.end()) {
    return nullptr;
  }
  return &it->second;
}

AMSTensorFieldMap* AMSHeterogeneousGraphFields::findEdgeStore(
    const EdgeType& edge_type) noexcept
{
  auto it = edge_stores.find(edge_type);
  if (it == edge_stores.end()) {
    return nullptr;
  }
  return &it->second;
}

const AMSTensorFieldMap* AMSHeterogeneousGraphFields::findEdgeStore(
    const EdgeType& edge_type) const noexcept
{
  auto it = edge_stores.find(edge_type);
  if (it == edge_stores.end()) {
    return nullptr;
  }
  return &it->second;
}

}  // namespace ams
