#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>

#include "AMSTensor.hpp"

namespace ams
{

using AMSTensorMap = std::unordered_map<std::string, AMSTensor>;

class AMSTensorFieldMap
{
  AMSTensorMap fields_;

public:
  using const_iterator = AMSTensorMap::const_iterator;

  // Explicit named tensor field store. There is intentionally no operator[]:
  // use set()/insert() to create fields and at()/find() to read them so missing
  // lookups never create invalid/default AMSTensors.
  AMSTensorFieldMap() = default;
  AMSTensorFieldMap(const AMSTensorFieldMap&) = delete;
  AMSTensorFieldMap& operator=(const AMSTensorFieldMap&) = delete;
  AMSTensorFieldMap(AMSTensorFieldMap&&) noexcept = default;
  AMSTensorFieldMap& operator=(AMSTensorFieldMap&&) noexcept = default;
  ~AMSTensorFieldMap() = default;

  bool contains(const std::string& name) const;

  AMSTensor* find(const std::string& name) noexcept;
  const AMSTensor* find(const std::string& name) const noexcept;

  AMSTensor& at(const std::string& name);
  const AMSTensor& at(const std::string& name) const;

  AMSTensor& insert(std::string name, AMSTensor tensor);
  AMSTensor& set(std::string name, AMSTensor tensor);

  const_iterator begin() const noexcept { return fields_.begin(); }
  const_iterator end() const noexcept { return fields_.end(); }
  const_iterator cbegin() const noexcept { return fields_.cbegin(); }
  const_iterator cend() const noexcept { return fields_.cend(); }

  bool empty() const noexcept { return fields_.empty(); }
  std::size_t size() const noexcept { return fields_.size(); }
  void clear() noexcept { fields_.clear(); }
};

struct AMSHomogeneousGraph {
  // Homogeneous graph input contract:
  // node_features [N, F_node] floating point
  // edge_index [2, E] integer connectivity, int64 canonical and int32 allowed
  // when the model handles it explicitly
  // edge_features [E, F_edge] floating point
  // global_features [F_global] floating point, with [0] meaning unused
  AMSTensor node_features;
  AMSTensor edge_index;
  AMSTensor edge_features;
  AMSTensor global_features;

  AMSTensorMap node_fields;
  AMSTensorMap edge_fields;
  AMSTensorMap global_fields;

  AMSHomogeneousGraph() = delete;
  AMSHomogeneousGraph(AMSTensor node_features_,
                      AMSTensor edge_index_,
                      AMSTensor edge_features_);
  AMSHomogeneousGraph(AMSTensor node_features_,
                      AMSTensor edge_index_,
                      AMSTensor edge_features_,
                      AMSTensor global_features_);
  AMSHomogeneousGraph(const AMSHomogeneousGraph&) = delete;
  AMSHomogeneousGraph& operator=(const AMSHomogeneousGraph&) = delete;
  AMSHomogeneousGraph(AMSHomogeneousGraph&&) noexcept = default;
  AMSHomogeneousGraph& operator=(AMSHomogeneousGraph&&) noexcept = default;
  ~AMSHomogeneousGraph() = default;

  void validate() const;
};

struct AMSHomogeneousGraphFields {
  AMSTensorFieldMap node_fields;
  AMSTensorFieldMap edge_fields;
  AMSTensorFieldMap global_fields;

  AMSHomogeneousGraphFields() = default;
  AMSHomogeneousGraphFields(const AMSHomogeneousGraphFields&) = delete;
  AMSHomogeneousGraphFields& operator=(const AMSHomogeneousGraphFields&) =
      delete;
  AMSHomogeneousGraphFields(AMSHomogeneousGraphFields&&) noexcept = default;
  AMSHomogeneousGraphFields& operator=(AMSHomogeneousGraphFields&&) noexcept =
      default;
  ~AMSHomogeneousGraphFields() = default;
};

struct EdgeType {
  std::string src;
  std::string rel;
  std::string dst;

  EdgeType() = default;
  EdgeType(std::string src_, std::string rel_, std::string dst_);

  bool operator==(const EdgeType& other) const noexcept;
  bool operator!=(const EdgeType& other) const noexcept
  {
    return !(*this == other);
  }
};

struct EdgeTypeHash {
  std::size_t operator()(const EdgeType& e) const noexcept;
};

std::string edgeTypeToString(const EdgeType& e);
ams::EdgeType edgeTypeFromString(const std::string& key);

// Generic helpers for any named tensor store.
// These are useful both for AMSHomogeneousGraph and for the stores inside
// AMSHeterogeneousGraph.
bool containsTensor(const AMSTensorMap& store, const std::string& name);

AMSTensor* findTensor(AMSTensorMap& store, const std::string& name) noexcept;
const AMSTensor* findTensor(const AMSTensorMap& store,
                            const std::string& name) noexcept;

// Insert a new tensor. Throws if the name already exists.
void insertTensor(AMSTensorMap& store, std::string name, AMSTensor&& tensor);

// Insert or replace an existing tensor.
void insertOrAssignTensor(AMSTensorMap& store,
                          std::string name,
                          AMSTensor&& tensor);

struct AMSHeterogeneousGraph {
  using NodeStoreMap = std::unordered_map<std::string, AMSTensorMap>;
  using EdgeStoreMap = std::unordered_map<EdgeType, AMSTensorMap, EdgeTypeHash>;

  NodeStoreMap node_stores;
  EdgeStoreMap edge_stores;
  AMSTensorMap global_store;

  AMSHeterogeneousGraph() = default;
  AMSHeterogeneousGraph(const AMSHeterogeneousGraph&) = delete;
  AMSHeterogeneousGraph& operator=(const AMSHeterogeneousGraph&) = delete;
  AMSHeterogeneousGraph(AMSHeterogeneousGraph&&) noexcept = default;
  AMSHeterogeneousGraph& operator=(AMSHeterogeneousGraph&&) noexcept = default;
  ~AMSHeterogeneousGraph() = default;

  bool containsNodeStore(const std::string& name) const;
  bool containsEdgeStore(const EdgeType& edge_type) const;

  AMSTensorMap& getOrCreateNodeStore(std::string name);
  AMSTensorMap& getOrCreateEdgeStore(EdgeType edge_type);

  AMSTensorMap* findNodeStore(const std::string& name) noexcept;
  const AMSTensorMap* findNodeStore(const std::string& name) const noexcept;

  AMSTensorMap* findEdgeStore(const EdgeType& edge_type) noexcept;
  const AMSTensorMap* findEdgeStore(const EdgeType& edge_type) const noexcept;
};

struct AMSHeterogeneousGraphFields {
  using NodeStoreMap = std::unordered_map<std::string, AMSTensorFieldMap>;
  using EdgeStoreMap =
      std::unordered_map<EdgeType, AMSTensorFieldMap, EdgeTypeHash>;

  NodeStoreMap node_stores;
  EdgeStoreMap edge_stores;
  AMSTensorFieldMap global_store;

  AMSHeterogeneousGraphFields() = default;
  AMSHeterogeneousGraphFields(const AMSHeterogeneousGraphFields&) = delete;
  AMSHeterogeneousGraphFields& operator=(const AMSHeterogeneousGraphFields&) =
      delete;
  AMSHeterogeneousGraphFields(AMSHeterogeneousGraphFields&&) noexcept = default;
  AMSHeterogeneousGraphFields& operator=(
      AMSHeterogeneousGraphFields&&) noexcept = default;
  ~AMSHeterogeneousGraphFields() = default;

  bool containsNodeStore(const std::string& name) const;
  bool containsEdgeStore(const EdgeType& edge_type) const;

  AMSTensorFieldMap& getOrCreateNodeStore(std::string name);
  AMSTensorFieldMap& getOrCreateEdgeStore(EdgeType edge_type);

  AMSTensorFieldMap* findNodeStore(const std::string& name) noexcept;
  const AMSTensorFieldMap* findNodeStore(
      const std::string& name) const noexcept;

  AMSTensorFieldMap* findEdgeStore(const EdgeType& edge_type) noexcept;
  const AMSTensorFieldMap* findEdgeStore(
      const EdgeType& edge_type) const noexcept;
};

}  // namespace ams
