/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "wf/jsondb.hpp"

#if defined(__AMS_ENABLE_TORCH__)
#include "AMSTorchInterop.hpp"
#endif

#include <algorithm>
#include <cstring>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>

#include "wf/debug.h"
#include "wf/resource_manager.hpp"

using namespace ams::db;
using namespace ams;

// ----------------------------------------------------------------------
// Helper functions
// ----------------------------------------------------------------------

namespace
{

namespace fs = std::experimental::filesystem;

// Check system endianness
bool isLittleEndian()
{
  uint32_t test = 0x01020304;
  return (*reinterpret_cast<uint8_t*>(&test)) == 0x04;
}

// Base64 encoding for pure JSON mode
std::string base64Encode(const uint8_t* data, size_t len)
{
  static const char* base64_chars =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

  std::string ret;
  int i = 0;
  uint8_t char_array_3[3];
  uint8_t char_array_4[4];

  while (len--) {
    char_array_3[i++] = *(data++);
    if (i == 3) {
      char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
      char_array_4[1] =
          ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
      char_array_4[2] =
          ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
      char_array_4[3] = char_array_3[2] & 0x3f;

      for (i = 0; i < 4; i++)
        ret += base64_chars[char_array_4[i]];
      i = 0;
    }
  }

  if (i) {
    for (int j = i; j < 3; j++)
      char_array_3[j] = '\0';

    char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
    char_array_4[1] =
        ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
    char_array_4[2] =
        ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

    for (int j = 0; j < i + 1; j++)
      ret += base64_chars[char_array_4[j]];

    while (i++ < 3)
      ret += '=';
  }

  return ret;
}

SmallVector<AMSTensor::IntDimType> contiguousStrides(
    ArrayRef<AMSTensor::IntDimType> shape)
{
  SmallVector<AMSTensor::IntDimType> strides(shape.size(), 1);
  AMSTensor::IntDimType stride = 1;
  for (size_t i = shape.size(); i-- > 0;) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

size_t logicalOffset(size_t linear,
                     ArrayRef<AMSTensor::IntDimType> shape,
                     ArrayRef<AMSTensor::IntDimType> strides)
{
  size_t offset = 0;
  for (size_t axis = shape.size(); axis-- > 0;) {
    const size_t dim = static_cast<size_t>(shape[axis]);
    const size_t index = dim == 0 ? 0 : linear % dim;
    if (dim != 0) linear /= dim;
    offset += index * static_cast<size_t>(strides[axis]);
  }
  return offset;
}

AMSTensor materializeContiguousHost(const AMSTensor& tensor)
{
  const auto shape = tensor.shape();
  const auto strides = contiguousStrides(shape);
  AMSTensor host = [&]() {
    switch (tensor.dtype()) {
      case AMS_SINGLE:
        return AMSTensor::create<float>(shape, strides, AMS_HOST);
      case AMS_DOUBLE:
        return AMSTensor::create<double>(shape, strides, AMS_HOST);
      case AMS_INT32:
        return AMSTensor::create<int32_t>(shape, strides, AMS_HOST);
      case AMS_INT64:
        return AMSTensor::create<int64_t>(shape, strides, AMS_HOST);
      default:
        throw std::invalid_argument("Unsupported AMSTensor dtype for JSONDB");
    }
  }();

  if (tensor.nbytes() == 0) return host;

  auto* destination = static_cast<uint8_t*>(host.data_ptr());
  auto* source = static_cast<const uint8_t*>(tensor.data_ptr());
  if (tensor.contiguous()) {
    internal::_raw_copy(const_cast<uint8_t*>(source),
                        tensor.location(),
                        destination,
                        AMS_HOST,
                        tensor.nbytes());
    return host;
  }

  for (size_t i = 0; i < static_cast<size_t>(tensor.elements()); ++i) {
    const size_t source_offset =
        logicalOffset(i, tensor.shape(), tensor.strides()) *
        static_cast<size_t>(tensor.element_size());
    const size_t destination_offset =
        i * static_cast<size_t>(tensor.element_size());
    internal::_raw_copy(const_cast<uint8_t*>(source + source_offset),
                        tensor.location(),
                        destination + destination_offset,
                        AMS_HOST,
                        static_cast<size_t>(tensor.element_size()));
  }
  return host;
}

}  // anonymous namespace

// ----------------------------------------------------------------------
// JSONDB Implementation
// ----------------------------------------------------------------------

JSONDB::JSONDB(std::string path,
               std::string domain_name,
               uint64_t rId,
               std::string json_mode)
    : FileDB(path, domain_name, "_jsondb.json", rId),
      json_mode_(json_mode),
      case_counter_(0),
      finalized_(false)
{
  if (!isLittleEndian()) {
    AMS_WARNING(JSONDB,
                "System is not little-endian. Binary output may not be "
                "compatible with Python loaders.");
  }

  if (json_mode_ != "binary" && json_mode_ != "json") {
    THROW(std::invalid_argument,
          ("Invalid json_mode: " + json_mode_ + ". Must be 'binary' or 'json'.")
              .c_str());
  }

  AMS_DBG(JSONDB, "Created JSONDB at '{}' with mode '{}'", fp, json_mode_);
}

JSONDB::~JSONDB()
{
  if (!finalized_) {
    try {
      close();
    } catch (const std::exception& e) {
      AMS_WARNING(JSONDB,
                  "Exception while automatically closing JSONDB: {}",
                  e.what());
    }
  }
}

std::string JSONDB::dtypeToString(AMSDType dtype) const
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
      throw std::invalid_argument("Unsupported AMSTensor dtype for JSONDB");
  }
}

size_t JSONDB::writeBinaryTensor(const AMSTensor& tensor,
                                 const std::string& path)
{
  const void* data = tensor.data_ptr();
  const size_t byte_size = tensor.nbytes();

  // Write binary file
  fs::path full_path = fs::path(fp) / path;
  fs::create_directories(full_path.parent_path());

  std::ofstream file(full_path.string(), std::ios::binary);
  if (!file.is_open()) {
    THROW(std::runtime_error,
          ("Failed to open file for writing: " + full_path.string()).c_str());
  }

  file.write(static_cast<const char*>(data), byte_size);
  file.close();

  AMS_DBG(JSONDB, "Wrote binary tensor to '{}' ({} bytes)", path, byte_size);

  return byte_size;
}

nlohmann::json JSONDB::encodeBase64Tensor(const AMSTensor& tensor)
{
  const uint8_t* data = static_cast<const uint8_t*>(tensor.data_ptr());
  const size_t byte_size = tensor.nbytes();

  nlohmann::json result;
  result["encoding"] = "base64";
  result["data"] = base64Encode(data, byte_size);
  result["dtype"] = dtypeToString(tensor.dtype());
  result["byte_size"] = byte_size;

  // Add shape
  auto shape_ref = tensor.shape();
  result["shape"] = std::vector<int64_t>(shape_ref.begin(), shape_ref.end());

  return result;
}

nlohmann::json JSONDB::serializeTensor(const AMSTensor& tensor,
                                       const std::string& binary_path)
{
  AMSTensor host = materializeContiguousHost(tensor);
  if (json_mode_ == "json") {
    return encodeBase64Tensor(host);
  }

  auto shape_ref = host.shape();
  std::vector<int64_t> shape(shape_ref.begin(), shape_ref.end());
  size_t byte_size = writeBinaryTensor(host, binary_path);
  return nlohmann::json{{"path", binary_path},
                        {"dtype", dtypeToString(host.dtype())},
                        {"shape", shape},
                        {"byte_size", byte_size}};
}

nlohmann::json JSONDB::serializeOutputFields(const AMSTensorFieldMap& fields,
                                             const std::string& case_dir,
                                             const std::string& association)
{
  std::vector<std::pair<std::string, const AMSTensor*>> sorted_fields;
  sorted_fields.reserve(fields.size());
  for (const auto& [name, tensor] : fields) {
    sorted_fields.emplace_back(name, &tensor);
  }
  std::sort(sorted_fields.begin(),
            sorted_fields.end(),
            [](const auto& lhs, const auto& rhs) {
              return lhs.first < rhs.first;
            });

  nlohmann::json output_json = nlohmann::json::object();
  for (size_t i = 0; i < sorted_fields.size(); ++i) {
    std::ostringstream filename;
    filename << "field_" << std::setw(6) << std::setfill('0') << i << ".bin";
    std::string binary_path =
        case_dir + "/outputs/" + association + "/" + filename.str();
    output_json[sorted_fields[i].first] =
        serializeTensor(*sorted_fields[i].second, binary_path);
  }
  return output_json;
}

void JSONDB::validateEdgeIndex(const AMSTensor& edge_index, int64_t num_nodes)
{
  auto shape_ref = edge_index.shape();
  if (shape_ref.size() != 2 || shape_ref[0] != 2) {
    std::ostringstream oss;
    oss << "edge_index must have shape [2, E], got [";
    for (size_t i = 0; i < shape_ref.size(); ++i) {
      oss << shape_ref[i];
      if (i < shape_ref.size() - 1) oss << ", ";
    }
    oss << "]";
    THROW(std::invalid_argument, oss.str().c_str());
  }

  // Check dtype is int64
  if (edge_index.dtype() != AMS_INT64) {
    THROW(std::invalid_argument, "edge_index must have dtype int64");
  }

  // Validate indices are in range
  AMSTensor host = materializeContiguousHost(edge_index);
  const int64_t* indices = host.data<int64_t>();
  int64_t num_edges = shape_ref[1];

  for (int64_t i = 0; i < 2 * num_edges; ++i) {
    if (indices[i] < 0 || indices[i] >= num_nodes) {
      std::ostringstream oss;
      oss << "edge_index contains out-of-range index: " << indices[i]
          << " (num_nodes=" << num_nodes << ")";
      THROW(std::invalid_argument, oss.str().c_str());
    }
  }

  // Check for self-loops
  for (int64_t i = 0; i < num_edges; ++i) {
    if (indices[i] == indices[i + num_edges]) {
      std::ostringstream oss;
      oss << "edge_index contains self-loop at edge " << i << ": " << indices[i]
          << " -> " << indices[i + num_edges];
      THROW(std::invalid_argument, oss.str().c_str());
    }
  }
}

// ----------------------------------------------------------------------
// Store methods
// ----------------------------------------------------------------------

void JSONDB::store(ArrayRef<AMSTensor> Inputs, ArrayRef<AMSTensor> Outputs)
{
  // Create case directory
  std::ostringstream case_name;
  case_name << "case_" << getId() << "_" << std::setw(6) << std::setfill('0')
            << case_counter_;
  std::string case_dir = case_name.str();

  nlohmann::json case_json;
  case_json["name"] = case_dir;
  case_json["case_index"] = case_counter_;

  nlohmann::json tensors_json;

  // Store inputs
  for (size_t i = 0; i < Inputs.size(); ++i) {
    std::ostringstream tensor_name;
    tensor_name << "input_" << i;
    std::string name = tensor_name.str();

    std::string rel_path = case_dir + "/" + name + ".bin";
    tensors_json[name] = serializeTensor(Inputs[i], rel_path);
  }

  // Store outputs
  for (size_t i = 0; i < Outputs.size(); ++i) {
    std::ostringstream tensor_name;
    tensor_name << "output_" << i;
    std::string name = tensor_name.str();

    std::string rel_path = case_dir + "/" + name + ".bin";
    tensors_json[name] = serializeTensor(Outputs[i], rel_path);
  }

  case_json["tensors"] = tensors_json;
  cases_.push_back(case_json);

  case_counter_++;

  AMS_DBG(JSONDB,
          "Stored tensor data for case {} ({} inputs, {} outputs)",
          case_dir,
          Inputs.size(),
          Outputs.size());
}

#if defined(__AMS_ENABLE_TORCH__)
void JSONDB::store(ArrayRef<torch::Tensor> Inputs,
                   ArrayRef<torch::Tensor> Outputs)
{
  std::vector<AMSTensor> ams_inputs;
  std::vector<AMSTensor> ams_outputs;
  ams_inputs.reserve(Inputs.size());
  ams_outputs.reserve(Outputs.size());
  for (const auto& tensor : Inputs)
    ams_inputs.emplace_back(ams::fromTorchView(tensor));
  for (const auto& tensor : Outputs)
    ams_outputs.emplace_back(ams::fromTorchView(tensor));
  store(ams_inputs, ams_outputs);
}
#endif

void JSONDB::store(const ams::AMSHomogeneousGraph& graph,
                   const ams::AMSHomogeneousGraphFields& outputs)
{
  // Create case directory
  std::ostringstream case_name;
  case_name << "step_" << getId() << "_" << std::setw(6) << std::setfill('0')
            << case_counter_;
  std::string case_dir = case_name.str();

  // Extract graph dimensions
  auto node_shape = graph.node_features.shape();
  auto edge_shape = graph.edge_index.shape();

  int64_t num_nodes = node_shape[0];
  int64_t num_edges = edge_shape[1];
  int64_t node_feature_dim = node_shape[1];
  int64_t edge_feature_dim = 0;
  int64_t global_feature_dim = 0;

  if (graph.edge_features.elements() > 0) {
    auto ef_shape = graph.edge_features.shape();
    edge_feature_dim = ef_shape[1];
  }

  if (graph.global_features.elements() > 0) {
    auto gf_shape = graph.global_features.shape();
    global_feature_dim = gf_shape[0];
  }

  // Validate edge_index
  validateEdgeIndex(graph.edge_index, num_nodes);

  // Build case metadata
  nlohmann::json case_json;
  case_json["name"] = case_dir;
  case_json["step_index"] = case_counter_;
  case_json["num_nodes"] = num_nodes;
  case_json["num_edges"] = num_edges;
  case_json["node_feature_dim"] = node_feature_dim;
  case_json["edge_feature_dim"] = edge_feature_dim;
  case_json["global_feature_dim"] = global_feature_dim;

  nlohmann::json tensors_json;

  tensors_json["node_features"] =
      serializeTensor(graph.node_features, case_dir + "/node_features.bin");

  tensors_json["edge_index"] =
      serializeTensor(graph.edge_index, case_dir + "/edge_index.bin");

  // Write edge_features
  if (edge_feature_dim > 0) {
    tensors_json["edge_features"] =
        serializeTensor(graph.edge_features, case_dir + "/edge_features.bin");
  }

  // Write global_features
  if (global_feature_dim > 0) {
    tensors_json["global_features"] =
        serializeTensor(graph.global_features,
                        case_dir + "/global_features.bin");
  }

  case_json["tensors"] = tensors_json;
  case_json["outputs"] = {
      {"node", serializeOutputFields(outputs.node_fields, case_dir, "node")},
      {"edge", serializeOutputFields(outputs.edge_fields, case_dir, "edge")},
      {"global",
       serializeOutputFields(outputs.global_fields, case_dir, "global")}};
  cases_.push_back(case_json);

  case_counter_++;

  AMS_DBG(JSONDB,
          "Stored graph data for step {} ({} nodes, {} edges)",
          case_dir,
          num_nodes,
          num_edges);
}

void JSONDB::store(const ams::AMSHeterogeneousGraph&,
                   const ams::AMSHeterogeneousGraphFields&)
{
  throw std::runtime_error(
      "Heterogeneous graph storage not yet implemented in JSONDB");
}

void JSONDB::close()
{
  if (finalized_) {
    AMS_DBG(JSONDB, "Manifest already finalized, skipping");
    return;
  }

  // Build complete manifest
  nlohmann::json manifest;
  manifest["format_version"] = 1;
  manifest["endianness"] = isLittleEndian() ? "little" : "big";

  // Add metadata if set
  if (!metadata_.is_null()) {
    manifest["metadata"] = metadata_;
  }

  // Add feature names if set
  if (!feature_names_.is_null()) {
    manifest["feature_names"] = feature_names_;
  }

  // Add all cases
  manifest["cases"] = cases_;

  // FileDB provides a domain- and rank-specific manifest filename.
  fs::path manifest_path = fn;
  std::ofstream manifest_file(manifest_path.string());
  if (!manifest_file.is_open()) {
    THROW(std::runtime_error,
          ("Failed to open manifest file for writing: " +
           manifest_path.string())
              .c_str());
  }

  manifest_file << std::setw(2) << manifest << std::endl;
  manifest_file.close();

  finalized_ = true;

  AMS_DBG(JSONDB,
          "Finalized manifest with {} cases at '{}'",
          cases_.size(),
          manifest_path.string());
}
