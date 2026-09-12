/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_JSON_DB__
#define __AMS_JSON_DB__

#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "AMSGraph.hpp"
#include "AMSTensor.hpp"
#include "wf/basedb.hpp"

#if defined(__AMS_ENABLE_TORCH__)
#include <torch/torch.h>
#endif

namespace ams
{
namespace db
{

/**
 * @brief JSON database backend for storing both tensor and graph data.
 *
 * This class provides a JSON-based storage backend that can serialize flat
 * tensors and homogeneous graphs. Heterogeneous graph storage is not yet
 * supported.
 *
 * Supports two modes:
 * - "binary": Binary tensor files + JSON manifest (default, efficient)
 * - "json": Self-contained JSON with base64-encoded tensor data
 *
 * Homogeneous graph inputs are stored under each case's "tensors" object.
 * Named graph outputs are stored under "outputs", grouped into "node", "edge",
 * and "global" objects using the application-provided field names.
 *
 * Output format is compatible with PyTorch Geometric data loaders.
 *
 * @note Case and step directory names include the database rank ID so
 * concurrent MPI ranks can write to the same output directory without tensor
 * file collisions.
 */
class JSONDB final : public FileDB
{
private:
  /** @brief JSON output mode: "binary" or "json" */
  std::string json_mode_;
  /** @brief Case counter for generating unique identifiers */
  int case_counter_;
  /** @brief Accumulated case metadata for manifest */
  std::vector<nlohmann::json> cases_;
  /** @brief Metadata from application (physics config, etc.) */
  nlohmann::json metadata_;
  /** @brief Feature names for each tensor type */
  nlohmann::json feature_names_;
  /** @brief Whether manifest has been finalized */
  bool finalized_;

  /**
   * @brief Write a tensor to binary file
   * @param[in] tensor The AMSTensor to write
   * @param[in] path Relative path from output directory
   * @return Actual byte size written
   */
  size_t writeBinaryTensor(const AMSTensor& tensor, const std::string& path);

  /**
   * @brief Encode tensor as base64 JSON (for pure JSON mode)
   * @param[in] tensor The AMSTensor to encode
   * @return JSON object with encoded data
   */
  nlohmann::json encodeBase64Tensor(const AMSTensor& tensor);

  /**
   * @brief Serialize an AMSTensor using the configured JSON mode
   * @param[in] tensor The tensor to serialize
   * @param[in] binary_path Relative binary path used in binary mode
   * @return JSON tensor descriptor
   */
  nlohmann::json serializeTensor(const AMSTensor& tensor,
                                 const std::string& binary_path);

  /**
   * @brief Serialize all named output fields for one graph association
   * @param[in] fields Named output fields
   * @param[in] case_dir Relative case directory
   * @param[in] association Output association: node, edge, or global
   * @return JSON object keyed by application field name
   */
  nlohmann::json serializeOutputFields(const AMSTensorFieldMap& fields,
                                       const std::string& case_dir,
                                       const std::string& association);

  /**
   * @brief Validate edge_index tensor format
   * @param[in] edge_index Edge connectivity tensor [2, E]
   * @param[in] num_nodes Number of nodes in graph
   */
  void validateEdgeIndex(const AMSTensor& edge_index, int64_t num_nodes);

  /**
   * @brief Convert AMSDType to string
   * @param[in] dtype The AMS data type
   * @return String representation ("float32", "float64", "int64")
   */
  std::string dtypeToString(AMSDType dtype) const;

public:
  /**
   * @brief Construct a JSON database
   * @param[in] path Directory path for output files
   * @param[in] domain_name Name of the domain (used in filenames)
   * @param[in] rId Rank ID for distributed execution
   * @param[in] json_mode "binary" (default) or "json" for output mode
   */
  JSONDB(std::string path,
         std::string domain_name,
         uint64_t rId,
         std::string json_mode = "binary");

  /**
   * @brief Destructor - finalizes manifest if not already done
   */
  ~JSONDB() override;

  // Delete copy/move constructors
  JSONDB(const JSONDB&) = delete;
  JSONDB& operator=(const JSONDB&) = delete;

  /**
   * @brief Store tensor data (inputs and outputs)
   * @param[in] Inputs Vector of input tensors
   * @param[in] Outputs Vector of output tensors
   */
  void store(ArrayRef<AMSTensor> Inputs, ArrayRef<AMSTensor> Outputs) override;

#if defined(__AMS_ENABLE_TORCH__)
  /**
   * @brief Store PyTorch tensors through the canonical AMSTensor path
   * @param[in] Inputs Vector of input tensors
   * @param[in] Outputs Vector of output tensors
   */
  void store(ArrayRef<torch::Tensor> Inputs, ArrayRef<torch::Tensor> Outputs);
#endif

  /**
   * @brief Store homogeneous graph data with outputs
   * @param[in] graph Input graph structure and features
   * @param[in] outputs Output/target fields for training
   */
  void store(const ams::AMSHomogeneousGraph& graph,
             const ams::AMSHomogeneousGraphFields& outputs) override;

  /**
   * @brief Store heterogeneous graph data with outputs
   * @param[in] graph Input graph structure and features
   * @param[in] outputs Output/target fields for training
   */
  void store(const ams::AMSHeterogeneousGraph& graph,
             const ams::AMSHeterogeneousGraphFields& outputs) override;

  /**
   * @brief Finalize and write the JSON manifest
   */
  void close() override;

  /**
   * @brief Set application metadata
   * @param[in] metadata JSON object with application-specific config
   */
  void setMetadata(const nlohmann::json& metadata) { metadata_ = metadata; }

  /**
   * @brief Set feature names for documentation
   * @param[in] feature_names JSON object mapping tensor types to name lists
   */
  void setFeatureNames(const nlohmann::json& feature_names)
  {
    feature_names_ = feature_names;
  }

  /**
   * @brief Database type identifier
   */
  std::string type() override { return "json"; }

  /**
   * @brief Database type enum
   */
  AMSDBType dbType() override { return AMSDBType::AMS_JSON; }
};

}  // namespace db
}  // namespace ams

#endif  // __AMS_JSON_DB__
