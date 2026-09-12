#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "AMS.h"
#include "AMSGraph.hpp"
#include "AMSTensor.hpp"
#include "ams_tensor_test_utils.hpp"

// This test is the C++ half of the MGN diffusion example.
//
// The Python generator does the learned-model work:
//
//   1. create synthetic graph diffusion data;
//   2. train a small pure-Torch MeshGraphNet-like model;
//   3. export that model as an AMS-wrapped TorchScript file;
//   4. write "fixtures": fixed graph inputs plus Python TorchScript reference
//      outputs.
//
// This C++ test does not know the diffusion formula and does not try to judge
// model quality. Its job is deployment parity: load the exact TorchScript model
// and graph tensors from the committed fixture directory, run them through AMS,
// and verify that the AMS output field node:delta_u matches the Python
// TorchScript reference output.
//
// Said differently: the opt-in Python training workflow proves "the model
// learned something"; this default test proves "AMS runs the checked-in model
// the same way Python did when its references were generated."

using namespace ams;
using ams::test::makeTensor;
using json = nlohmann::json;

using Dim = AMSTensor::IntDimType;

// Normal testing reads a checked-in TorchScript model and self-contained JSON
// manifest. The expensive training/export workflow remains available as an
// opt-in CTest path for deliberate artifact regeneration.
#ifndef AMS_MGN_DIFFUSION_FIXTURE_DIR
#define AMS_MGN_DIFFUSION_FIXTURE_DIR ""
#endif

namespace
{

constexpr int kFixtureFormatVersion = 2;
constexpr std::int64_t kNodeFeatureDim = 4;
constexpr std::int64_t kEdgeFeatureDim = 4;
constexpr std::int64_t kGlobalFeatureDim = 1;
constexpr std::int64_t kReferenceOutputDim = 1;
constexpr double kUnusedGraphSurrogateThreshold = 1.0;
constexpr int kSingleRankProcessId = 0;
constexpr int kSingleRankWorldSize = 1;
constexpr bool kStoreTrainingData = false;
constexpr const char* kDomainName = "test_mgn_graph_diffusion_surrogate";

// Glossary for readers new to this path:
//
// manifest: fixtures.json, the self-contained graph inputs and references.
// fixture directory: source-tree directory containing fixtures.json and the
//   checked-in TorchScript model.
// AMSHomogeneousGraph: AMS C++ container for node_features, edge_index,
//   edge_features, and optional global_features.
// node:delta_u: TorchScript output key. AMS parses this into the node field
//   named "delta_u".
// fallback callback: application callback AMS calls only when surrogate
//   inference is unavailable or rejected.
// parity: equality of AMS/LibTorch inference and Python TorchScript inference.

struct TensorMetadata {
  std::string dtype;
  std::vector<std::int64_t> shape;
};

static Dim toDim(std::int64_t value)
{
  CATCH_REQUIRE(value >= 0);
  CATCH_REQUIRE(value <= std::numeric_limits<Dim>::max());
  return static_cast<Dim>(value);
}

static std::uintmax_t shapeElementCount(const std::vector<std::int64_t>& shape)
{
  std::uintmax_t count = 1;
  for (std::int64_t dim : shape) {
    CATCH_REQUIRE(dim >= 0);
    count *= static_cast<std::uintmax_t>(dim);
  }
  return count;
}

static TensorMetadata parseTensorMetadata(const json& tensor)
{
  // Pull the small per-tensor manifest record into a typed C++ struct. This is
  // intentionally separate from validation so missing fields, unsupported
  // dtypes, and wrong shapes fail at the most helpful point in the test.
  CATCH_REQUIRE(tensor.contains("dtype"));
  CATCH_REQUIRE(tensor.contains("shape"));
  CATCH_REQUIRE(tensor.contains("values"));
  CATCH_REQUIRE(tensor.at("values").is_array());

  TensorMetadata metadata;
  metadata.dtype = tensor.at("dtype").get<std::string>();
  metadata.shape = tensor.at("shape").get<std::vector<std::int64_t>>();
  return metadata;
}

static void validateTensorMetadata(
    const TensorMetadata& metadata,
    const std::string& expected_dtype,
    const std::vector<std::int64_t>& expected_shape)
{
  // Validate the self-contained manifest before allocating AMSTensors. Shape
  // and dtype remain explicit even though values are ordinary JSON numbers.
  CATCH_REQUIRE(metadata.dtype == expected_dtype);
  CATCH_REQUIRE(metadata.shape == expected_shape);
}

template <typename T>
static std::vector<T> readTensorValues(
    const json& tensor,
    const std::string& expected_dtype,
    const std::vector<std::int64_t>& expected_shape)
{
  TensorMetadata metadata = parseTensorMetadata(tensor);
  validateTensorMetadata(metadata, expected_dtype, expected_shape);

  const std::uintmax_t element_count = shapeElementCount(metadata.shape);
  CATCH_REQUIRE(element_count <= std::numeric_limits<std::size_t>::max());
  CATCH_REQUIRE(tensor.at("values").size() ==
                static_cast<std::size_t>(element_count));
  std::vector<T> values = tensor.at("values").get<std::vector<T>>();
  return values;
}

static json loadManifest(const std::filesystem::path& fixture_dir)
{
  // fixtures.json is generated deliberately and checked into the source tree.
  // If it is missing, the default parity test has no graph inputs or references.
  const std::filesystem::path manifest_path = fixture_dir / "fixtures.json";
  if (!std::filesystem::exists(manifest_path)) {
    CATCH_FAIL("Missing MGN graph diffusion fixtures at " << manifest_path
                                                          << ". Regenerate and "
                                                             "commit "
                                                             "mgn_graph_"
                                                             "diffusion.pt and "
                                                             "fixtures.json.");
  }

  std::ifstream input(manifest_path);
  CATCH_REQUIRE(input);
  json manifest = json::parse(input);

  // Keep the manifest version check close to parsing so future fixture format
  // changes fail loudly instead of being interpreted as the current embedded
  // tensor contract.
  CATCH_REQUIRE(manifest.contains("format_version"));
  CATCH_REQUIRE(manifest.at("format_version").get<int>() ==
                kFixtureFormatVersion);
  CATCH_REQUIRE(manifest.contains("model"));
  CATCH_REQUIRE(manifest.contains("metadata"));
  CATCH_REQUIRE(manifest.contains("cases"));
  CATCH_REQUIRE(manifest.contains("comparison"));
  CATCH_REQUIRE(manifest.at("cases").is_array());
  return manifest;
}

static std::filesystem::path resolveManifestRelativePath(
    const std::filesystem::path& manifest_dir,
    const json& object)
{
  // The model path is relative by design so the two checked-in fixture files
  // remain relocatable as a unit.
  const std::filesystem::path relative_path =
      object.at("path").get<std::string>();
  CATCH_REQUIRE_FALSE(relative_path.empty());
  CATCH_REQUIRE_FALSE(relative_path.is_absolute());
  for (const auto& part : relative_path) {
    CATCH_REQUIRE(part.string() != "..");
  }
  return manifest_dir / relative_path;
}

static AMSHomogeneousGraph makeGraph(const json& graph_case)
{
  // Runtime fixture values become the same AMSTensor-backed homogeneous graph
  // that an application would pass to AMS:
  //
  //   node_features   [N, 4]  -> x, y, u, kappa
  //   edge_index      [2, E]  -> source row, destination row
  //   edge_features   [E, 4]  -> dx, dy, distance, message
  //   global_features [1]     -> dt
  //
  // Keeping this conversion here, instead of compiling arrays into the test,
  // makes the test exercise the private fixture schema at runtime.
  const std::int64_t num_nodes = graph_case.at("num_nodes").get<std::int64_t>();
  const std::int64_t num_edges = graph_case.at("num_edges").get<std::int64_t>();
  const std::int64_t node_dim =
      graph_case.at("node_feature_dim").get<std::int64_t>();
  const std::int64_t edge_dim =
      graph_case.at("edge_feature_dim").get<std::int64_t>();
  const std::int64_t global_dim =
      graph_case.at("global_feature_dim").get<std::int64_t>();

  // These feature dimensions are part of this particular learned example. If
  // the generator changes the graph contract, the C++ parity test should fail
  // and be updated deliberately.
  CATCH_REQUIRE(node_dim == kNodeFeatureDim);
  CATCH_REQUIRE(edge_dim == kEdgeFeatureDim);
  CATCH_REQUIRE(global_dim == kGlobalFeatureDim);

  const json& tensors = graph_case.at("tensors");

  // edge_index is the only integer tensor. It must remain int64 because the
  // TorchScript model canonicalizes and indexes with 64-bit node ids.
  auto node_features = readTensorValues<float>(tensors.at("node_features"),
                                               "float32",
                                               {num_nodes, node_dim});
  auto edge_index = readTensorValues<std::int64_t>(tensors.at("edge_index"),
                                                   "int64",
                                                   {2, num_edges});
  auto edge_features = readTensorValues<float>(tensors.at("edge_features"),
                                               "float32",
                                               {num_edges, edge_dim});
  auto global_features = readTensorValues<float>(tensors.at("global_features"),
                                                 "float32",
                                                 {global_dim});

  return AMSHomogeneousGraph(
      makeTensor<float>({toDim(num_nodes), toDim(node_dim)}, node_features),
      makeTensor<std::int64_t>({2, toDim(num_edges)}, edge_index),
      makeTensor<float>({toDim(num_edges), toDim(edge_dim)}, edge_features),
      makeTensor<float>({toDim(global_dim)}, global_features));
}

static std::vector<float> loadReferenceDeltaU(const json& graph_case)
{
  // Reference output is one scalar delta_u per node. It was produced in Python
  // by reloading the exported TorchScript model, so it is the closest available
  // representation of what LibTorch should compute in AMS.
  const std::int64_t num_nodes = graph_case.at("num_nodes").get<std::int64_t>();
  const std::int64_t output_dim =
      graph_case.at("reference_output_dim").get<std::int64_t>();
  CATCH_REQUIRE(output_dim == kReferenceOutputDim);
  return readTensorValues<float>(graph_case.at("tensors").at("reference_delta_"
                                                             "u"),
                                 "float32",
                                 {num_nodes, output_dim});
}

static void verifyDeltaU(const json& graph_case,
                         const std::vector<float>& reference_delta_u,
                         const AMSHomogeneousGraphFields& outputs,
                         double rtol,
                         double atol)
{
  // The reference is the Python TorchScript output, not the exact synthetic
  // diffusion target. Training mode already checks whether the model learned
  // the synthetic formula. This test is about AMS/LibTorch deployment parity.
  const std::int64_t num_nodes = graph_case.at("num_nodes").get<std::int64_t>();
  const std::int64_t output_dim =
      graph_case.at("reference_output_dim").get<std::int64_t>();
  CATCH_REQUIRE(output_dim == kReferenceOutputDim);

  CATCH_REQUIRE(outputs.node_fields.contains("delta_u"));

  // The Python model returns "node:delta_u". AMS strips the "node:" namespace
  // and stores the tensor under outputs.node_fields.at("delta_u").
  const auto& delta_u = outputs.node_fields.at("delta_u");
  CATCH_REQUIRE(delta_u.shape()[0] == num_nodes);
  CATCH_REQUIRE(delta_u.shape()[1] == output_dim);
  CATCH_REQUIRE(reference_delta_u.size() ==
                static_cast<std::size_t>(num_nodes * output_dim));

  const float* actual = delta_u.data<float>();
  const std::int64_t count = num_nodes * output_dim;
  for (std::int64_t i = 0; i < count; ++i) {
    CATCH_REQUIRE(
        actual[i] ==
        Catch::Approx(reference_delta_u[i]).epsilon(rtol).margin(atol));
  }
}

}  // namespace

CATCH_TEST_CASE("AMSExecute homogeneous graph MGN diffusion surrogate",
                "[wf][graph][surrogate][mgn]")
{
  // CMake sets AMS_MGN_DIFFUSION_FIXTURE_DIR to the committed source-tree
  // fixture directory. The manifest embeds graph cases and names the adjacent
  // TorchScript model.
  const std::filesystem::path fixture_dir = AMS_MGN_DIFFUSION_FIXTURE_DIR;
  CATCH_REQUIRE_FALSE(fixture_dir.empty());

  const json manifest = loadManifest(fixture_dir);
  const std::filesystem::path manifest_dir = fixture_dir;

  // The model path also comes from the manifest rather than being hard-coded in
  // the test. That keeps the C++ loader tied to the same artifact table of
  // contents as the embedded tensor references.
  const std::filesystem::path model_path =
      resolveManifestRelativePath(manifest_dir, manifest.at("model"));
  CATCH_REQUIRE(std::filesystem::exists(model_path));
  CATCH_REQUIRE(std::filesystem::is_regular_file(model_path));

  const double rtol = manifest.at("comparison").at("rtol").get<double>();
  const double atol = manifest.at("comparison").at("atol").get<double>();
  CATCH_REQUIRE(manifest.at("cases").size() == 2);

  // Register the exported TorchScript model with AMS and create an executor
  // exactly as an application would before calling AMSExecute.
  //
  // This test runs as a single-process executor, so process_id/world_size are
  // named constants rather than bare 0/1 literals.
  //
  // The threshold argument has no acceptance/rejection meaning for homogeneous
  // graph surrogate execution today: the graph path either runs the model or
  // falls back if the model cannot be used. AMSRegisterAbstractModel still
  // requires a threshold because the API is shared with pointwise surrogates.
  // Use 1.0 here as a readable "accept everything" value for that unused
  // graph-path parameter.
  //
  // The store_data argument is also inherited from the shared pointwise
  // workflow API. It controls whether an AMSWorkflow opens a training-data DB,
  // but this graph parity test is inference-only and requires the fallback
  // callback to remain unused. Current graph fallback data storage is also not
  // implemented, so false keeps the test from opening unrelated DB state.
  AMSInit();
  const std::string model_path_string = model_path.string();
  auto model = AMSRegisterAbstractModel(kDomainName,
                                        kUnusedGraphSurrogateThreshold,
                                        model_path_string.c_str(),
                                        kStoreTrainingData);
  AMSExecutor executor =
      AMSCreateExecutor(model, kSingleRankProcessId, kSingleRankWorldSize);

  // The two fixed fixture sizes intentionally have different N and E values.
  // Running both cases catches accidental static-shape assumptions in the
  // exported TorchScript model or in AMS graph tensor handling.
  //
  // N is checked explicitly here because this test is meant to prove the model
  // handles both fixture graph sizes, not just "whatever happened to be in the
  // manifest".
  const std::vector<std::int64_t> expected_node_counts = {24, 73};
  std::size_t case_index = 0;
  for (const json& graph_case : manifest.at("cases")) {
    CATCH_REQUIRE(graph_case.at("num_nodes").get<std::int64_t>() ==
                  expected_node_counts.at(case_index));
    CATCH_DYNAMIC_SECTION("fixture "
                          << graph_case.at("name").get<std::string>())
    {
      // Load one graph case from JSON, then run it through AMS. Each case has
      // its own N, E, input tensors, and reference output.
      AMSHomogeneousGraph graph = makeGraph(graph_case);
      std::vector<float> reference_delta_u = loadReferenceDeltaU(graph_case);

      bool callback_invoked = false;
      // If the surrogate path fails, AMS would call the domain fallback. For
      // this parity test that would hide a deployment failure, so the callback
      // records whether it was invoked and supplies the reference only as a
      // valid fallback value.
      //
      // The assertion below requires callback_invoked == false. That means AMS
      // accepted the graph surrogate output instead of falling back to this
      // lambda.
      HomogeneousGraphDomainFn callback =
          [&](const AMSHomogeneousGraph&, AMSHomogeneousGraphFields& outputs) {
            callback_invoked = true;
            const std::int64_t num_nodes =
                graph_case.at("num_nodes").get<std::int64_t>();
            const std::int64_t output_dim =
                graph_case.at("reference_output_dim").get<std::int64_t>();
            outputs.node_fields.set("delta_u",
                                    makeTensor<float>({toDim(num_nodes),
                                                       toDim(output_dim)},
                                                      reference_delta_u));
          };

      AMSHomogeneousGraphFields outputs;
      AMSExecute(executor, callback, graph, outputs);

      CATCH_REQUIRE_FALSE(callback_invoked);
      verifyDeltaU(graph_case, reference_delta_u, outputs, rtol, atol);
    }
    ++case_index;
  }
}
