#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "AMS.h"
#include "AMSGraph.hpp"
#include "AMSTensor.hpp"
#include "ams_tensor_test_utils.hpp"
#include "nlohmann/json.hpp"

using namespace ams;
using ams::test::makeTensor;
namespace fs = std::filesystem;

namespace
{
template <typename T>
void requireBinaryValues(const fs::path& root,
                         const nlohmann::json& descriptor,
                         const std::vector<T>& expected)
{
  CATCH_REQUIRE(descriptor["byte_size"] == expected.size() * sizeof(T));
  std::ifstream input(root / descriptor["path"].get<std::string>(),
                      std::ios::binary);
  CATCH_REQUIRE(input.is_open());
  std::vector<T> actual(expected.size());
  input.read(reinterpret_cast<char*>(actual.data()),
             static_cast<std::streamsize>(actual.size() * sizeof(T)));
  CATCH_REQUIRE(input.gcount() ==
                static_cast<std::streamsize>(expected.size() * sizeof(T)));
  CATCH_REQUIRE(actual == expected);
}

nlohmann::json readManifest(const fs::path& path)
{
  std::ifstream input(path);
  CATCH_REQUIRE(input.is_open());
  nlohmann::json manifest;
  input >> manifest;
  return manifest;
}

AMSHomogeneousGraph makeGraph()
{
  return AMSHomogeneousGraph(
      makeTensor<float>({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}),
      makeTensor<int64_t>({2, 2}, {0, 1, 1, 2}),
      makeTensor<float>({2, 1}, {0.25f, 0.5f}),
      makeTensor<float>({2}, {7.0f, 8.0f}));
}
}  // namespace

CATCH_TEST_CASE("Torch-disabled AMSExecute uses workflow fallback and storage",
                "[ams][workflow][no-torch]")
{
  AMSInit();

  {
    AMSCAbstrModel model =
        AMSRegisterAbstractModel("no_torch_without_database", 0.5, "", true);
    AMSExecutor executor = AMSCreateExecutor(model, 0, 1);

    SmallVector<AMSTensor> inputs;
    SmallVector<AMSTensor> inouts;
    SmallVector<AMSTensor> outputs;

    int callback_count = 0;
    DomainLambda physics = [&](const SmallVector<AMSTensor>&,
                               SmallVector<AMSTensor>&,
                               SmallVector<AMSTensor>&) { ++callback_count; };

    CATCH_REQUIRE_NOTHROW(
        AMSExecute(executor, physics, inputs, inouts, outputs));
    CATCH_REQUIRE(callback_count == 1);
    CATCH_REQUIRE(AMSGetDatabaseName(executor).empty());
    AMSDestroyExecutor(executor);
  }

  const fs::path root =
      fs::temp_directory_path() / "ams_no_torch_workflow_storage";
  fs::remove_all(root);
  fs::create_directories(root);
  setenv("AMS_JSON_MODE", "binary", 1);
  AMSConfigureFSDatabase(AMSDBType::AMS_JSON, root.string().c_str());

  {
    AMSCAbstrModel model =
        AMSRegisterAbstractModel("no_torch_tensor_storage", 0.5, "", true);
    AMSExecutor executor = AMSCreateExecutor(model, 0, 1);
    const fs::path manifest_path = AMSGetDatabaseName(executor);

    SmallVector<AMSTensor> inputs;
    SmallVector<AMSTensor> inouts;
    SmallVector<AMSTensor> outputs;
    inputs.push_back(makeTensor<float>({2, 1}, {1.0f, 2.0f}));
    inouts.push_back(makeTensor<float>({2, 1}, {10.0f, 20.0f}));
    outputs.push_back(makeTensor<float>({2, 1}, {0.0f, 0.0f}));

    int callback_count = 0;
    DomainLambda physics = [&](const SmallVector<AMSTensor>& callback_inputs,
                               SmallVector<AMSTensor>& callback_inouts,
                               SmallVector<AMSTensor>& callback_outputs) {
      ++callback_count;
      for (int i = 0; i < 2; ++i) {
        callback_outputs[0].data<float>()[i] =
            callback_inputs[0].data<float>()[i] +
            callback_inouts[0].data<float>()[i];
        callback_inouts[0].data<float>()[i] += 5.0f;
      }
    };

    AMSExecute(executor, physics, inputs, inouts, outputs);
    CATCH_REQUIRE(callback_count == 1);
    CATCH_REQUIRE(outputs[0].data<float>()[0] == 11.0f);
    CATCH_REQUIRE(outputs[0].data<float>()[1] == 22.0f);
    CATCH_REQUIRE(inouts[0].data<float>()[0] == 15.0f);
    CATCH_REQUIRE(inouts[0].data<float>()[1] == 25.0f);

    AMSDestroyExecutor(executor);
    const nlohmann::json manifest = readManifest(manifest_path);
    CATCH_REQUIRE(manifest["cases"].size() == 1);
    const auto& tensors = manifest["cases"][0]["tensors"];
    requireBinaryValues<float>(root, tensors["input_0"], {1.0f, 2.0f});
    requireBinaryValues<float>(root, tensors["input_1"], {10.0f, 20.0f});
    requireBinaryValues<float>(root, tensors["output_0"], {11.0f, 22.0f});
    requireBinaryValues<float>(root, tensors["output_1"], {15.0f, 25.0f});
  }

  {
    AMSCAbstrModel model =
        AMSRegisterAbstractModel("no_torch_graph_storage", 0.5, "", true);
    AMSExecutor executor = AMSCreateExecutor(model, 0, 1);
    const fs::path manifest_path = AMSGetDatabaseName(executor);
    AMSHomogeneousGraph graph = makeGraph();
    AMSHomogeneousGraphFields outputs;

    int callback_count = 0;
    HomogeneousGraphDomainFn physics =
        [&](const AMSHomogeneousGraph& callback_graph,
            AMSHomogeneousGraphFields& callback_outputs) {
          ++callback_count;
          CATCH_REQUIRE(callback_graph.node_features.shape()[0] == 3);
          callback_outputs.node_fields.set(
              "prediction", makeTensor<float>({3, 1}, {2.0f, 4.0f, 6.0f}));
        };

    AMSExecute(executor, physics, graph, outputs);
    CATCH_REQUIRE(callback_count == 1);
    CATCH_REQUIRE(outputs.node_fields.at("prediction").data<float>()[1] ==
                  4.0f);

    AMSDestroyExecutor(executor);
    const nlohmann::json manifest = readManifest(manifest_path);
    CATCH_REQUIRE(manifest["cases"].size() == 1);
    CATCH_REQUIRE(
        manifest["cases"][0]["outputs"]["node"].contains("prediction"));
  }

  fs::remove_all(root);
}
