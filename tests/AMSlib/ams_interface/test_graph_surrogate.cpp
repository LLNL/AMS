#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <vector>

#include "AMS.h"
#include "AMSGraph.hpp"
#include "AMSTensor.hpp"
#include "ams_tensor_test_utils.hpp"

using namespace ams;
using ams::test::makeTensor;

using Dim = AMSTensor::IntDimType;

static const char* HOMOGENEOUS_GRAPH_MODEL_PATH =
    "../models/homogeneous_graph.pt";
static const char* HETEROGENEOUS_GRAPH_MODEL_PATH =
    "../models/heterogeneous_graph.pt";
static const char* BAD_KEY_GRAPH_MODEL_PATH =
    "../models/homogeneous_graph_bad_key.pt";
static const char* BAD_SHAPE_GRAPH_MODEL_PATH =
    "../models/homogeneous_graph_bad_shape.pt";

static AMSTensor makeMessageNodeFeatures()
{
  auto tensor = makeTensor<float>({4, 2});
  float* data = tensor.template data<float>();
  data[0] = 1.0f;
  data[1] = 10.0f;
  data[2] = 2.0f;
  data[3] = 20.0f;
  data[4] = 3.0f;
  data[5] = 30.0f;
  data[6] = 4.0f;
  data[7] = 40.0f;
  return tensor;
}

template <typename T>
static AMSTensor makeMessageEdgeIndex()
{
  auto tensor = makeTensor<T>({2, 5});
  T* data = tensor.template data<T>();
  data[0] = static_cast<T>(0);
  data[1] = static_cast<T>(1);
  data[2] = static_cast<T>(2);
  data[3] = static_cast<T>(0);
  data[4] = static_cast<T>(3);
  data[5] = static_cast<T>(1);
  data[6] = static_cast<T>(2);
  data[7] = static_cast<T>(3);
  data[8] = static_cast<T>(3);
  data[9] = static_cast<T>(0);
  return tensor;
}

static AMSTensor makeMessageEdgeFeatures()
{
  auto tensor = makeTensor<float>({5, 1});
  float* data = tensor.template data<float>();
  data[0] = 0.5f;
  data[1] = 1.0f;
  data[2] = 1.5f;
  data[3] = 2.0f;
  data[4] = 0.25f;
  return tensor;
}

static AMSTensor makeMessageGlobalFeatures()
{
  auto tensor = makeTensor<float>({1});
  tensor.data<float>()[0] = 0.125f;
  return tensor;
}

template <typename EdgeScalar>
static AMSHomogeneousGraph makeMessageGraph()
{
  return AMSHomogeneousGraph(makeMessageNodeFeatures(),
                             makeMessageEdgeIndex<EdgeScalar>(),
                             makeMessageEdgeFeatures(),
                             makeMessageGlobalFeatures());
}

template <typename EdgeScalar>
static AMSHomogeneousGraph makeMessageGraphWithoutGlobals()
{
  return AMSHomogeneousGraph(makeMessageNodeFeatures(),
                             makeMessageEdgeIndex<EdgeScalar>(),
                             makeMessageEdgeFeatures());
}

static void verifyMessagePrediction(const AMSHomogeneousGraphFields& outputs)
{
  CATCH_REQUIRE(outputs.node_fields.contains("prediction"));
  const auto& prediction = outputs.node_fields.at("prediction");
  CATCH_REQUIRE(prediction.shape()[0] == 4);
  CATCH_REQUIRE(prediction.shape()[1] == 1);

  const float expected[] = {2.0f, 2.5f, 5.0f, 10.5f};
  const float* data = prediction.data<float>();
  for (int i = 0; i < 4; ++i) {
    CATCH_REQUIRE(data[i] == Catch::Approx(expected[i]));
  }
}

template <typename EdgeScalar>
static void runHomogeneousSurrogate(const char* domain_name)
{
  auto model = AMSRegisterAbstractModel(domain_name,
                                        0.5,
                                        HOMOGENEOUS_GRAPH_MODEL_PATH,
                                        false);
  AMSExecutor executor = AMSCreateExecutor(model, 0, 1);
  AMSHomogeneousGraph graph = makeMessageGraph<EdgeScalar>();

  bool callback_invoked = false;
  HomogeneousGraphDomainFn callback = [&](const AMSHomogeneousGraph&,
                                          AMSHomogeneousGraphFields& outputs) {
    callback_invoked = true;
    outputs.node_fields.set("prediction", makeTensor<float>({4, 1}));
  };

  AMSHomogeneousGraphFields outputs;
  AMSExecute(executor, callback, graph, outputs);

  CATCH_REQUIRE_FALSE(callback_invoked);
  verifyMessagePrediction(outputs);
}

CATCH_TEST_CASE("AMSExecute homogeneous graph surrogate message passing",
                "[wf][graph][surrogate]")
{
  AMSInit();

  runHomogeneousSurrogate<int64_t>("test_homo_surrogate_message_int64");
  runHomogeneousSurrogate<int32_t>("test_homo_surrogate_message_int32");
}

CATCH_TEST_CASE("AMSExecute homogeneous graph surrogate without globals",
                "[wf][graph][surrogate]")
{
  AMSInit();

  auto model = AMSRegisterAbstractModel("test_homo_surrogate_no_globals",
                                        0.5,
                                        HOMOGENEOUS_GRAPH_MODEL_PATH,
                                        false);
  AMSExecutor executor = AMSCreateExecutor(model, 0, 1);
  AMSHomogeneousGraph graph = makeMessageGraphWithoutGlobals<int64_t>();
  CATCH_REQUIRE(graph.global_features.shape().size() == 1);
  CATCH_REQUIRE(graph.global_features.shape()[0] == 0);

  bool callback_invoked = false;
  HomogeneousGraphDomainFn callback = [&](const AMSHomogeneousGraph&,
                                          AMSHomogeneousGraphFields& outputs) {
    callback_invoked = true;
    outputs.node_fields.set("prediction", makeTensor<float>({4, 1}));
  };

  AMSHomogeneousGraphFields outputs;
  AMSExecute(executor, callback, graph, outputs);

  CATCH_REQUIRE_FALSE(callback_invoked);
  verifyMessagePrediction(outputs);
}

CATCH_TEST_CASE("AMSExecute heterogeneous graph surrogate execution",
                "[wf][graph][surrogate]")
{
  AMSInit();

  auto model = AMSRegisterAbstractModel("test_hetero_surrogate_fields",
                                        0.5,
                                        HETEROGENEOUS_GRAPH_MODEL_PATH,
                                        false);
  AMSExecutor executor = AMSCreateExecutor(model, 0, 1);

  AMSHeterogeneousGraph graph;
  auto& node_store = graph.getOrCreateNodeStore("node");
  auto node_features = makeTensor<float>({10, 16});
  float* node_data = node_features.data<float>();
  for (int i = 0; i < 160; ++i) {
    node_data[i] = static_cast<float>(i) * 0.1f;
  }
  insertTensor(node_store, "x", std::move(node_features));

  auto& edge_store =
      graph.getOrCreateEdgeStore(EdgeType{"node", "edge", "node"});
  insertTensor(edge_store, "edge_index", makeMessageEdgeIndex<int64_t>());
  insertTensor(edge_store, "features", makeMessageEdgeFeatures());
  insertTensor(graph.global_store, "dummy", makeMessageGlobalFeatures());

  bool callback_invoked = false;
  HeterogeneousGraphDomainFn callback =
      [&](const AMSHeterogeneousGraph&, AMSHeterogeneousGraphFields& outputs) {
        callback_invoked = true;
        outputs.getOrCreateNodeStore("node").set("prediction",
                                                 makeTensor<float>({10, 8}));
      };

  AMSHeterogeneousGraphFields outputs;
  AMSExecute(executor, callback, graph, outputs);

  CATCH_REQUIRE_FALSE(callback_invoked);
  const auto* node_outputs = outputs.findNodeStore("node");
  CATCH_REQUIRE(node_outputs != nullptr);
  const auto& prediction = node_outputs->at("prediction");
  CATCH_REQUIRE(prediction.shape()[0] == 10);
  CATCH_REQUIRE(prediction.shape()[1] == 8);
}

CATCH_TEST_CASE("Graph surrogate with no model triggers fallback",
                "[wf][graph][surrogate][fallback]")
{
  AMSInit();

  auto model =
      AMSRegisterAbstractModel("test_no_model_graph_fields", 0.5, "", false);
  AMSExecutor executor = AMSCreateExecutor(model, 0, 1);
  AMSHomogeneousGraph graph = makeMessageGraph<int64_t>();

  bool callback_invoked = false;
  HomogeneousGraphDomainFn callback = [&](const AMSHomogeneousGraph&,
                                          AMSHomogeneousGraphFields& outputs) {
    callback_invoked = true;
    auto out = makeTensor<float>({4, 1});
    out.data<float>()[0] = 42.0f;
    outputs.node_fields.set("prediction", std::move(out));
  };

  AMSHomogeneousGraphFields outputs;
  AMSExecute(executor, callback, graph, outputs);

  CATCH_REQUIRE(callback_invoked);
  CATCH_REQUIRE(outputs.node_fields.at("prediction").data<float>()[0] == 42.0f);
}

CATCH_TEST_CASE("Malformed homogeneous graph surrogate outputs fail loudly",
                "[wf][graph][surrogate][failure]")
{
  AMSInit();

  HomogeneousGraphDomainFn callback = [](const AMSHomogeneousGraph&,
                                         AMSHomogeneousGraphFields&) {};

  {
    auto model = AMSRegisterAbstractModel("test_bad_graph_key",
                                          0.5,
                                          BAD_KEY_GRAPH_MODEL_PATH,
                                          false);
    AMSExecutor executor = AMSCreateExecutor(model, 0, 1);
    AMSHomogeneousGraph graph = makeMessageGraph<int64_t>();
    AMSHomogeneousGraphFields outputs;
    CATCH_REQUIRE_THROWS_AS(AMSExecute(executor, callback, graph, outputs),
                            std::runtime_error);
  }

  {
    auto model = AMSRegisterAbstractModel("test_bad_graph_shape",
                                          0.5,
                                          BAD_SHAPE_GRAPH_MODEL_PATH,
                                          false);
    AMSExecutor executor = AMSCreateExecutor(model, 0, 1);
    AMSHomogeneousGraph graph = makeMessageGraph<int64_t>();
    AMSHomogeneousGraphFields outputs;
    CATCH_REQUIRE_THROWS_AS(AMSExecute(executor, callback, graph, outputs),
                            std::runtime_error);
  }
}
