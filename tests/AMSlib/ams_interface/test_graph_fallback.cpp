#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "AMS.h"
#include "AMSGraph.hpp"
#include "AMSTensor.hpp"

using namespace ams;

using Dim = AMSTensor::IntDimType;

static std::vector<Dim> contiguousStrides(const std::vector<Dim>& shape)
{
  std::vector<Dim> strides(shape.size(), 1);
  Dim stride = 1;
  for (std::size_t i = shape.size(); i-- > 0;) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

template <typename T>
static AMSTensor makeTensor(std::vector<Dim> shape)
{
  std::vector<Dim> strides = contiguousStrides(shape);
  return AMSTensor::create<T>(shape, strides, AMSResourceType::AMS_HOST);
}

static AMSTensor makeNodeFeatures(Dim nodes = 3, Dim features = 2)
{
  auto tensor = makeTensor<float>({nodes, features});
  float* data = tensor.data<float>();
  for (Dim i = 0; i < tensor.elements(); ++i) {
    data[i] = static_cast<float>(i + 1);
  }
  return tensor;
}

static AMSTensor makeEdgeIndex64(Dim edges = 2)
{
  auto tensor = makeTensor<int64_t>({2, edges});
  int64_t* data = tensor.data<int64_t>();
  for (Dim e = 0; e < edges; ++e) {
    data[e] = e % 3;
    data[edges + e] = (e + 1) % 3;
  }
  return tensor;
}

static AMSTensor makeEdgeIndex32(Dim edges = 2)
{
  auto tensor = makeTensor<int32_t>({2, edges});
  int32_t* data = tensor.data<int32_t>();
  for (Dim e = 0; e < edges; ++e) {
    data[e] = static_cast<int32_t>(e % 3);
    data[edges + e] = static_cast<int32_t>((e + 1) % 3);
  }
  return tensor;
}

static AMSTensor makeEdgeFeatures(Dim edges = 2, Dim features = 1)
{
  auto tensor = makeTensor<float>({edges, features});
  float* data = tensor.data<float>();
  for (Dim i = 0; i < tensor.elements(); ++i) {
    data[i] = 0.5f + static_cast<float>(i);
  }
  return tensor;
}

static AMSTensor makeGlobalFeatures()
{
  auto tensor = makeTensor<float>({2});
  tensor.data<float>()[0] = 0.25f;
  tensor.data<float>()[1] = 0.5f;
  return tensor;
}

static AMSHomogeneousGraph makeValidGraph()
{
  return AMSHomogeneousGraph(makeNodeFeatures(),
                             makeEdgeIndex64(),
                             makeEdgeFeatures(),
                             makeGlobalFeatures());
}

CATCH_TEST_CASE("AMSTensorFieldMap explicit field API", "[wf][graph]")
{
  AMSInit();

  CATCH_STATIC_REQUIRE(!std::is_default_constructible_v<AMSTensor>);

  AMSTensorFieldMap fields;

  fields.set("prediction", makeTensor<float>({2, 1}));
  CATCH_REQUIRE(fields.contains("prediction"));
  CATCH_REQUIRE(fields.find("missing") == nullptr);
  CATCH_REQUIRE(fields.at("prediction").shape()[0] == 2);

  auto replacement = makeTensor<float>({3, 1});
  float* data = replacement.data<float>();
  data[0] = 7.0f;
  data[1] = 8.0f;
  data[2] = 9.0f;
  fields.set("prediction", std::move(replacement));

  const auto& prediction = fields.at("prediction");
  CATCH_REQUIRE(prediction.shape()[0] == 3);
  CATCH_REQUIRE(prediction.data<float>()[0] == 7.0f);
  CATCH_REQUIRE(prediction.data<float>()[2] == 9.0f);

  fields.insert("flux", makeTensor<float>({2, 1}));
  CATCH_REQUIRE_THROWS_AS(fields.insert("flux", makeTensor<float>({2, 1})),
                          std::runtime_error);
  CATCH_REQUIRE_THROWS_AS(fields.at("absent"), std::out_of_range);

  const AMSTensorFieldMap& const_fields = fields;
  CATCH_REQUIRE(const_fields.cbegin() != const_fields.cend());
  std::size_t visited = 0;
  bool saw_prediction = false;
  bool saw_flux = false;
  for (const auto& [name, tensor] : const_fields) {
    ++visited;
    if (name == "prediction") {
      saw_prediction = true;
      CATCH_REQUIRE(tensor.shape()[0] == 3);
    } else if (name == "flux") {
      saw_flux = true;
      CATCH_REQUIRE(tensor.shape()[0] == 2);
    }
  }
  CATCH_REQUIRE(visited == 2);
  CATCH_REQUIRE(saw_prediction);
  CATCH_REQUIRE(saw_flux);
}

CATCH_TEST_CASE("AMSHomogeneousGraph validates construction",
                "[wf][graph][validation]")
{
  AMSInit();

  CATCH_REQUIRE_NOTHROW(makeValidGraph());
  {
    AMSHomogeneousGraph graph(makeNodeFeatures(),
                              makeEdgeIndex32(),
                              makeEdgeFeatures());
    CATCH_REQUIRE(graph.global_features.shape().size() == 1);
    CATCH_REQUIRE(graph.global_features.shape()[0] == 0);
  }
  CATCH_REQUIRE_NOTHROW(AMSHomogeneousGraph(makeNodeFeatures(),
                                            makeEdgeIndex64(),
                                            makeEdgeFeatures(),
                                            makeTensor<float>({0})));
  CATCH_REQUIRE_NOTHROW(AMSHomogeneousGraph(makeNodeFeatures(),
                                            makeEdgeIndex64(),
                                            makeEdgeFeatures(),
                                            makeGlobalFeatures()));

  CATCH_REQUIRE_THROWS_AS(AMSHomogeneousGraph(makeTensor<int64_t>({3, 2}),
                                              makeEdgeIndex64(),
                                              makeEdgeFeatures()),
                          std::runtime_error);
  CATCH_REQUIRE_THROWS_AS(AMSHomogeneousGraph(makeTensor<float>({3}),
                                              makeEdgeIndex64(),
                                              makeEdgeFeatures()),
                          std::runtime_error);
  CATCH_REQUIRE_THROWS_AS(AMSHomogeneousGraph(makeNodeFeatures(),
                                              makeTensor<float>({2, 2}),
                                              makeEdgeFeatures()),
                          std::runtime_error);
  CATCH_REQUIRE_THROWS_AS(AMSHomogeneousGraph(makeNodeFeatures(),
                                              makeTensor<int64_t>({2}),
                                              makeEdgeFeatures()),
                          std::runtime_error);
  CATCH_REQUIRE_THROWS_AS(AMSHomogeneousGraph(makeNodeFeatures(),
                                              makeTensor<int64_t>({3, 2}),
                                              makeEdgeFeatures()),
                          std::runtime_error);
  CATCH_REQUIRE_THROWS_AS(AMSHomogeneousGraph(makeNodeFeatures(),
                                              makeEdgeIndex64(),
                                              makeTensor<float>({2})),
                          std::runtime_error);
  CATCH_REQUIRE_THROWS_AS(AMSHomogeneousGraph(makeNodeFeatures(),
                                              makeEdgeIndex64(),
                                              makeEdgeFeatures(3)),
                          std::runtime_error);
  CATCH_REQUIRE_THROWS_AS(AMSHomogeneousGraph(makeNodeFeatures(),
                                              makeEdgeIndex64(),
                                              makeTensor<int64_t>({2, 1})),
                          std::runtime_error);
  CATCH_REQUIRE_THROWS_AS(AMSHomogeneousGraph(makeNodeFeatures(),
                                              makeEdgeIndex64(),
                                              makeEdgeFeatures(),
                                              makeTensor<float>({1, 2})),
                          std::runtime_error);
  CATCH_REQUIRE_THROWS_AS(AMSHomogeneousGraph(makeNodeFeatures(),
                                              makeEdgeIndex64(),
                                              makeEdgeFeatures(),
                                              makeTensor<float>({2, 1})),
                          std::runtime_error);
  CATCH_REQUIRE_THROWS_AS(AMSHomogeneousGraph(makeNodeFeatures(),
                                              makeEdgeIndex64(),
                                              makeEdgeFeatures(),
                                              makeTensor<int64_t>({1})),
                          std::runtime_error);
}

CATCH_TEST_CASE("AMSExecute homogeneous graph fallback path", "[wf][graph]")
{
  AMSInit();

  auto model =
      AMSRegisterAbstractModel("test_homo_graph_fields", 0.5, "", false);
  AMSExecutor executor = AMSCreateExecutor(model, 0, 1);
  AMSHomogeneousGraph graph = makeValidGraph();

  bool callback_invoked = false;
  HomogeneousGraphDomainFn callback = [&](const AMSHomogeneousGraph& g,
                                          AMSHomogeneousGraphFields& outputs) {
    callback_invoked = true;
    CATCH_REQUIRE(g.node_features.shape()[0] == 3);
    CATCH_REQUIRE(g.edge_index.shape()[0] == 2);

    auto out = makeTensor<float>({3, 1});
    float* out_data = out.data<float>();
    out_data[0] = 2.0f;
    out_data[1] = 4.0f;
    out_data[2] = 6.0f;
    outputs.node_fields.set("prediction", std::move(out));
  };

  AMSHomogeneousGraphFields outputs;
  AMSExecute(executor, callback, graph, outputs);

  CATCH_REQUIRE(callback_invoked);
  const auto& prediction = outputs.node_fields.at("prediction");
  CATCH_REQUIRE(prediction.shape()[0] == 3);
  CATCH_REQUIRE(prediction.data<float>()[1] == 4.0f);
}

CATCH_TEST_CASE("AMSExecute heterogeneous graph fallback path", "[wf][graph]")
{
  AMSInit();

  auto model =
      AMSRegisterAbstractModel("test_hetero_graph_fields", 0.5, "", false);
  AMSExecutor executor = AMSCreateExecutor(model, 0, 1);

  AMSHeterogeneousGraph graph;
  auto& atom_store = graph.getOrCreateNodeStore("atom");
  insertTensor(atom_store, "features", makeNodeFeatures(5, 2));

  auto& edge_store =
      graph.getOrCreateEdgeStore(EdgeType{"atom", "bond", "atom"});
  insertTensor(edge_store, "edge_index", makeEdgeIndex64(4));
  insertTensor(edge_store, "features", makeEdgeFeatures(4, 1));
  insertTensor(graph.global_store, "global", makeGlobalFeatures());

  bool callback_invoked = false;
  HeterogeneousGraphDomainFn callback =
      [&](const AMSHeterogeneousGraph& g,
          AMSHeterogeneousGraphFields& outputs) {
        callback_invoked = true;
        CATCH_REQUIRE(g.containsNodeStore("atom"));
        CATCH_REQUIRE(g.containsEdgeStore(EdgeType{"atom", "bond", "atom"}));

        auto out = makeTensor<float>({5, 1});
        float* out_data = out.data<float>();
        for (int i = 0; i < 5; ++i) {
          out_data[i] = static_cast<float>(i * 3);
        }
        outputs.getOrCreateNodeStore("atom").set("prediction", std::move(out));
      };

  AMSHeterogeneousGraphFields outputs;
  AMSExecute(executor, callback, graph, outputs);

  CATCH_REQUIRE(callback_invoked);
  const auto* atom_outputs = outputs.findNodeStore("atom");
  CATCH_REQUIRE(atom_outputs != nullptr);
  const auto& prediction = atom_outputs->at("prediction");
  CATCH_REQUIRE(prediction.data<float>()[4] == 12.0f);
}

CATCH_TEST_CASE("Graph callback type safety", "[wf][graph]")
{
  HomogeneousGraphDomainFn homo_fn = [](const AMSHomogeneousGraph&,
                                        AMSHomogeneousGraphFields&) {};

  HeterogeneousGraphDomainFn hetero_fn = [](const AMSHeterogeneousGraph&,
                                            AMSHeterogeneousGraphFields&) {};

  (void)homo_fn;
  (void)hetero_fn;
  CATCH_SUCCEED("Type safety validated at compile time");
}
