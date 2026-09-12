#include "wf/tensor_bundle.hpp"

#include <ATen/ATen.h>

#include <catch2/catch_test_macros.hpp>

CATCH_TEST_CASE("TensorBundle basic construction", "[tensorbundle]")
{
  ams::TensorBundle tb;

  CATCH_REQUIRE(tb.size() == 0);
  CATCH_REQUIRE(tb.empty());
}

CATCH_TEST_CASE("TensorBundle add and access items", "[tensorbundle]")
{
  ams::TensorBundle tb;

  at::Tensor t1 = at::ones({3});
  at::Tensor t2 = at::zeros({2});

  tb.add("a", t1);
  tb.add("b", t2);

  CATCH_REQUIRE(tb.size() == 2);
  CATCH_REQUIRE_FALSE(tb.empty());

  CATCH_REQUIRE(tb[0].name == "a");
  CATCH_REQUIRE(tb[1].name == "b");

  CATCH_REQUIRE(tb[0].tensor.equal(t1));
  CATCH_REQUIRE(tb[1].tensor.equal(t2));
}

CATCH_TEST_CASE("TensorBundle iteration works", "[tensorbundle]")
{
  ams::TensorBundle tb;
  tb.add("x", at::full({1}, 42));
  tb.add("y", at::full({1}, 13));

  std::vector<std::string> names;
  for (auto& item : tb) {
    names.push_back(item.name);
  }

  CATCH_REQUIRE(names.size() == 2);
  CATCH_REQUIRE(names[0] == "x");
  CATCH_REQUIRE(names[1] == "y");
}

CATCH_TEST_CASE("TensorBundle copy semantics", "[tensorbundle]")
{
  ams::TensorBundle tb;
  tb.add("z", at::ones({5}));

  ams::TensorBundle tb2 = tb;  // copy

  CATCH_REQUIRE(tb2.size() == 1);
  CATCH_REQUIRE(tb2[0].name == "z");
  CATCH_REQUIRE(tb2[0].tensor.equal(tb[0].tensor));
}

CATCH_TEST_CASE("TensorBundle move semantics", "[tensorbundle]")
{
  ams::TensorBundle tb;
  tb.add("m", at::rand({4}));

  at::Tensor original = tb[0].tensor;

  ams::TensorBundle tb2 = std::move(tb);

  CATCH_REQUIRE(tb2.size() == 1);
  CATCH_REQUIRE(tb2[0].name == "m");
  CATCH_REQUIRE(tb2[0].tensor.equal(original));

  // moved-from tb should be valid but empty
  CATCH_REQUIRE(tb.size() == 0);
  CATCH_REQUIRE(tb.empty());
}

CATCH_TEST_CASE("TensorBundle clear()", "[tensorbundle]")
{
  ams::TensorBundle tb;

  tb.add("a", at::rand({1}));
  tb.add("b", at::rand({1}));

  CATCH_REQUIRE(tb.size() == 2);

  tb.clear();

  CATCH_REQUIRE(tb.size() == 0);
  CATCH_REQUIRE(tb.empty());
}

CATCH_TEST_CASE("TensorBundle duplicate names are rejected", "[tensorbundle]")
{
  ams::TensorBundle tb;

  tb.add("x", at::ones({2}));
  CATCH_REQUIRE(tb.size() == 1);

  // Adding a tensor with the same name should throw
  CATCH_REQUIRE_THROWS_AS(tb.add("x", at::zeros({3})), std::invalid_argument);

  // Bundle should still have only the first tensor
  CATCH_REQUIRE(tb.size() == 1);
  CATCH_REQUIRE(tb[0].name == "x");
}

CATCH_TEST_CASE("TensorBundle contains() method", "[tensorbundle]")
{
  ams::TensorBundle tb;

  tb.add("alpha", at::ones({1}));
  tb.add("beta", at::zeros({1}));

  CATCH_REQUIRE(tb.contains("alpha"));
  CATCH_REQUIRE(tb.contains("beta"));
  CATCH_REQUIRE_FALSE(tb.contains("gamma"));
  CATCH_REQUIRE_FALSE(tb.contains(""));
}

CATCH_TEST_CASE("TensorBundle find() method", "[tensorbundle]")
{
  ams::TensorBundle tb;

  at::Tensor t1 = at::full({3}, 42);
  at::Tensor t2 = at::full({2}, 13);

  tb.add("foo", t1);
  tb.add("bar", t2);

  // Find existing items
  auto* item1 = tb.find("foo");
  CATCH_REQUIRE(item1 != nullptr);
  CATCH_REQUIRE(item1->name == "foo");
  CATCH_REQUIRE(item1->tensor.equal(t1));

  auto* item2 = tb.find("bar");
  CATCH_REQUIRE(item2 != nullptr);
  CATCH_REQUIRE(item2->name == "bar");
  CATCH_REQUIRE(item2->tensor.equal(t2));

  // Find non-existing item
  auto* item3 = tb.find("baz");
  CATCH_REQUIRE(item3 == nullptr);
}

CATCH_TEST_CASE("TensorBundle find() const method", "[tensorbundle]")
{
  ams::TensorBundle tb;
  tb.add("test", at::ones({5}));

  const ams::TensorBundle& const_tb = tb;

  const auto* item = const_tb.find("test");
  CATCH_REQUIRE(item != nullptr);
  CATCH_REQUIRE(item->name == "test");

  const auto* missing = const_tb.find("missing");
  CATCH_REQUIRE(missing == nullptr);
}

CATCH_TEST_CASE("TensorBundle at() bounds checking", "[tensorbundle]")
{
  ams::TensorBundle tb;
  tb.add("x", at::ones({2}));
  tb.add("y", at::zeros({3}));

  // Valid access should work
  CATCH_REQUIRE(tb.at(0).name == "x");
  CATCH_REQUIRE(tb.at(1).name == "y");

  // Out of bounds access should throw
  CATCH_REQUIRE_THROWS_AS(tb.at(2), std::out_of_range);
  CATCH_REQUIRE_THROWS_AS(tb.at(100), std::out_of_range);
}