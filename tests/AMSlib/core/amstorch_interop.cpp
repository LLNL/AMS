#include <torch/torch.h>

#include <catch2/catch_test_macros.hpp>

#include "AMSTorchInterop.hpp"

using namespace ams;

CATCH_TEST_CASE("Torch interop views are zero-copy and retain storage",
                "[ams][torch][interop]")
{
  auto source = torch::arange(6, torch::TensorOptions().dtype(torch::kFloat32))
                    .reshape({2, 3});
  void* pointer = source.data_ptr();
  auto view = fromTorchView(source);
  CATCH_REQUIRE(view.data_ptr() == pointer);
  source = torch::Tensor();
  view.data<float>()[4] = 99.0f;
  CATCH_REQUIRE(view.data<float>()[4] == 99.0f);

  auto torchView = toTorchView(view);
  CATCH_REQUIRE(torchView.data_ptr() == view.data_ptr());
  torchView[0][0] = 17.0f;
  CATCH_REQUIRE(view.data<float>()[0] == 17.0f);
}

CATCH_TEST_CASE("Torch interop copies own independent contiguous storage",
                "[ams][torch][interop]")
{
  auto source = torch::arange(12, torch::TensorOptions().dtype(torch::kInt64))
                    .reshape({3, 4})
                    .transpose(0, 1);
  auto copy = fromTorchCopy(source);
  CATCH_REQUIRE(copy.contiguous());
  CATCH_REQUIRE(copy.data_ptr() != source.data_ptr());
  source[0][0] = 999;
  CATCH_REQUIRE(copy.data<int64_t>()[0] == 0);

  auto torchCopy = toTorchCopy(copy);
  CATCH_REQUIRE(torchCopy.is_contiguous());
  CATCH_REQUIRE(torchCopy.data_ptr() != copy.data_ptr());
  copy.data<int64_t>()[0] = 123;
  CATCH_REQUIRE(torchCopy[0][0].item<int64_t>() == 0);
}

CATCH_TEST_CASE("Torch interop validates unsupported dtype and layout",
                "[ams][torch][interop]")
{
  CATCH_REQUIRE_THROWS_AS(fromTorchView(torch::ones({2}, torch::kBool)),
                          std::invalid_argument);
  auto expanded = torch::ones({1, 3}).expand({4, 3});
  CATCH_REQUIRE_THROWS_AS(fromTorchView(expanded), std::invalid_argument);
}
