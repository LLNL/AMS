#pragma once

#if defined(__AMS_ENABLE_TORCH__)

#include <ATen/ATen.h>
#include <torch/script.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "wf/index_map.hpp"
#include "wf/layout_transform.hpp"
#include "wf/tensor_bundle.hpp"

namespace ams
{

/// PointwiseConcatTransform:
///
/// Converts Inputs + InOuts into a single matrix [N, SUM(K_i)] where:
///   - N = batch size (outer dim)
///   - K_i = flattened size of each tensor field except the batch dimension
///
/// Supports:
///   ✔ Scalar fields (shape [N])
///   ✔ Multi-channel fields (shape [N, K])
///   ✔ Arbitrary shapes [N, ...] → flattened to [N, M]
///   ✔ Prediction-only models
///   ✔ Uncertainty-aware models returning (pred, uncertainty)
///
/// Produces IndexMap for both pack() and unpack().
class PointwiseConcatTransform : public LayoutTransform
{
public:
  const char* name() const noexcept override
  {
    return "PointwiseConcatTransform";
  }

  // ------------------------------------------------------------------
  // PACK
  // ------------------------------------------------------------------
  AMSExpected<IndexMap> pack(const TensorBundle& Inputs,
                             const TensorBundle& InOuts,
                             at::Tensor& ModelInput) override
  {
    IndexMap map;
    std::vector<at::Tensor> cols;
    int total_cols{0};

    if (auto st = process(
            Inputs, IndexMap::FieldInfo::Kind::Input, map, cols, total_cols);
        !st)
      return tl::unexpected(st.error());
    if (auto st = process(
            InOuts, IndexMap::FieldInfo::Kind::InOut, map, cols, total_cols);
        !st)
      return tl::unexpected(st.error());

    if (total_cols <= 0) {
      return AMS_MAKE_ERROR(AMSErrorType::InvalidShapes,
                            fmt::format("PointwiseConcatTransform expected at "
                                        "least a single dimension in pack"));
    }
    // Concatenate horizontally
    ModelInput = at::cat(cols, /*dim=*/1);
    return map;
  }

  // ------------------------------------------------------------------
  // UNPACK
  // ------------------------------------------------------------------
  AMSStatus unpack(const torch::jit::IValue& ModelOutput,
                   TensorBundle& Outs,
                   TensorBundle& InOuts,
                   std::optional<at::Tensor>& Uncertainties) override
  {
    at::Tensor ModelOut;
    at::Tensor Uncertainty;
    bool has_uncertainty = false;

    // --------------------------------------------
    // Case 1: Single tensor prediction
    // --------------------------------------------
    if (ModelOutput.isTensor()) {
      ModelOut = ModelOutput.toTensor();
    }
    // --------------------------------------------
    // Case 2: Tuple(pred, uncertainty)
    // --------------------------------------------
    else if (ModelOutput.isTuple()) {
      auto tup = ModelOutput.toTuple();
      if (tup->elements().size() != 2)
        return AMS_MAKE_ERROR(AMSErrorType::InvalidShapes,
                              "PointwiseConcatTransform: expected "
                              "tuple(pred,uncertainty).");

      ModelOut = tup->elements()[0].toTensor();
      Uncertainty = tup->elements()[1].toTensor();
      has_uncertainty = true;
    } else {
      return AMS_MAKE_ERROR(AMSErrorType::InvalidShapes,
                            "PointwiseConcatTransform: ModelOutput must be "
                            "tensor or "
                            "tuple.");
    }

    // Uncertainties
    if (has_uncertainty) {
      Uncertainties = Uncertainty;
    } else {
      Uncertainties.reset();
    }

    if (ModelOut.size(1) != Outs.size() + InOuts.size())
      return AMS_MAKE_ERROR(AMSErrorType::InvalidShapes,
                            "Expected the output size to match the Application "
                            "output dimensions");

    int k = 0;
    for (; k < Outs.size(); ++k) {
      Outs[k].tensor =
          ModelOut.narrow(/*dim=*/1, /*start=*/k, /*length=*/1).squeeze();
    }

    for (int i = 0; i < InOuts.size(); ++k, ++i) {
      InOuts[i].tensor =
          ModelOut.narrow(/*dim=*/1, /*start=*/k, /*length=*/1).squeeze();
    }

    return {};
  }

private:
  AMSStatus process(const TensorBundle& tb,
                    IndexMap::FieldInfo::Kind kind,
                    IndexMap& map,
                    std::vector<at::Tensor>& cols,
                    int& total_cols)
  {
    for (size_t i = 0; i < tb.size(); i++) {
      const auto& item = tb.items[i];
      at::Tensor t = item.tensor;

      if (t.dim() < 1)
        return AMS_MAKE_ERROR(AMSErrorType::InvalidShapes,
                              fmt::format("PointwiseConcatTransform for "
                                          "field {} must have at least 1 "
                                          "dimension",
                                          item.name));
      int64_t N = t.size(0);

      // Flatten everything except outer dimension.
      at::Tensor flat = t.reshape({N, -1});
      int64_t M = flat.size(1);

      int64_t offset = total_cols;
      total_cols += M;

      map.Fields.push_back({item.name, kind, offset, M});

      cols.push_back(flat);
    }
    return {};
  }
};

}  // namespace ams
#endif  // __AMS_ENABLE_TORCH__