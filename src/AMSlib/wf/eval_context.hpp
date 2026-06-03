#pragma once

#if defined(__AMS_ENABLE_TORCH__)
#include <ATen/ATen.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "wf/tensor_bundle.hpp"

namespace ams
{
namespace ml
{
class InferenceModel;  // forward declaration
}
class LayoutTransform;  // forward declaration

/// EvalContext is the shared state for all Actions executed during
/// an AMS evaluation pipeline. It contains user-provided tensors,
/// model references, layout handlers, and intermediate storage.
///
/// This structure intentionally contains no behavior. All semantics
/// are implemented by Actions operating on EvalContext.
struct EvalContext {

  // ------------------------------------------------------------------
  // User-provided data
  // ------------------------------------------------------------------
  TensorBundle Inputs;   ///< Pure inputs (not modified)
  TensorBundle Inouts;   ///< Tensors modified in-place by evaluation
  TensorBundle Outputs;  ///< Pure outputs written by the model or fallback

  // ------------------------------------------------------------------
  // Model and control configuration
  // ------------------------------------------------------------------
  const ams::ml::InferenceModel* Model =
      nullptr;                        ///< Surrogate model, may be null
  LayoutTransform* Layout = nullptr;  ///< Layout transform handler
  std::optional<float> Threshold;     ///< Uncertainty threshold (if used)

  // ------------------------------------------------------------------
  // Intermediate tensors
  // ------------------------------------------------------------------
  at::Tensor ModelInput;   ///< Model-side input tensor
  at::Tensor ModelOutput;  ///< Model-side output tensor
  std::optional<at::Tensor>
      Uncertainties;  ///< Uncertainty predictions if the model produces them

  // ------------------------------------------------------------------
  // Fallback control and indices
  // ------------------------------------------------------------------
  std::vector<int64_t> FallbackIndices;  ///< Samples requiring fallback

  // ------------------------------------------------------------------
  // Constructors
  // ------------------------------------------------------------------
  EvalContext() = default;

  EvalContext(TensorBundle inputs,
              TensorBundle inouts,
              TensorBundle outputs,
              const ams::ml::InferenceModel* model,
              LayoutTransform* layout,
              std::optional<float> threshold)
      : Inputs(std::move(inputs)),
        Inouts(std::move(inouts)),
        Outputs(std::move(outputs)),
        Model(model),
        Layout(layout),
        Threshold(std::move(threshold))
  {
  }
};

}  // namespace ams
#endif // __AMS_ENABLE_TORCH__