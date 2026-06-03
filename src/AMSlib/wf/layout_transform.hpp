#pragma once

#if defined(__AMS_ENABLE_TORCH__)

#include <ATen/ATen.h>
#include <torch/script.h>  // for torch::jit::IValue

#include <optional>

#include "AMSError.hpp"
#include "wf/index_map.hpp"
#include "wf/tensor_bundle.hpp"

namespace ams
{

/// Abstract base class describing how AMS transforms application-level data
/// (Inputs, Inouts, Outputs) into contiguous model inputs and vice versa.
///
/// - pack() produces the tensor that is fed into the surrogate model.
/// - unpack() receives the model output (an IValue that may contain multiple
///   tensors) and maps it back into Outputs, Inouts, and optionally Uncertainties.
///
/// The AMS pipeline never assumes any particular layout; all shape and packing
/// logic lives in concrete LayoutTransform implementations.
class LayoutTransform
{
public:
  virtual ~LayoutTransform() = default;

  virtual AMSExpected<IndexMap> pack(const TensorBundle& Inputs,
                                     const TensorBundle& InOuts,
                                     at::Tensor& ModelInput) = 0;

  virtual AMSStatus unpack(const torch::jit::IValue& ModelOutput,
                           TensorBundle& Outs,
                           TensorBundle& InOuts,
                           std::optional<at::Tensor>& Uncertainties) = 0;

  virtual const char* name() const noexcept = 0;
};

}  // namespace ams
#endif // __AMS_ENABLE_TORCH__