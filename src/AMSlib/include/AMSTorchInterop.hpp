#pragma once

#include <torch/torch.h>

#include "AMSTensor.hpp"

namespace ams
{
/** Zero-copy view retaining the Torch tensor's storage owner. */
AMSTensor fromTorchView(torch::Tensor tensor);

/** Independent, contiguous AMS-managed copy. */
AMSTensor fromTorchCopy(const torch::Tensor& tensor);

/** Zero-copy mutable Torch view retaining AMS-managed storage when present. */
torch::Tensor toTorchView(AMSTensor& tensor);

/** Independent, contiguous Torch-managed copy. */
torch::Tensor toTorchCopy(const AMSTensor& tensor);
}  // namespace ams
