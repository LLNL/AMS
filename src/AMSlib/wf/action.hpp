#pragma once

#if defined(__AMS_ENABLE_TORCH__)

#include "AMSError.hpp"

namespace ams
{

struct EvalContext;  // forward declaration

/// Base class for a single step in an AMS evaluation pipeline.
///
/// Actions mutate the shared EvalContext and may fail; failures are reported
/// via AMSStatus so pipelines can short-circuit cleanly.
class Action
{
public:
  virtual ~Action() = default;

  /// Execute this action on the evaluation context.
  virtual AMSStatus run(EvalContext& ctx) = 0;

  /// Human-readable name for debugging, logging, and tracing.
  virtual const char* name() const noexcept = 0;
};

}  // namespace ams
#endif  // __AMS_ENABLE_TORCH__