#pragma once

#if defined(__AMS_ENABLE_TORCH__)
#include "wf/pipeline.hpp"

namespace ams
{

namespace ml
{
class InferenceModel;
}

class LayoutTransform;

/// Policies are factories that construct Pipelines.
///
/// A Policy encodes *what* should happen (control flow, fallback strategy),
/// while the Pipeline and Actions encode *how* it happens.
class Policy
{
public:
  virtual ~Policy() = default;

  /// Construct a pipeline for the given model and layout. The, potentially
  /// nullable, Model is a non-owning pointer.
  ///
  /// The returned Pipeline is ready to run.
  virtual Pipeline makePipeline(const ml::InferenceModel* Model,
                                LayoutTransform& Layout) const = 0;
  virtual const char* name() const noexcept = 0;
};

}  // namespace ams
#endif  // __AMS_ENABLE_TORCH__