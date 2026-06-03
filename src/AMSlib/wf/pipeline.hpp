#pragma once

#if defined(__AMS_ENABLE_TORCH__)

#include <memory>
#include <vector>

#include "AMSError.hpp"   // AMSStatus
#include "wf/action.hpp"  // Action

namespace ams
{

struct EvalContext;

/// A linear sequence of Actions executed in order.
///
/// If any Action fails, execution stops and the error is returned.
class Pipeline
{
public:
  using ActionPtr = std::unique_ptr<Action>;

  Pipeline() = default;

  /// Append an Action to the pipeline.
  Pipeline& add(ActionPtr Act)
  {
    Actions.emplace_back(std::move(Act));
    return *this;
  }

  /// Execute all actions in order; stops on first error.
  AMSStatus run(EvalContext& Ctx) const
  {
    for (const auto& Act : Actions) {
      if (auto St = Act->run(Ctx); !St) {
        return St;
      }
    }
    return {};
  }

  /// Number of actions in the pipeline.
  size_t size() const noexcept { return Actions.size(); }

  /// True if there are no actions.
  bool empty() const noexcept { return Actions.empty(); }

  /// Remove all actions.
  void clear() noexcept { Actions.clear(); }

private:
  std::vector<ActionPtr> Actions;
};

}  // namespace ams
#endif // __AMS_ENABLE_TORCH__