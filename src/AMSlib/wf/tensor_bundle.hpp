#pragma once

#if defined(__AMS_ENABLE_TORCH__)

#include <ATen/ATen.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ams
{

/// A lightweight container that groups named tensors together.
/// This is the primary structure used to represent inputs,
/// in-out parameters, and outputs inside AMS evaluation pipelines.
struct TensorBundle {

  /// A single named tensor.
  struct Item {
    std::string name;
    at::Tensor tensor;

    Item(std::string n, at::Tensor t) : name(std::move(n)), tensor(std::move(t))
    {
    }
  };

  /// Ordered list of items.
  std::vector<Item> items;

  /// Default construction.
  TensorBundle() = default;

  /// Move operations for efficiency.
  TensorBundle(TensorBundle&&) noexcept = default;
  TensorBundle& operator=(TensorBundle&&) noexcept = default;

  /// Copy operations allowed (torch::Tensor has cheap refcounted semantics).
  TensorBundle(const TensorBundle&) = default;
  TensorBundle& operator=(const TensorBundle&) = default;

  /// Add a named tensor to the bundle.
  /// Throws std::invalid_argument if a tensor with the same name already exists.
  void add(std::string name, at::Tensor t)
  {
    if (contains(name)) {
      throw std::invalid_argument(
          "TensorBundle already contains a tensor named '" + name + "'");
    }
    items.emplace_back(std::move(name), std::move(t));
  }

  /// Check if a tensor with the given name exists in the bundle.
  /// Note: This performs a linear search (O(n) complexity).
  bool contains(const std::string& name) const noexcept
  {
    return std::any_of(items.begin(), items.end(), [&name](const Item& item) {
      return item.name == name;
    });
  }

  /// Find a tensor by name. Returns nullptr if not found.
  /// Note: This performs a linear search (O(n) complexity).
  Item* find(const std::string& name) noexcept
  {
    auto it =
        std::find_if(items.begin(), items.end(), [&name](const Item& item) {
          return item.name == name;
        });
    return it != items.end() ? &(*it) : nullptr;
  }

  /// Find a tensor by name (const version). Returns nullptr if not found.
  /// Note: This performs a linear search (O(n) complexity).
  const Item* find(const std::string& name) const noexcept
  {
    auto it =
        std::find_if(items.begin(), items.end(), [&name](const Item& item) {
          return item.name == name;
        });
    return it != items.end() ? &(*it) : nullptr;
  }

  /// Number of tensors in the bundle.
  size_t size() const noexcept { return items.size(); }

  /// Random access to items (unchecked).
  /// Callers must ensure 0 <= i < size() to avoid undefined behavior.
  Item& operator[](size_t i) noexcept { return items[i]; }

  const Item& operator[](size_t i) const noexcept { return items[i]; }

  /// Bounds-checked random access to items.
  /// Throws std::out_of_range if i >= size().
  Item& at(size_t i) { return items.at(i); }

  const Item& at(size_t i) const { return items.at(i); }

  /// Iterators.
  auto begin() noexcept { return items.begin(); }
  auto end() noexcept { return items.end(); }
  auto begin() const noexcept { return items.begin(); }
  auto end() const noexcept { return items.end(); }

  /// Check if empty.
  bool empty() const noexcept { return items.empty(); }

  /// Remove all items.
  void clear() noexcept { items.clear(); }
};

}  // namespace ams
#endif  // __AMS_ENABLE_TORCH__