#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#include <version>

#if !defined(OPENMX_CORE_FORCE_KOKKOS_MDSPAN) && defined(__cpp_lib_mdspan) && __cpp_lib_mdspan >= 202207L
#include <mdspan>
#define OPENMX_CORE_STD_MDSPAN 1
#else
#include <mdspan/mdspan.hpp>
#define OPENMX_CORE_STD_MDSPAN 0
#endif

namespace openmx::core {

#if OPENMX_CORE_STD_MDSPAN
namespace md = std;
#else
namespace md = Kokkos;
#endif

/* Host storage has a single owner and a contiguous allocation. Views never
   cache an owning object's address, so copy/move cannot leave stale row
   pointers behind. As with std::span, views expire when their owner does. */
template<class T, std::size_t Rank>
class Tensor {
 public:
  using extents_type = md::dextents<std::size_t, Rank>;
  using view_type = md::mdspan<T, extents_type, md::layout_right>;
  using const_view_type = md::mdspan<const T, extents_type, md::layout_right>;

  explicit Tensor(const std::array<std::size_t, Rank>& shape)
      : shape_(shape), storage_(checked_size(shape)) {}

  Tensor(const Tensor&) = default;
  Tensor& operator=(const Tensor& other) {
    if (this != &other) {
      // Preserve the shape/storage invariant if allocation or copying throws.
      Tensor replacement(other);
      shape_.swap(replacement.shape_);
      storage_.swap(replacement.storage_);
    }
    return *this;
  }
  Tensor(Tensor&& other) noexcept
      : shape_(std::exchange(other.shape_, {})), storage_(std::move(other.storage_)) {}
  Tensor& operator=(Tensor&& other) noexcept {
    if (this != &other) {
      shape_ = std::exchange(other.shape_, {});
      storage_ = std::move(other.storage_);
    }
    return *this;
  }

  view_type view() { return {storage_.data(), extents_type(shape_)}; }
  const_view_type view() const { return {storage_.data(), extents_type(shape_)}; }
  T* data() noexcept { return storage_.data(); }
  const T* data() const noexcept { return storage_.data(); }
  std::size_t size() const noexcept { return storage_.size(); }
  std::size_t extent(std::size_t axis) const { return shape_.at(axis); }

 private:
  static std::size_t checked_size(const std::array<std::size_t, Rank>& shape) {
    std::size_t size = 1;
    for (auto extent : shape) {
      if (extent && size > std::numeric_limits<std::size_t>::max()/extent)
        throw std::length_error("tensor extent product overflows size_t");
      size *= extent;
    }
    if (size > std::vector<T>().max_size())
      throw std::length_error("tensor exceeds allocator maximum size");
    return size;
  }
  std::array<std::size_t, Rank> shape_;
  std::vector<T> storage_;
};

/* The standard mapping/accessor interface works with both std::mdspan and
   the Kokkos C++20 backport, irrespective of subscript syntax support. */
template<class View, class... Indices>
decltype(auto) element(const View& view, Indices... indices) {
  return view.accessor().access(view.data_handle(), view.mapping()(indices...));
}

} // namespace openmx::core
