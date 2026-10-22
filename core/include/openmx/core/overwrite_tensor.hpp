#pragma once
#include <openmx/core/tensor.hpp>
#include <memory>
#include <type_traits>

namespace openmx::core {
/* Contiguous host scratch for algorithms that write every consumed element.
   Unlike Tensor's value-initialized storage, trivial elements are left
   uninitialized, matching a native malloc buffer without a redundant memory
   pass. Reading before writing is invalid. Ownership and mdspan layout remain
   explicit; moving invalidates views, as it does for Tensor. */
template<class T, std::size_t Rank>
class OverwriteTensor {
  static_assert(std::is_trivially_default_constructible_v<T> && std::is_trivially_destructible_v<T>);
 public:
  using extents_type=md::dextents<std::size_t,Rank>;
  using view_type=md::mdspan<T,extents_type,md::layout_right>;
  using const_view_type=md::mdspan<const T,extents_type,md::layout_right>;
  explicit OverwriteTensor(const std::array<std::size_t,Rank>& shape)
      :shape_(shape),size_(checked_size(shape)),storage_(size_?std::make_unique_for_overwrite<T[]>(size_):nullptr) {}
  OverwriteTensor(const OverwriteTensor&)=delete;
  OverwriteTensor& operator=(const OverwriteTensor&)=delete;
  OverwriteTensor(OverwriteTensor&& other) noexcept
      :shape_(std::exchange(other.shape_,{})),size_(std::exchange(other.size_,0)),storage_(std::move(other.storage_)) {}
  OverwriteTensor& operator=(OverwriteTensor&& other) noexcept {
    if(this!=&other){
      shape_=std::exchange(other.shape_,{});size_=std::exchange(other.size_,0);storage_=std::move(other.storage_);
    }
    return *this;
  }
  view_type view() { return {storage_.get(),extents_type(shape_)}; }
  const_view_type view() const { return {storage_.get(),extents_type(shape_)}; }
  T* data() noexcept { return storage_.get(); }
  const T* data() const noexcept { return storage_.get(); }
  std::size_t size() const noexcept { return size_; }
  std::size_t extent(std::size_t axis) const { return shape_.at(axis); }
 private:
  static std::size_t checked_size(const std::array<std::size_t,Rank>& shape) {
    std::size_t count=1;
    for(auto extent:shape){
      if(extent && count>std::numeric_limits<std::size_t>::max()/extent)
        throw std::length_error("overwrite tensor extent product overflows size_t");
      count*=extent;
    }
    if(count>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(T))
      throw std::length_error("overwrite tensor exceeds allocator maximum size");
    return count;
  }
  std::array<std::size_t,Rank> shape_;
  std::size_t size_;
  std::unique_ptr<T[]> storage_;
};
} // namespace openmx::core
