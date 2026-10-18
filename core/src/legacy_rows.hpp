#pragma once

#include <openmx/core/tensor.hpp>

namespace openmx::core::compat {

// Borrowed numeric rows for the legacy C boundary. Owners keep their tensors
// fixed in memory. Only pointer tables are allocated here, never numeric rows.
template<class T> class Rows2 {
 public:
  explicit Rows2(typename Tensor<T,2>::view_type view) : rows_(view.extent(0)) {
    for (std::size_t i = 0; i < rows_.size(); ++i)
      rows_[i] = &element(view,i,0);
  }
  T** data() noexcept { return rows_.data(); }
  Rows2(const Rows2&) = delete;
  Rows2& operator=(const Rows2&) = delete;
  Rows2(Rows2&&) = delete;
  Rows2& operator=(Rows2&&) = delete;
 private:
  std::vector<T*> rows_;
};

template<class T> class Rows3 {
 public:
  explicit Rows3(typename Tensor<T,3>::view_type view)
    : outer_(view.extent(0)), rows_(view.extent(0)*view.extent(1)) {
    for (std::size_t i = 0; i < view.extent(0); ++i) {
      outer_[i] = rows_.data()+i*view.extent(1);
      for (std::size_t j = 0; j < view.extent(1); ++j)
        rows_[i*view.extent(1)+j] = &element(view,i,j,0);
    }
  }
  T*** data() noexcept { return outer_.data(); }
  Rows3(const Rows3&) = delete;
  Rows3& operator=(const Rows3&) = delete;
  Rows3(Rows3&&) = delete;
  Rows3& operator=(Rows3&&) = delete;
 private:
  std::vector<T**> outer_;
  std::vector<T*> rows_;
};

template<class T> class Rows4 {
 public:
  explicit Rows4(typename Tensor<T,4>::view_type view)
    : outer_(view.extent(0)), middle_(view.extent(0)*view.extent(1)),
      rows_(view.extent(0)*view.extent(1)*view.extent(2)) {
    for (std::size_t i = 0; i < view.extent(0); ++i) {
      outer_[i] = middle_.data()+i*view.extent(1);
      for (std::size_t j = 0; j < view.extent(1); ++j) {
        const auto block = i*view.extent(1)+j;
        middle_[block] = rows_.data()+block*view.extent(2);
        for (std::size_t k = 0; k < view.extent(2); ++k)
          rows_[block*view.extent(2)+k] = &element(view,i,j,k,0);
      }
    }
  }
  T**** data() noexcept { return outer_.data(); }
  Rows4(const Rows4&) = delete;
  Rows4& operator=(const Rows4&) = delete;
  Rows4(Rows4&&) = delete;
  Rows4& operator=(Rows4&&) = delete;
 private:
  std::vector<T***> outer_;
  std::vector<T**> middle_;
  std::vector<T*> rows_;
};

} // namespace openmx::core::compat
