#pragma once
#include <openmx/core/local_basis_state.hpp>
#include "openmx_core_local_basis.h"

namespace openmx::core::compat {
class LocalBasisOwner {
 public:
  explicit LocalBasisOwner(const OpenMXLocalBasisInput&);
  OpenMXLocalBasisViews views() noexcept;

 private:
  template<class T> struct MatrixRows {
    std::vector<T*> rows;
    std::vector<T**> atoms;
    std::vector<T***> components;
    T empty{};
    void bind(AtomBlockMatrix<T>&);
  };
  struct TensorRows {
    std::vector<double*> rows;
    std::vector<double**> planes;
    double empty{};
    void bind(Tensor<double,3>&);
  };
  std::unique_ptr<LocalBasisState> state_;
  std::array<MatrixRows<double>,OPENMX_LOCAL_REAL_COUNT> real_;
  std::array<MatrixRows<dcomplex>,OPENMX_LOCAL_COMPLEX_COUNT> complex_;
  std::array<TensorRows,OPENMX_LOCAL_TENSOR_COUNT> tensors_;
  std::vector<double****> density_groups_;
  std::vector<dcomplex****> occupation_spins_,potential_spins_;
  std::vector<dcomplex*****> occupation_history_;
};
} // namespace openmx::core::compat
