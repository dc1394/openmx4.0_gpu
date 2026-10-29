#pragma once
#include <openmx/basis/real_harmonics.h>
#include <complex>
#include <span>
#include <vector>

namespace openmx::basis {

// Owns the unitary complex-to-real spherical-harmonic transformation for
// degrees 0..maximum_degree (0..64). Columns correspond to m=-l,...,+l and
// include the Condon-Shortley phase used by integral_harmonic().
// Rows: s; p=(x,y,z); d=(z2,x2-y2,xy,xz,yz); for l>=3:
// m=0, cos(phi), sin(phi), cos(2phi), sin(2phi),...,cos(l phi),sin(l phi).
// The positive real Cartesian signs and the inherited OpenMX order are kept.
// All storage is contiguous, read-only after construction and independent of
// OpenMX/MPI/Kokkos initialization. Copies own independent data. Views expire
// on destruction or assignment; concurrent readers do not share mutable state.
class RealHarmonicTransform {
 public:
  explicit RealHarmonicTransform(int maximum_degree);
  RealHarmonicTransform(const RealHarmonicTransform&)=default;
  RealHarmonicTransform(RealHarmonicTransform&& other) noexcept;
  RealHarmonicTransform& operator=(const RealHarmonicTransform& other);
  RealHarmonicTransform& operator=(RealHarmonicTransform&& other) noexcept;

  int maximum_degree() const noexcept { return maximum_; }
  OpenMXRealHarmonicData view() const noexcept;
  std::span<const dcomplex> coefficients() const noexcept { return maximum_<0?std::span<const dcomplex>{}:std::span<const dcomplex>{storage_}; }
  // Checked active indices: degree<=maximum_degree and row,column in [0,2*l].
  const dcomplex& coefficient(int degree, int row, int column) const;

  // T_l times a vector of complex harmonics. Both spans must contain exactly
  // 2*l+1 values. Arbitrary overlapping spans, including in-place use, are
  // supported through bounded stack scratch; no heap allocation. Validation
  // failure preserves output. General input vectors may give complex output.
  void transform(int degree, std::span<const std::complex<double>> input,
      std::span<std::complex<double>> output) const;

 private:
  std::vector<dcomplex> storage_;
  int maximum_=-1;
};
} // namespace openmx::basis
