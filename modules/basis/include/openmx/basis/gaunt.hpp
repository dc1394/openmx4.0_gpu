#pragma once
#include <openmx/basis/gaunt.h>
#include <span>
#include <vector>

namespace openmx::basis {

struct HarmonicIndex { int degree, order; };

// Integral conj(Y_bra)*Y_left*Y_right dOmega. Same normalization and phase as
// integral_harmonic(). Integer degrees 0..38, |order|<=degree. For allowed
// triangles, the degree sum must be <=38; other high degrees represent zeros.
// The bound preserves the original double factorial sum's representable range;
// this API does not promise stable arbitrary high-order angular-momentum algebra.
// Valid calls allocate nothing and share no mutable state. Invalid indices throw.
double gaunt(HarmonicIndex bra, HarmonicIndex left, HarmonicIndex right);

// Owns G(l0,m0,l1,m1,LL,m0-m1) for all active indices. Selection-rule zeros
// outside |m0-m1|<=LL retain the original arithmetic, including signed zero.
// Maxima are 0..38; every allowed triangle must have degree sum<=38. Larger
// rectangular selection-zero bounds are supported. Copies own independent storage.
// Immutable after construction; concurrent readers are safe. Views expire on
// destruction or assignment. A moved-from object exposes an empty view.
class GauntTable {
 public:
  GauntTable(int l0max, int l1max, int llmax);
  GauntTable(const GauntTable&)=default;
  GauntTable(GauntTable&& other) noexcept;
  GauntTable& operator=(const GauntTable& other);
  GauntTable& operator=(GauntTable&& other) noexcept;
  OpenMXGauntTableData view() const noexcept;
  std::span<const double> values() const noexcept;
  // Checked active row. Values are ordered m1=-l1,...,+l1 without padding.
  std::span<const double> row(int l0, int m0, int ll, int l1) const;
  double coefficient(int l0, int m0, int l1, int m1, int ll) const;
 private:
  std::vector<double> storage_;
  int l0max_=-1, l1max_=-1, llmax_=-1;
};
} // namespace openmx::basis
