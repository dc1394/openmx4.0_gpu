#ifndef OPENMX_XC_SRC_WPBEH_KERNEL_HPP
#define OPENMX_XC_SRC_WPBEH_KERNEL_HPP

// Internal to the xc library: the short-range omega-PBE exchange of one spin
// channel.  The arithmetic is in wpbeh_kernel.cpp, which carries its own
// licence notice (MPL 2.0); functionals.cpp adds the density floors and the
// output conventions of the PBE kernel.

namespace openmx::xc::detail {

struct ChannelEnergy {
  double energy;              // short-range exchange energy per volume of the channel
  double denergy_ddensity;    // derivative with respect to the spin density
  double denergy_dgradient;   // derivative with respect to the norm of its gradient
};

// density > 0 is the spin density rho_s, gradient_norm >= 0 is |grad rho_s|,
// omega >= 0 is the screening parameter in 1/bohr.
ChannelEnergy wpbeh_channel_energy(double density, double gradient_norm, double omega) noexcept;

}  // namespace openmx::xc::detail

#endif
