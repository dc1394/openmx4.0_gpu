// Short-range omega-PBE exchange (wPBEh) of one spin channel.  The arithmetic
// and the statement of its source (Quantum ESPRESSO 7.6) are in
// openmx/xc/hybrid_kernels.h, shared with C callers and with the
// application's GPU device code.

#include "wpbeh_kernel.hpp"

#include <openmx/xc/hybrid_kernels.h>

namespace openmx::xc::detail {

ChannelEnergy wpbeh_channel_energy(double density, double gradient_norm, double omega) noexcept
{
  double out[3];
  openmx_xc_kernel_wpbeh_channel(density, gradient_norm, omega, out);
  return {out[0], out[1], out[2]};
}

}  // namespace openmx::xc::detail
