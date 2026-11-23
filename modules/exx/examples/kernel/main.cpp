// The interaction kernels of the exact exchange in reciprocal space: the
// short-range kernel of HSE06 (omega = 0.11 bohr^-1) and the Coulomb kernel
// truncated at R_c = 12 bohr, next to the bare Coulomb kernel.
#include <openmx/exx/kernel.hpp>

#include <cstdio>
#include <initializer_list>

int main()
{
  using namespace openmx::exx;
  const Kernel bare = coulomb_kernel();
  const Kernel hse = short_range_kernel(0.11);
  const Kernel truncated = truncated_coulomb_kernel(12.0);
  std::printf("kappa (1/bohr)   4 pi/kappa^2    short range     truncated\n");
  for (double kappa : {0.0,0.05,0.1,0.2,0.5,1.0,2.0,5.0})
    std::printf("%10.3f %16.6f %14.6f %14.6f\n",kappa,kernel_value(bare,kappa),
                kernel_value(hse,kappa),kernel_value(truncated,kappa));
  std::printf("limits at kappa = 0: %.6f and %.6f\n",kernel_limit(hse),kernel_limit(truncated));
  return 0;
}
