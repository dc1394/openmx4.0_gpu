#include "openmx_core_vna_invocation.h"
#include <openmx/core/tensor.hpp>
#include <cmath>
#include <cstdio>
#include <memory>

struct OpenMXVnaInvocation {
  OpenMXVnaInvocation(const OpenMXVnaInvocationInput& input) : wavevectors({checked_points(input)}) {
    char error[256];
    preparation.reset(openmx_core_vna_preparation_create(&input.basis,&views.preparation,error,sizeof(error)));
    if (!preparation) throw std::runtime_error(error);
    pairs.reset(openmx_core_atom_pairs_create(&input.geometry,&views.pairs,error,sizeof(error)));
    if (!pairs) throw std::runtime_error(error);
    const double dk=input.maximum_wavevector-input.minimum_wavevector;
    const double sk=input.maximum_wavevector+input.minimum_wavevector;
    for (int i=0;i<input.basis.points;++i) {
      if (!std::isfinite(input.abscissae[i])) throw std::invalid_argument("nonfinite VNA abscissa");
      wavevectors.data()[i]=0.5*(dk*input.abscissae[i]+sk);
      if (!std::isfinite(wavevectors.data()[i])) throw std::invalid_argument("VNA wavevector overflow");
    }
    views.preparation_owner=preparation.get();views.wavevectors=wavevectors.data();
  }
  static std::size_t checked_points(const OpenMXVnaInvocationInput& input) {
    if (input.basis.points<1 || !input.abscissae || input.basis.species!=input.geometry.species ||
        !std::isfinite(input.minimum_wavevector) || !std::isfinite(input.maximum_wavevector) ||
        input.minimum_wavevector<0 || input.maximum_wavevector<input.minimum_wavevector)
      throw std::invalid_argument("invalid VNA invocation grid or species");
    return static_cast<std::size_t>(input.basis.points);
  }
  std::unique_ptr<OpenMXVnaPreparation,decltype(&openmx_core_vna_preparation_destroy)> preparation{
      nullptr,openmx_core_vna_preparation_destroy};
  std::unique_ptr<OpenMXAtomPairPlan,decltype(&openmx_core_atom_pairs_destroy)> pairs{
      nullptr,openmx_core_atom_pairs_destroy};
  openmx::core::Tensor<double,1> wavevectors;
  OpenMXVnaInvocationBuffers views{};
};

extern "C" OpenMXVnaInvocation *openmx_core_vna_invocation_create(const OpenMXVnaInvocationInput *input,
    OpenMXVnaInvocationBuffers *buffers,char *error,std::size_t capacity)
{
  if (error && capacity) error[0]='\0';
  try {
    if (!input || !buffers) throw std::invalid_argument("null VNA invocation input or views");
    auto result=std::make_unique<OpenMXVnaInvocation>(*input);
    *buffers=result->views;
    return result.release();
  }
  catch (const std::exception& e) { if (error && capacity) std::snprintf(error,capacity,"%s",e.what()); }
  catch (...) { if (error && capacity) std::snprintf(error,capacity,"unknown VNA invocation failure"); }
  return nullptr;
}
extern "C" void openmx_core_vna_invocation_destroy(OpenMXVnaInvocation *invocation) { delete invocation; }
