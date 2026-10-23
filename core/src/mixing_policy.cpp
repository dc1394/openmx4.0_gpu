#include <openmx/core/mixing_policy.hpp>
#include <openmx/core/tensor.hpp>
#include <cstdio>
#include <exception>
#include <stdexcept>

namespace openmx::core {
void shift_mixing_history(OpenMXMixingHistory& history) noexcept
{
  using view=md::mdspan<double,md::extents<std::size_t,5>>;
  const view norm{history.norm},energy{history.energy};
  for(int i=4;i>=1;--i) {
    element(norm,i)=element(norm,i-1);
    element(energy,i)=element(energy,i-1);
  }
}

void adapt_mixing_history(OpenMXMixingHistory& history,OpenMXMixingPolicy policy,
    double minimum,double maximum,const OpenMXMixingScalarOperations& operations)
{
  if(policy!=OPENMX_MIXING_DENSITY_RATIO && policy!=OPENMX_MIXING_KERKER_POTENTIAL_GAIN)
    throw std::invalid_argument("invalid adaptive mixing policy");
  if(!operations.sign || !operations.maximum || !operations.clamp)
    throw std::invalid_argument("missing native mixing scalar operations");
  using view=md::mdspan<const double,md::extents<std::size_t,5>>;
  const view norm{history.norm},energy{history.energy};
  double weight=history.weight;
  int consecutive=history.consecutive;
  const auto sign=[&](double x){return static_cast<int>(operations.sign(operations.context,x));};
  const auto ratio=[&](bool falling){
    const double difference=falling ? element(norm,1)-element(norm,0)
                                    : element(norm,1)+element(norm,0);
    return element(norm,1)/operations.maximum(operations.context,difference,10e-10)*weight;
  };
  const auto clamp=[&](double candidate,bool falling){
    int upper_hit=0;
    weight=operations.clamp(operations.context,candidate,minimum,maximum,&upper_hit);
    if(falling && upper_hit)++consecutive;
    if(!falling)consecutive=-1;
  };
  // Keep all four energy/residual trend cases. Kerker potential mixing has
  // different gains in each; DIIS potential mixing instead uses the ratio.
  // Primitives run before publishing either value, so exceptions are atomic.
  if(sign(element(energy,0)-element(energy,1))==sign(element(energy,1)-element(energy,2))
      && element(norm,0)<element(norm,1)) {
    clamp(policy==OPENMX_MIXING_DENSITY_RATIO ? ratio(true) : 1.6*weight,true);
  } else if(sign(element(energy,0)-element(energy,1))==sign(element(energy,1)-element(energy,2))
      && element(norm,1)<element(norm,0)) {
    clamp(policy==OPENMX_MIXING_DENSITY_RATIO ? ratio(false) : weight/1.6,false);
  } else if(sign(element(energy,0)-element(energy,1))!=sign(element(energy,1)-element(energy,2))
      && element(norm,0)<element(norm,1)) {
    clamp(policy==OPENMX_MIXING_DENSITY_RATIO ? ratio(true) : 1.2*weight,true);
  } else if(sign(element(energy,0)-element(energy,1))!=sign(element(energy,1)-element(energy,2))
      && element(norm,1)<element(norm,0)) {
    clamp(policy==OPENMX_MIXING_DENSITY_RATIO ? ratio(false) : weight/2.0,false);
  }
  history.weight=weight;
  history.consecutive=consecutive;
}
} // namespace openmx::core

extern "C" int openmx_core_mixing_shift(OpenMXMixingHistory* history,char* error,size_t capacity)
{
  if(!history) {
    if(error && capacity)std::snprintf(error,capacity,"missing mixing history");
    return 0;
  }
  openmx::core::shift_mixing_history(*history);
  if(error && capacity)*error='\0';
  return 1;
}

extern "C" int openmx_core_mixing_adapt(OpenMXMixingHistory* history,int policy,
    double minimum,double maximum,const OpenMXMixingScalarOperations* operations,
    char* error,size_t capacity)
{
  try {
    if(!history || !operations)throw std::invalid_argument("missing mixing history/operations");
    if(policy!=OPENMX_MIXING_DENSITY_RATIO && policy!=OPENMX_MIXING_KERKER_POTENTIAL_GAIN)
      throw std::invalid_argument("invalid adaptive mixing policy");
    openmx::core::adapt_mixing_history(*history,static_cast<OpenMXMixingPolicy>(policy),minimum,maximum,*operations);
    if(error && capacity)*error='\0';
    return 1;
  } catch(const std::exception& exception) {
    if(error && capacity)std::snprintf(error,capacity,"%s",exception.what());
  } catch(...) {
    if(error && capacity)std::snprintf(error,capacity,"native mixing scalar operation failed");
  }
  return 0;
}
