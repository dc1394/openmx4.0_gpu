#ifndef OPENMX_MIXING_POLICY_H
#define OPENMX_MIXING_POLICY_H
#include "openmx_core_mixing.h"
#include <stdio.h>
#include <stdlib.h>

/* Include after the native sgn/largest declarations. These operations are
   supplied on the stack for each call, never retained by the C++ core. */
static inline double OpenMX_Mixing_Sign(void *context,double value)
{ (void)context; return sgn(value); }
static inline double OpenMX_Mixing_Maximum(void *context,double a,double b)
{ (void)context; return largest(a,b); }
static inline double OpenMX_Mixing_Clamp(void *context,double candidate,double minimum,
                                        double maximum,int *upper_hit)
{
  double weight;
  (void)context;
  if(candidate<maximum) {
    if(minimum<candidate)weight=candidate;
    else weight=minimum;
  }
  else weight=maximum;
  *upper_hit=!(candidate<maximum);
  return weight;
}

static inline void OpenMX_Mixing_Shift(OpenMXMixingHistory *history)
{
  char error[160];
  if(!openmx_core_mixing_shift(history,error,sizeof(error))) {
    fprintf(stderr,"mixing history shift: %s\n",error);
    abort();
  }
}

static inline void OpenMX_Mixing_Adapt(OpenMXMixingHistory *history,OpenMXMixingPolicy policy,
                                      double minimum,double maximum)
{
  OpenMXMixingScalarOperations operations;
  char error[160];
  operations.context=NULL;
  operations.sign=OpenMX_Mixing_Sign;
  operations.maximum=OpenMX_Mixing_Maximum;
  operations.clamp=OpenMX_Mixing_Clamp;
  if(!openmx_core_mixing_adapt(history,policy,minimum,maximum,&operations,error,sizeof(error))) {
    fprintf(stderr,"adaptive mixing weight: %s\n",error);
    abort();
  }
}

/* Keep bound preparation at its original position before residual reduction.
   DIIS has fixed limits and does not use this adaptive ceiling. */
static inline double OpenMX_Mixing_PrepareLimit(OpenMXMixingHistory *history,double configured_maximum)
{
  if(history->consecutive==-1) {
    history->adaptive_max_weight=configured_maximum;
    return configured_maximum;
  }
  if(history->consecutive==1000) {
    history->adaptive_max_weight=2.0*history->adaptive_max_weight;
    if(0.7<history->adaptive_max_weight)history->adaptive_max_weight=0.7;
    history->consecutive=0;
  }
  return history->adaptive_max_weight;
}
#endif
