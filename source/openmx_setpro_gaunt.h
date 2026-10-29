#ifndef OPENMX_SETPRO_GAUNT_H
#define OPENMX_SETPRO_GAUNT_H
/* Invocation-owned immutable Gaunt tables. The preparation outlives all CPU
   workers and both VNA23 orientations. GPU-only paths allocate no host table. */
#include "openmx_core_vna_preparation.h"
#include "openmx_setpro_host.h"
typedef OpenMXGauntTableData SetProGauntTable;

static inline size_t SetPro_GauntTable_Row(const SetProGauntTable *t,int L0,int M0,int LL,int L1)
{
  return openmx_gaunt_table_row_offset(t,L0,M0,LL,L1);
}
static inline void SetPro_GauntTable_Borrow(SetProGauntTable *t,
    const OpenMXVnaPreparation *owner,enum OpenMXVnaGauntKind kind)
{
  char error[160];
  if(!openmx_core_vna_gaunt_table(owner,kind,t,error,sizeof(error))) {
    SetPro_gpu_abort(error);abort();
  }
}
static inline double SetPro_GauntTable_At(const SetProGauntTable *t,int L0,int M0,int L1,int M1,int LL)
{
  return t->values[SetPro_GauntTable_Row(t,L0,M0,LL,L1)+(size_t)(L1+M1)];
}
#endif
