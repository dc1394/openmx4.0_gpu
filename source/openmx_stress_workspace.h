#ifndef OPENMX_STRESS_WORKSPACE_H
#define OPENMX_STRESS_WORKSPACE_H
#include "openmx_mechanical_workspace.h"
#include "openmx_xc_workspace.h"

/* Nine active components per worker. A 256-byte row separates their writes
   even for 128-byte cache lines; unused padding is never read or zeroed.
   Prepare only between joined worker regions. All stages reuse this slot. */
static inline void OpenMX_Stress_ThreadsPrepare(OpenMXSolverInvocation *owner,
    int workers,OpenMXSolverRealTensor *out)
{
  char error[256];const size_t shape[4]={(size_t)workers,32,1,1};
  if(workers<1)OpenMX_Solver_Failure("invalid stress worker count");
  if(!openmx_core_solver_real_overwrite(owner,omp_get_thread_num(),"stress:thread-sums",
        shape,out,error,sizeof(error)))OpenMX_Solver_Failure(error);
}

/* FFT/MPI interfaces retain a nonnull row on empty ranks. Even columns
   preserve the alignment of both borrowed real/imaginary rows. */
static inline void OpenMX_Stress_ReciprocalPrepare(OpenMXSolverInvocation *owner,
    int points,OpenMXSolverRealTensor *out)
{
  char error[256];const size_t columns=OpenMX_Solver_SimdColumns(points>0?(size_t)points:1);
  const size_t shape[4]={2,columns,1,1};
  if(points<0)OpenMX_Solver_Failure("invalid stress reciprocal grid count");
  if(!openmx_core_solver_real_overwrite(owner,omp_get_thread_num(),"stress:hartree-density",
        shape,out,error,sizeof(error)))OpenMX_Solver_Failure(error);
}

typedef struct {
  OpenMXSolverRealTensor potential,derivatives,displacements;
} OpenMXStressGridScratch;

static inline void OpenMX_Stress_OverwritePrepare(OpenMXSolverInvocation *owner,
    const char *name,const size_t shape[4],OpenMXSolverRealTensor *out)
{
  char error[256];
  if(!openmx_core_solver_real_overwrite(owner,omp_get_thread_num(),name,
        shape,out,error,sizeof(error)))OpenMX_Solver_Failure(error);
}

/* Main grid caches are shared across joined worksharing regions; only
   active points/orbitals are produced and consumed. Empty shapes keep a
   nonnull row for native interfaces. No numeric values are initialized. */
static inline void OpenMX_Stress_GridPrepare(OpenMXSolverInvocation *owner,
    int spin,int points,int orbitals,OpenMXStressGridScratch *out)
{
  if((spin!=0 && spin!=1 && spin!=3)||points<0||orbitals<1)
    OpenMX_Solver_Failure("invalid stress grid shape");
  const size_t count=points>0?(size_t)points:1;
  const size_t potential[4]={(size_t)spin+1,OpenMX_Solver_SimdColumns(count),1,1};
  const size_t derivatives[4]={count,(size_t)orbitals,3,1};
  const size_t displacements[4]={count,3,1,1};
  OpenMX_Stress_OverwritePrepare(owner,"stress:grid:potential",potential,&out->potential);
  OpenMX_Stress_OverwritePrepare(owner,"stress:grid:derivatives",derivatives,&out->derivatives);
  OpenMX_Stress_OverwritePrepare(owner,"stress:grid:displacements",displacements,&out->displacements);
}
static inline double *OpenMX_Stress_DerivativeRow(const OpenMXSolverRealTensor *view,
    size_t point,size_t orbital)
{
  return view->values+(point*view->extent[1]+orbital)*view->extent[2];
}

/* Four fixed borrowed rows adapt the existing orbital producer ABI. */
static inline void OpenMX_Stress_OrbitalPrepare(OpenMXSolverInvocation *owner,
    int orbitals,OpenMXSolverRealTensor *out)
{
  if(orbitals<1)OpenMX_Solver_Failure("invalid stress orbital count");
  const size_t shape[4]={4,OpenMX_Solver_SimdColumns((size_t)orbitals),1,1};
  OpenMX_Stress_OverwritePrepare(owner,"stress:grid:orbital-rows",shape,out);
}

/* [field (XC derivative/density gradient),spin,axis,grid] replaces four
   pointer trees. Bind the fixed legacy D rows only for Set_XC_Grid. */
static inline void OpenMX_Stress_GradientPrepare(OpenMXSolverInvocation *owner,
    const char *name,int points,OpenMXSolverRealTensor *out)
{
  if(points<0)OpenMX_Solver_Failure("invalid stress gradient grid count");
  const size_t shape[4]={2,2,3,OpenMX_Solver_SimdColumns(points>0?(size_t)points:1)};
  OpenMX_Stress_OverwritePrepare(owner,name,shape,out);
}
static inline double *OpenMX_Stress_GradientRow(const OpenMXSolverRealTensor *view,
    size_t field,size_t spin,size_t axis)
{
  return view->values+((field*view->extent[1]+spin)*view->extent[2]+axis)*view->extent[3];
}

/* Populate explicit outputs inside native OpenMP; no large aggregate return.
   Each pair retains the complete old zero loop before kinetic assembly. */
static inline void OpenMX_Stress_KineticPrepare(OpenMXSolverInvocation *owner,
    const char *name,int orbitals,OpenMXSolverComplexTensor *out)
{
  if(orbitals<1)OpenMX_Solver_Failure("invalid stress kinetic orbital count");
  const size_t shape[4]={3,(size_t)orbitals,(size_t)orbitals,1};char error[256];
  if(!openmx_core_solver_complex_overwrite(owner,omp_get_thread_num(),name,
        shape,out,error,sizeof(error)))OpenMX_Solver_Failure(error);
}
/* MPI retains a nonnull pointer on empty messages. Rebind only after the
   corresponding send has completed; receive storage uses a separate key. */
static inline double *OpenMX_Stress_HaloBuffer(OpenMXSolverInvocation *owner,
    const char *name,int count)
{
  if(count<0)OpenMX_Solver_Failure("invalid stress halo message count");
  return OpenMX_Solver_OverwriteReal(owner,name,count>0?(size_t)count:1);
}

/* Active VNA lists retain their two legacy padding elements. */
static inline void OpenMX_Stress_VnaListsPrepare(OpenMXSolverInvocation *owner,
    int radial,int maximum_angular,int **angular_list,int **radial_list)
{
  if(radial<0 || maximum_angular<0)OpenMX_Solver_Failure("invalid stress VNA projector bounds");
  const size_t channels=(size_t)maximum_angular+1;
  if((size_t)radial>((size_t)-1-2)/channels)
    OpenMX_Solver_Failure("stress VNA projector list size overflow");
  const size_t count=(size_t)radial*channels+2;
  *angular_list=OpenMX_Solver_Int(owner,"stress:vna:angular-list",count);
  *radial_list=OpenMX_Solver_Int(owner,"stress:vna:radial-list",count);
}

/* One worker reuses these fields across joined VNA stages. Values are
   uninitialized until the derivative producer writes its active rectangle. */
static inline void OpenMX_Stress_VnaPrepare(OpenMXSolverInvocation *owner,int orbitals,
    OpenMXSolverRealTensor *x,OpenMXSolverRealTensor *y,
    OpenMXSolverRealTensor *z,OpenMXSolverRealTensor *strain)
{
  if(orbitals<1)OpenMX_Solver_Failure("invalid stress VNA orbital count");
  const size_t shape[4]={(size_t)orbitals,(size_t)orbitals,1,1};
  const size_t strain_shape[4]={4,4,(size_t)orbitals,(size_t)orbitals};
  OpenMX_Stress_OverwritePrepare(owner,"stress:vna:x",shape,x);
  OpenMX_Stress_OverwritePrepare(owner,"stress:vna:y",shape,y);
  OpenMX_Stress_OverwritePrepare(owner,"stress:vna:z",shape,z);
  OpenMX_Stress_OverwritePrepare(owner,"stress:vna:strain",strain_shape,strain);
}
static inline double *OpenMX_Stress_VnaStrainRow(const OpenMXSolverRealTensor *view,
    size_t axis,size_t derivative,size_t orbital)
{
  return view->values+((axis*view->extent[1]+derivative)*view->extent[2]+orbital)*view->extent[3];
}

/* Six complex derivatives coexist for the two orientations. Serial worker
   zero lends its slots to the later joined worker region. Keep explicit
   outputs: native OpenMP must not return a large aggregate by value. */
static inline void OpenMX_Stress_HnlPrepare(OpenMXSolverInvocation *owner,int orbitals,
    OpenMXSolverComplexTensor *x0,OpenMXSolverComplexTensor *y0,OpenMXSolverComplexTensor *z0,
    OpenMXSolverComplexTensor *x1,OpenMXSolverComplexTensor *y1,OpenMXSolverComplexTensor *z1,
    OpenMXSolverComplexTensor *strain)
{
  if(orbitals<1)OpenMX_Solver_Failure("invalid stress nonlocal orbital count");
  OpenMX_Stress_KineticPrepare(owner,"stress:nonlocal:x0",orbitals,x0);
  OpenMX_Stress_KineticPrepare(owner,"stress:nonlocal:y0",orbitals,y0);
  OpenMX_Stress_KineticPrepare(owner,"stress:nonlocal:z0",orbitals,z0);
  OpenMX_Stress_KineticPrepare(owner,"stress:nonlocal:x1",orbitals,x1);
  OpenMX_Stress_KineticPrepare(owner,"stress:nonlocal:y1",orbitals,y1);
  OpenMX_Stress_KineticPrepare(owner,"stress:nonlocal:z1",orbitals,z1);
  /* Fold [spin,axis,derivative,row,column] into the rank-four C view. */
  const size_t shape[4]={12,4,(size_t)orbitals,(size_t)orbitals};char error[256];
  if(!openmx_core_solver_complex_overwrite(owner,omp_get_thread_num(),"stress:nonlocal:strain",
        shape,strain,error,sizeof(error)))OpenMX_Solver_Failure(error);
}
static inline dcomplex *OpenMX_Stress_HnlStrainRow(const OpenMXSolverComplexTensor *view,
    size_t spin,size_t axis,size_t derivative,size_t orbital)
{
  return view->values+(((spin*4+axis)*view->extent[1]+derivative)*view->extent[2]+orbital)*view->extent[3];
}
#endif
