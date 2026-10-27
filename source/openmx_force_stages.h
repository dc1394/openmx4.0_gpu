#ifndef OPENMX_FORCE_STAGES_H
#define OPENMX_FORCE_STAGES_H
/* Included after the native stage declarations. A nested C++ invocation owns
   each stage's host scratch. In particular, Force3 grid derivatives cease to
   occupy host memory before VNA/nonlocal GPU archives are prepared. */
typedef enum {
    OPENMX_FORCE_GRID, OPENMX_FORCE_VNA, OPENMX_FORCE_NONLOCAL, OPENMX_FORCE_CORE_HOLE
} OpenMXForceStageKind;
typedef struct {
    OpenMXForceStageKind kind;
    OpenMXHamiltonianGpuState *hamiltonian_state;
    const OpenMXAtomicViews *atomic;
    const OpenMXVnaMatrixViews *vna;
    const OpenMXReciprocalViews *reciprocal;
    double *****density,*****imaginary;
    const OpenMXElectronicViews *electronic;
    const OpenMXLocalBasisViews *local_basis;
} OpenMXForceStageArguments;
static int OpenMX_ForceStageOperation(OpenMXSolverInvocation *scratch,void *opaque,double *result)
{
    const OpenMXForceStageArguments *a=opaque;
    switch(a->kind) {
    case OPENMX_FORCE_GRID:
        Force3(a->hamiltonian_state,a->atomic,a->electronic,a->local_basis,scratch);break;
    case OPENMX_FORCE_VNA:
        Force4B(a->vna,a->reciprocal,a->density,a->local_basis,scratch);break;
    case OPENMX_FORCE_NONLOCAL:
        Force_HNL(a->atomic,a->density,a->imaginary,a->electronic,a->local_basis,scratch);break;
    case OPENMX_FORCE_CORE_HOLE:
        Force_CoreHole(a->atomic,a->density,a->imaginary,a->electronic,scratch);break;
    default:
        return 0;
    }
    *result=0;return 1;
}
static inline void OpenMX_ForceGrid(OpenMXHamiltonianGpuState *hamiltonian_state, const OpenMXAtomicViews *atomic,
    const OpenMXElectronicViews *electronic,const OpenMXLocalBasisViews *local_basis)
{
    OpenMXForceStageArguments a={.kind=OPENMX_FORCE_GRID,.hamiltonian_state=hamiltonian_state,.atomic=atomic,
        .electronic=electronic,.local_basis=local_basis};
    (void)OpenMX_Solver_Run(OpenMX_ForceStageOperation,&a);
}
static inline void OpenMX_ForceVna(const OpenMXVnaMatrixViews *vna,
    const OpenMXReciprocalViews *reciprocal,double *****density,const OpenMXLocalBasisViews *local_basis)
{
    OpenMXForceStageArguments a={.kind=OPENMX_FORCE_VNA,.vna=vna,.reciprocal=reciprocal,
        .density=density,.local_basis=local_basis};
    (void)OpenMX_Solver_Run(OpenMX_ForceStageOperation,&a);
}
static inline void OpenMX_ForceNonlocal(const OpenMXAtomicViews *atomic,
    double *****density,double *****imaginary,const OpenMXElectronicViews *electronic,
    const OpenMXLocalBasisViews *local_basis)
{
    OpenMXForceStageArguments a={.kind=OPENMX_FORCE_NONLOCAL,.atomic=atomic,
        .density=density,.imaginary=imaginary,.electronic=electronic,.local_basis=local_basis};
    (void)OpenMX_Solver_Run(OpenMX_ForceStageOperation,&a);
}
static inline void OpenMX_ForceCoreHole(const OpenMXAtomicViews *atomic,double *****density,double *****imaginary,
    const OpenMXElectronicViews *electronic)
{
    OpenMXForceStageArguments a={.kind=OPENMX_FORCE_CORE_HOLE,.atomic=atomic,
        .density=density,.imaginary=imaginary,.electronic=electronic};
    (void)OpenMX_Solver_Run(OpenMX_ForceStageOperation,&a);
}
#endif
