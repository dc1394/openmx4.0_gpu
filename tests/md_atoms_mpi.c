#include "openmx_core_host_simd.h"
#include "openmx_md_step.h"

#include <assert.h>
#include <math.h>

static int simd_mode;

int OpenMX_HostSimdMode(void) { return simd_mode; }

static double field(int atom, int column)
{
  return (16*atom+column)*0.00390625;
}

static double position(int atom, int axis)
{
  static const double fractional[]={-2.25, 2.5, 0.25, 1.0};
  return fractional[(atom+axis)%4];
}

static int exchanged(OpenMXMDRecords kind, int column)
{
  if (column<4) return 1;
  switch (kind) {
    case OPENMX_MD_RECORD_VERLET: return column>=17 && column<30;
    case OPENMX_MD_RECORD_SCALING: return (column>=14 && column<17) || (column>=24 && column<27);
    case OPENMX_MD_RECORD_NOSE: return column>=14 && column<33;
    case OPENMX_MD_RECORD_LANGEVIN: return 1;
    case OPENMX_MD_RECORD_NPT: return column<33;
  }
  return 0;
}

static void check(OpenMXCoreContext *core, int rank, int peers,
                  OpenMXMDRecords kind, int phase)
{
  enum { atoms=145, width=40 };
  double xyz[atoms+1][width], fractions[atoms+1][4], charge[3*atoms];
  double frames[atoms+1][4];
  double *xyz_rows[atoms+1], *frac_rows[atoms+1], *frame_rows[atoms+1];
  int owners[atoms+1], local[atoms+1], fixed[atoms+1][4], *fixed_rows[atoms+1];
  int owned=0;
  const double lengths[]={8,16,32}, origin[]={0.5,-1.0,1.25};
  const OpenMXMDCell cell={
    {{8,0,0},{0,16,0},{0,0,32}}, {{0.25,0,0},{0,0.125,0},{0,0,0.0625}},
    {0.5,-1.0,1.25}, 1.0
  };
  for (int g=atoms; g>=1; --g) {
    /* Refresh ownership with one empty rank, in reversed local ordering. */
    const int group=g%19==0 ? 1 : g%23==0 ? 2 : 0;
    owners[g]=(group+phase)%peers;
    const int owner=owners[g]==rank;
    if (owner) local[++owned]=g;
    xyz_rows[g]=xyz[g];
    frac_rows[g]=fractions[g];
    frame_rows[g]=frames[g];
    fixed_rows[g]=fixed[g];
    for (int c=0; c<width; ++c) xyz[g][c]=owner ? field(g,c) : -99.0;
    for (int d=0; d<3; ++d) {
      const double x=position(g,d)*lengths[d]+origin[d];
      if (owner) xyz[g][d+1]=x;
      fixed[g][d+1]=(g+d)%7==0;
      frames[g][d+1]=x+0.0625;
      charge[3*(g-1)+d]=x-0.125;
    }
  }
  const OpenMXMDAtoms input={atoms,owned,owners,local,xyz_rows,fixed_rows};
  double *charge_frames[]={charge};
  double **geometry_frames[]={frame_rows};
  const OpenMXMDHistory history={frac_rows,charge_frames,geometry_frames,1,1};
  OpenMXMD *md=OpenMX_MD_Begin(core,MPI_COMM_WORLD,&input,0,0);
  if (kind==OPENMX_MD_RECORD_VERLET) {
    const OpenMXMDStep kick={OPENMX_MD_VERLET_KICK,0.25,0.0,0.0};
    OpenMX_MD_Move(md,kick,2.0,1.0,NULL,NULL);
  }
  assert(OpenMX_MD_Finish(md,kind,&cell,&history));

  for (int g=1; g<=atoms; ++g) {
    for (int d=0; d<3; ++d) {
      const double f=position(g,d);
      const double shift=f<0.0 ? 3.0 : f>1.0 ? -2.0 : 0.0;
      const double x=(f+shift)*lengths[d]+origin[d];
      assert(xyz[g][d+1]==x);
      assert(frames[g][d+1]==x+0.0625);
      assert(charge[3*(g-1)+d]==x-0.125);
      if (owners[g]==rank) assert(fractions[g][d+1]==f+shift);
    }
    for (int c=0; c<width; ++c) {
      if (1<=c && c<=3) continue;
      double expected=owners[g]==rank || exchanged(kind,c) ? field(g,c) : -99.0;
      if (kind==OPENMX_MD_RECORD_VERLET && c>=24 && c<27 && !fixed[g][c-23]) {
        const int d=c-24;
        expected=field(g,27+d)-(field(g,17+d)/(2.0*field(g,20)))*0.125;
      }
      assert(fabs(xyz[g][c]-expected)<1.0e-14);
    }
  }
}

int main(int argc, char **argv)
{
  int rank, peers, provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&peers);
  char error[256];
  OpenMXCoreContext *core=openmx_core_create_host(error,sizeof(error));
  assert(core);
  const char *modes[]={"scalar","avx2","avx512"};
  for (int i=0; i<3; ++i) {
    simd_mode=openmx_core_host_simd_mode(modes[i]);
    for (int kind=OPENMX_MD_RECORD_VERLET; kind<=OPENMX_MD_RECORD_NPT; ++kind)
      check(core,rank,peers,(OpenMXMDRecords)kind,i+kind);
  }
  openmx_core_destroy(core);
  MPI_Finalize();
  return 0;
}
