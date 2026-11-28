#include "openmx_md_step.h"
#include "openmx_atom_layout.h"
#include "openmx_env.h"
#include "openmx_gpu_device.h"
#include "openmx_host_simd.h"
#if OPENMX_GPU_CUDA || OPENMX_GPU_HIP
#include "openmx_md_gpu.h"
#endif

#include <stdio.h>
#include <stdlib.h>

typedef struct {
  void *data;
  size_t bytes;
} MDBuffer;

struct OpenMXMD {
  MPI_Comm comm;
  OpenMXNativeResources *resources;
  OpenMXMDAtoms atoms;
  OpenMXAtomLayout layout;
  int rank, ranks, verbosity;
  int movable;
  MDBuffer components, tiles, records, npt_coordinates;
  OpenMXHostMDMove move;
  OpenMXHostMDPeriodic periodic;
  OpenMXHostMDNPT npt;
#if OPENMX_GPU_CUDA || OPENMX_GPU_HIP
  int use_gpu, gpu_failed, gpu_reported;
  OpenMXMDGPU *gpu;
#endif
};

static void md_abort(MPI_Comm comm, const char *why)
{
  fprintf(stderr,"<MD> %s\n",why);
  MPI_Abort(comm,1);
  abort();
}

static void *reserve(OpenMXMD *md, MDBuffer *buffer, size_t n, size_t width)
{
  if (n>SIZE_MAX/width) md_abort(md->comm,"workspace size overflow");
  const size_t bytes=n*width;
  if (bytes>buffer->bytes) {
    void *data=openmx_core_native_realloc(md->resources,buffer->data,bytes);
    if (!data) md_abort(md->comm,"cannot allocate workspace");
    buffer->data=data;
    buffer->bytes=bytes;
  }
  return buffer->data;
}

static void release_state(OpenMXNativeResources *owner, void *state)
{
  (void)owner;
#if OPENMX_GPU_CUDA || OPENMX_GPU_HIP
  OpenMXMD *md=(OpenMXMD *)state;
  OpenMX_MD_GPU_Destroy(md->gpu);
  md->gpu=NULL;
#else
  (void)state;
#endif
}

OpenMXMD *OpenMX_MD_Begin(OpenMXCoreContext *core, MPI_Comm comm,
    const OpenMXMDAtoms *atoms, int use_gpu, int verbosity)
{
  char error[256];
  OpenMXNativeResources *resources=NULL;
  void *state=NULL;
  if (!openmx_core_native_resources_state(core,"Molecular_Dynamics",sizeof(OpenMXMD),
                                         release_state,&resources,&state,error,sizeof(error)))
    md_abort(comm,error);
  OpenMXMD *md=(OpenMXMD *)state;
  md->resources=resources;
  md->comm=comm;
  md->atoms=*atoms;
  md->verbosity=verbosity;
  md->movable=-1;
  MPI_Comm_rank(comm,&md->rank);
  MPI_Comm_size(comm,&md->ranks);
  if (!OpenMX_Atom_Layout_Bind(&md->layout,resources,md->ranks,atoms->atoms,atoms->owners))
    md_abort(comm,"cannot allocate atom layout");

  const int mode=OpenMX_HostSimdMode();
  md->move=openmx_core_host_md_move(mode);
  md->periodic=openmx_core_host_md_periodic(mode);
  md->npt=openmx_core_host_md_npt(mode);
  if (!md->move) md->move=openmx_core_host_md_move(OPENMX_GRID_SCALAR);
  if (!md->periodic) md->periodic=openmx_core_host_md_periodic(OPENMX_GRID_SCALAR);
  if (!md->npt) md->npt=openmx_core_host_md_npt(OPENMX_GRID_SCALAR);
#if OPENMX_GPU_CUDA || OPENMX_GPU_HIP
  const long minimum=openmx_env_long_at(OPENMX_ENV_MD_GPU_MIN,32768);
  md->use_gpu=use_gpu && minimum>0 && atoms->local_atoms>=minimum
      && gpu_rank_device_usable() && !OpenMX_HostSimdPreferred("MD");
#else
  (void)use_gpu;
#endif
  return md;
}

#if OPENMX_GPU_CUDA || OPENMX_GPU_HIP
static void gpu_result(OpenMXMD *md, int ok)
{
  if (ok) {
    if (!md->gpu_reported && md->rank==0 && md->verbosity>=2)
      printf("<MD> GPU atom updates active with %s.\n",OPENMX_GPU_RUNTIME_NAME);
    md->gpu_reported=1;
    return;
  }
  OpenMX_MD_GPU_Destroy(md->gpu);
  md->gpu=NULL;
  md->gpu_failed=1;
  if (md->verbosity>=2) printf("<MD> Rank %d continuing atom updates on the CPU.\n",md->rank);
}
static void *gpu_workspace(OpenMXMD *md, size_t bytes)
{
  if (!bytes || !md->use_gpu || md->gpu_failed) return NULL;
  if (!md->gpu) md->gpu=OpenMX_MD_GPU_Create();
  void *data=md->gpu ? OpenMX_MD_GPU_Workspace(md->gpu,bytes) : NULL;
  if (!data) gpu_result(md,0);
  return data;
}
#endif

/* MD atom ownership can change after truncation. The packed list excludes
   fixed components and keeps the legacy local atom and axis order. */
static void list_components(OpenMXMD *md)
{
  const OpenMXMDAtoms a=md->atoms;
  int *indices=reserve(md,&md->components,3*(size_t)a.local_atoms,sizeof(int));
  md->movable=0;
  for (int local=1; local<=a.local_atoms; ++local) {
    const int atom=a.local_to_global[local];
    for (int axis=0; axis<3; ++axis)
      if (!a.fixed[atom][axis+1]) indices[md->movable++]=3*(atom-1)+axis;
  }
}

static void pack_move_tile(OpenMXMD *md, OpenMXMDMoveTile *tile, int first,
    OpenMXMDMove kind, double mass_scale, double scale, const double *scales, const int *groups)
{
  const int *indices=md->components.data;
  const int start=kind==OPENMX_MD_VERLET_KICK ? 27 : 24;
  for (int l=0; l<OPENMX_MD_TILE; ++l) {
    const int i=first+l;
    if (i<md->movable) {
      const int atom=indices[i]/3+1, axis=indices[i]%3;
      const double *xyz=md->atoms.xyz[atom];
      tile->x[l]=xyz[1+axis];
      tile->v[l]=xyz[start+axis];
      tile->gradient[l]=xyz[17+axis];
      tile->mass[l]=xyz[20]*mass_scale;
      if (kind==OPENMX_MD_LANGEVIN_DRIFT) {
        tile->scale[l]=xyz[21+axis];
        tile->previous_gradient[l]=xyz[14+axis];
      }
      else {
        tile->scale[l]=scales ? scales[groups ? groups[atom] : atom] : scale;
        tile->previous_gradient[l]=0.0;
      }
    }
    else {
      tile->x[l]=tile->v[l]=tile->gradient[l]=tile->previous_gradient[l]=0.0;
      tile->mass[l]=tile->scale[l]=1.0;
    }
  }
}

static void store_move_tile(OpenMXMD *md, const OpenMXMDMoveTile *tile, int first, OpenMXMDMove kind)
{
  const int *indices=md->components.data;
  const int drift=kind!=OPENMX_MD_VERLET_KICK && kind!=OPENMX_MD_SCALE_KICK;
  const int save_pos=kind==OPENMX_MD_VERLET_DRIFT || kind==OPENMX_MD_LANGEVIN_START
                     || kind==OPENMX_MD_LANGEVIN_DRIFT;
  for (int l=0; l<OPENMX_MD_TILE && first+l<md->movable; ++l) {
    const int i=indices[first+l], axis=i%3;
    double *xyz=md->atoms.xyz[i/3+1];
    xyz[24+axis]=tile->v[l];
    if (drift) {
      if (save_pos) xyz[21+axis]=xyz[1+axis];
      if (kind==OPENMX_MD_VERLET_DRIFT) xyz[27+axis]=tile->v[l];
      else xyz[14+axis]=xyz[17+axis];
      xyz[1+axis]=tile->x[l];
    }
  }
}

void OpenMX_MD_Move(OpenMXMD *md, OpenMXMDStep step, double mass_scale, double scale,
    const double *scales, const int *groups)
{
  if (md->movable<0) list_components(md);
  if (md->movable==0) return;
  const int count=(md->movable-1)/OPENMX_MD_TILE+1;
  OpenMXMDMoveTile *data=NULL;
#if OPENMX_GPU_CUDA || OPENMX_GPU_HIP
  data=gpu_workspace(md,(size_t)count*sizeof(*data));
  if (data) {
#pragma omp parallel for schedule(static) if(count>=16)
    for (int t=0; t<count; ++t)
      pack_move_tile(md,data+t,t*OPENMX_MD_TILE,step.move,mass_scale,scale,scales,groups);
    const int gpu=OpenMX_MD_GPU_Move(md->gpu,data,count,&step);
    gpu_result(md,gpu);
    if (!gpu) data=NULL;
  }
#endif
#pragma omp parallel for schedule(static) if(count>=16)
  for (int t=0; t<count; ++t) {
    OpenMXMDMoveTile tile;
    const int first=t*OPENMX_MD_TILE;
    if (!data) {
      pack_move_tile(md,&tile,first,step.move,mass_scale,scale,scales,groups);
      md->move(&tile,&step);
    }
    store_move_tile(md,data ? data+t : &tile,first,step.move);
  }
}

void OpenMX_MD_Center(OpenMXMD *md, int column)
{
  const OpenMXMDAtoms a=md->atoms;
  double sum[6]={0.0};
#pragma omp parallel for schedule(static) reduction(+:sum[:6]) if(a.local_atoms>=128)
  for (int local=1; local<=a.local_atoms; ++local) {
    const int atom=a.local_to_global[local];
    for (int d=0; d<3; ++d) if (!a.fixed[atom][d+1]) {
      sum[d]+=a.xyz[atom][column+d];
      sum[3+d]+=1.0;
    }
  }
  if (md->ranks>1) MPI_Allreduce(MPI_IN_PLACE,sum,6,MPI_DOUBLE,MPI_SUM,md->comm);
  for (int d=0; d<3; ++d) if (sum[3+d]>0.0) sum[d]/=sum[3+d];
#pragma omp parallel for schedule(static) if(a.local_atoms>=128)
  for (int local=1; local<=a.local_atoms; ++local) {
    const int atom=a.local_to_global[local];
    for (int d=0; d<3; ++d) if (!a.fixed[atom][d+1]) a.xyz[atom][column+d]-=sum[d];
  }
}

double OpenMX_MD_Kinetic(OpenMXMD *md, double mass_scale, int exclude_fixed)
{
  const OpenMXMDAtoms a=md->atoms;
  double energy=0.0;
#pragma omp parallel for schedule(static) reduction(+:energy) if(a.local_atoms>=128)
  for (int local=1; local<=a.local_atoms; ++local) {
    const int atom=a.local_to_global[local];
    const double *xyz=a.xyz[atom];
    double speed2=0.0;
    for (int d=0; d<3; ++d)
      if (!exclude_fixed || !a.fixed[atom][d+1]) speed2+=xyz[24+d]*xyz[24+d];
    energy+=0.5*xyz[20]*mass_scale*speed2;
  }
  if (md->ranks>1) MPI_Allreduce(MPI_IN_PLACE,&energy,1,MPI_DOUBLE,MPI_SUM,md->comm);
  return energy;
}

static void pack_npt_tile(const OpenMXMDAtoms *atoms, OpenMXMDNPTTile *tile,
    int first, double mass_scale)
{
  for (int l=0; l<OPENMX_MD_TILE; ++l) {
    const int local=first+l;
    if (local<=atoms->local_atoms) {
      const double *xyz=atoms->xyz[atoms->local_to_global[local]];
      for (int d=0; d<3; ++d) {
        tile->position[d][l]=xyz[1+d];
        tile->velocity[d][l]=xyz[24+d];
        tile->gradient[d][l]=xyz[17+d];
      }
      tile->mass[l]=xyz[20]*mass_scale;
    }
    else {
      for (int d=0; d<3; ++d)
        tile->position[d][l]=tile->velocity[d][l]=tile->gradient[d][l]=0.0;
      tile->mass[l]=1.0;
    }
    tile->kinetic[l]=0.0;
  }
}

OpenMXMDLocalVector *OpenMX_MD_NPT_Workspace(OpenMXMD *md)
{
  return reserve(md,&md->npt_coordinates,2*((size_t)md->atoms.local_atoms+1),
                 sizeof(OpenMXMDLocalVector));
}

double OpenMX_MD_NPT_Kick(OpenMXMD *md, const OpenMXMDNPTStep *step, double mass_scale,
    double (*coordinates)[4], double (*velocities)[4])
{
  const int atoms=md->atoms.local_atoms, tiles=(atoms+OPENMX_MD_TILE-1)/OPENMX_MD_TILE;
  OpenMXMDNPTTile *data=NULL;
#if OPENMX_GPU_CUDA || OPENMX_GPU_HIP
  data=gpu_workspace(md,(size_t)tiles*sizeof(*data));
  if (data) {
#pragma omp parallel for schedule(static) if(tiles>=4)
    for (int t=0; t<tiles; ++t) pack_npt_tile(&md->atoms,data+t,t*OPENMX_MD_TILE+1,mass_scale);
    const int gpu=OpenMX_MD_GPU_NPT(md->gpu,data,tiles,step);
    gpu_result(md,gpu);
    if (!gpu) data=NULL;
  }
#endif
  double kinetic=0.0;
#pragma omp parallel for schedule(static) reduction(+:kinetic) if(tiles>=4)
  for (int t=0; t<tiles; ++t) {
    OpenMXMDNPTTile tile;
    const int first=t*OPENMX_MD_TILE+1;
    const OpenMXMDNPTTile *out=data ? data+t : &tile;
    if (!data) {
      pack_npt_tile(&md->atoms,&tile,first,mass_scale);
      md->npt(&tile,step);
    }
    for (int l=0; l<OPENMX_MD_TILE && first+l<=atoms; ++l) {
      for (int d=0; d<3; ++d) {
        coordinates[first+l][d+1]=out->position[d][l];
        velocities[first+l][d+1]=out->velocity[d][l];
      }
      kinetic+=out->kinetic[l];
    }
  }
  if (md->ranks>1) MPI_Allreduce(MPI_IN_PLACE,&kinetic,1,MPI_DOUBLE,MPI_SUM,md->comm);
  return kinetic;
}

static void pack_positions(OpenMXMD *md, OpenMXMDPeriodicTile *tiles, int count)
{
  const int first=md->layout.offset[md->rank];
  const int end=md->layout.offset[md->rank+1];
#pragma omp parallel for schedule(static) if(count>=4)
  for (int t=0; t<count; ++t) {
    for (int l=0; l<OPENMX_MD_TILE; ++l) {
      const int i=first+t*OPENMX_MD_TILE+l;
      for (int d=0; d<3; ++d) {
        tiles[t].xyz[d][l]=i<end ? md->atoms.xyz[md->layout.order[i]][d+1] : 0.0;
        tiles[t].frac[d][l]=tiles[t].shift[d][l]=0.0;
      }
    }
  }
}

typedef struct { int first, count; } RecordField;

static const RecordField *record_fields(OpenMXMDRecords kind)
{
  static const RecordField fields[][3]={
    {{0,4},{17,13},{0,0}},
    {{0,4},{14,3},{24,3}},
    {{0,4},{14,19},{0,0}},
    {{0,40},{0,0},{0,0}},
    {{0,33},{0,0},{0,0}}
  };
  return fields[kind];
}

static void pack_record(OpenMXMD *md, const RecordField fields[3],
    const OpenMXMDHistory *history, const OpenMXMDPeriodicTile *tiles,
    int atom, int index, double *out)
{
  const OpenMXMDPeriodicTile *t=&tiles[index/OPENMX_MD_TILE];
  const int lane=index%OPENMX_MD_TILE;
  double *xyz=md->atoms.xyz[atom];
  for (int d=0; d<3; ++d) {
    xyz[1+d]=t->xyz[d][lane];
    history->fractional[atom][1+d]=t->frac[d][lane];
    out[d]=t->shift[d][lane];
    if (out[d]!=0.0 && md->verbosity>0)
      printf("The fractional coordinate of %c-axis for atom %d was translated within the range (0 to 1).\n",
             'a'+d,atom);
  }
  out+=3;
  for (int f=0; f<3; ++f) {
    memcpy(out,xyz+fields[f].first,(size_t)fields[f].count*sizeof(double));
    out+=fields[f].count;
  }
}

static void translate_frame(double *coords, const OpenMXMDCell *cell, int axis, double shift)
{
  const double x=coords[0]-cell->origin[0];
  const double y=coords[1]-cell->origin[1];
  const double z=coords[2]-cell->origin[2];
  double frac[3];
  for (int d=0; d<3; ++d)
    frac[d]=(x*cell->reciprocal[d][0]+y*cell->reciprocal[d][1]+z*cell->reciprocal[d][2])*0.5/cell->pi;
  frac[axis]+=shift;
  for (int d=0; d<3; ++d)
    coords[d]=frac[0]*cell->real[0][d]+frac[1]*cell->real[1][d]+frac[2]*cell->real[2][d]+cell->origin[d];
}

static int unpack_record(OpenMXMD *md, const RecordField fields[3], const OpenMXMDCell *cell,
    const OpenMXMDHistory *history, int atom, const double *in)
{
  double *xyz=md->atoms.xyz[atom];
  const double *data=in+3;
  if (md->atoms.owners[atom]!=md->rank) {
    for (int f=0; f<3; ++f) {
      memcpy(xyz+fields[f].first,data,(size_t)fields[f].count*sizeof(double));
      data+=fields[f].count;
    }
  }
  int changed=0;
  /* Use the same sequence of translations when an atom crosses several
     faces on one step. Every rank has the same history before wrapping. */
  for (int axis=0; axis<3; ++axis) if (in[axis]!=0.0) {
    changed=1;
    for (int f=0; f<history->charge_frames; ++f)
      translate_frame(history->charge_history[f]+3*(size_t)(atom-1),cell,axis,in[axis]);
    for (int f=0; f<history->geometry_frames; ++f)
      translate_frame(history->geometry_history[f][atom]+1,cell,axis,in[axis]);
  }
  return changed;
}

int OpenMX_MD_Finish(OpenMXMD *md, OpenMXMDRecords kind, const OpenMXMDCell *cell,
    const OpenMXMDHistory *history)
{
  const int first=md->layout.offset[md->rank], end=md->layout.offset[md->rank+1];
  const int count=(end-first+OPENMX_MD_TILE-1)/OPENMX_MD_TILE;
  OpenMXMDPeriodicTile *tiles=NULL;
#if OPENMX_GPU_CUDA || OPENMX_GPU_HIP
  tiles=gpu_workspace(md,(size_t)count*sizeof(*tiles));
  if (tiles) {
    pack_positions(md,tiles,count);
    const int gpu=OpenMX_MD_GPU_Periodic(md->gpu,tiles,count,cell);
    gpu_result(md,gpu);
    if (!gpu) tiles=NULL;
  }
#endif
  if (!tiles) {
    tiles=reserve(md,&md->tiles,count,sizeof(*tiles));
    pack_positions(md,tiles,count);
#pragma omp parallel for schedule(static) if(count>=4)
    for (int t=0; t<count; ++t) md->periodic(tiles+t,cell);
  }

  const RecordField *fields=record_fields(kind);
  const int width=3+fields[0].count+fields[1].count+fields[2].count;
  const int atoms=md->atoms.atoms;
  int changed=0;
  if (!OpenMX_Atom_Layout_Records(&md->layout,md->ranks,width)) {
    double record[43];
    for (int i=0; i<atoms; ++i) {
      const int atom=md->layout.order[i], owner=md->atoms.owners[atom];
      if (md->rank==owner) pack_record(md,fields,history,tiles,atom,i-first,record);
      MPI_Bcast(record,width,MPI_DOUBLE,owner,md->comm);
      changed|=unpack_record(md,fields,cell,history,atom,record);
    }
    return changed;
  }

  double *records=reserve(md,&md->records,atoms,(size_t)width*sizeof(double));
#pragma omp parallel for schedule(static) if(end-first>=128)
  for (int i=first; i<end; ++i)
    pack_record(md,fields,history,tiles,md->layout.order[i],i-first,records+(size_t)i*width);
  if (md->ranks>1)
    MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DOUBLE,records,md->layout.counts,md->layout.displs,MPI_DOUBLE,md->comm);
#pragma omp parallel for schedule(static) reduction(|:changed) if(atoms>=128)
  for (int i=0; i<atoms; ++i)
    changed|=unpack_record(md,fields,cell,history,md->layout.order[i],records+(size_t)i*width);
  return changed;
}
