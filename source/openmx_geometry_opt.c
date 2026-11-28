#include "openmx_geometry_opt.h"
#include "openmx_common.h"
#include "openmx_core_native_resources.h"
#include "openmx_env.h"
#include "openmx_host_simd.h"
#include "openmx_gpu_device.h"
#include "lapack_prototypes.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void F77_NAME(dgemv,DGEMV)(char *, int *, int *, double *, double *, int *,
                                 double *, int *, double *, double *, int *);

typedef struct {
  void *data;
  size_t capacity;
} GeometryBuffer;

struct OpenMXGeometryOpt {
  OpenMXNativeResources *owner;
  MPI_Comm comm;
  int rank, ranks, host, atoms;
  const int *atom_owner;
  int *order, *offset, *counts, *displs;
  GeometryBuffer indices, transfer, diis, scratch, matrix, vectors, values, work, iwork;
  OpenMXHostDot dot;
  OpenMXHostUpdateRow update_row;
};

static void geometry_abort(MPI_Comm comm, const char *message)
{
  fprintf(stderr,"<Geometry_Opt> %s\n",message);
  MPI_Abort(comm,1);
  abort();
}

static void *reserve(OpenMXGeometryOpt *g, GeometryBuffer *buffer, size_t count, size_t size)
{
  if (count>SIZE_MAX/size) geometry_abort(g->comm,"workspace size overflow");
  const size_t bytes=count*size;
  if (bytes>buffer->capacity) {
    void *data=openmx_core_native_realloc(g->owner,buffer->data,bytes);
    if (!data) geometry_abort(g->comm,"cannot allocate geometry workspace");
    buffer->data=data;
    buffer->capacity=bytes;
  }
  return buffer->data;
}

OpenMXGeometryOpt *OpenMX_Geometry_Begin(OpenMXCoreContext *core, MPI_Comm comm,
    int host, int atoms, const int *owner)
{
  OpenMXNativeResources *resources=NULL;
  void *metadata=NULL;
  char error[256];
  if (!openmx_core_native_resources_state(core,"Geometry_Opt",sizeof(OpenMXGeometryOpt),NULL,
                                         &resources,&metadata,error,sizeof(error)))
    geometry_abort(comm,error);
  OpenMXGeometryOpt *g=(OpenMXGeometryOpt *)metadata;
  g->owner=resources;
  g->comm=comm;
  g->host=host;
  g->atoms=atoms;
  g->atom_owner=owner;
  MPI_Comm_rank(comm,&g->rank);
  MPI_Comm_size(comm,&g->ranks);
  const int mode=OpenMX_HostSimdMode();
  const int avx=mode==OPENMX_GRID_AVX2 || mode==OPENMX_GRID_AVX512;
  g->dot=avx ? openmx_core_host_dot(mode) : NULL;
  g->update_row=avx ? openmx_core_host_update_row(mode) : NULL;
  if (g->ranks==1) return g;

  /* Compact global atom order by owner. This also handles empty ranks. */
  const size_t ranks=(size_t)g->ranks;
  g->offset=reserve(g,&g->indices,3*ranks+1+(size_t)atoms,sizeof(int));
  g->counts=g->offset+ranks+1;
  g->displs=g->counts+ranks;
  g->order=g->displs+ranks;
  memset(g->offset,0,(ranks+1)*sizeof(int));
  for (int a=1; a<=atoms; ++a) ++g->offset[owner[a]+1];
  for (int p=0; p<g->ranks; ++p) g->offset[p+1]+=g->offset[p];
  memcpy(g->counts,g->offset,ranks*sizeof(int));
  for (int a=1; a<=atoms; ++a) g->order[g->counts[owner[a]]++]=a;
  return g;
}

static void layout(OpenMXGeometryOpt *g, int width)
{
  for (int p=0; p<g->ranks; ++p) {
    g->displs[p]=width*g->offset[p];
    g->counts[p]=width*(g->offset[p+1]-g->offset[p]);
  }
}

void OpenMX_Geometry_Share_Atoms(OpenMXGeometryOpt *g, double **xyz)
{
  if (g->ranks==1) return;
  const int atoms=g->atoms;
  if (atoms>INT_MAX/6) {
    for (int a=1; a<=atoms; ++a) {
      MPI_Bcast(xyz[a]+1,3,MPI_DOUBLE,g->atom_owner[a],g->comm);
      MPI_Bcast(xyz[a]+17,3,MPI_DOUBLE,g->atom_owner[a],g->comm);
    }
    return;
  }
  double *data=reserve(g,&g->transfer,6*(size_t)atoms,sizeof(double));
  layout(g,6);
  const int first=g->offset[g->rank], last=g->offset[g->rank+1];
  for (int i=first; i<last; ++i) {
    const int a=g->order[i];
    memcpy(data+6*i,xyz[a]+1,3*sizeof(double));
    memcpy(data+6*i+3,xyz[a]+17,3*sizeof(double));
  }
  MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DOUBLE,data,g->counts,g->displs,MPI_DOUBLE,g->comm);
  for (int i=0; i<atoms; ++i) {
    const int a=g->order[i];
    memcpy(xyz[a]+1,data+6*i,3*sizeof(double));
    memcpy(xyz[a]+17,data+6*i+3,3*sizeof(double));
  }
}

void OpenMX_Geometry_Publish(OpenMXGeometryOpt *g, double **xyz, double (*cell)[4])
{
  if (g->ranks==1) return;
  /* RFC stores the three lattice vectors in the rows after the atoms. */
  const int positions=g->atoms+(cell ? 3 : 0);
  const size_t elements=3*(size_t)positions+(cell ? 9 : 0);
  double *data=reserve(g,&g->transfer,elements,sizeof(double));
  double *lattice=data+3*(size_t)positions;
  if (g->rank==g->host) {
    for (int a=1; a<=positions; ++a) memcpy(data+3*(size_t)(a-1),xyz[a]+1,3*sizeof(double));
    if (cell)
      for (int a=1; a<=3; ++a) memcpy(lattice+3*(a-1),cell[a]+1,3*sizeof(double));
  }
  for (size_t offset=0; offset<elements;) {
    const int count=elements-offset>INT_MAX ? INT_MAX : (int)(elements-offset);
    MPI_Bcast(data+offset,count,MPI_DOUBLE,g->host,g->comm);
    offset+=(size_t)count;
  }
  if (g->rank!=g->host) {
    for (int a=1; a<=positions; ++a) memcpy(xyz[a]+1,data+3*(size_t)(a-1),3*sizeof(double));
    if (cell)
      for (int a=1; a<=3; ++a) memcpy(cell[a]+1,lattice+3*(a-1),3*sizeof(double));
  }
}

void OpenMX_Geometry_Gather_Hessian(OpenMXGeometryOpt *g, double **hessian, int cell_columns)
{
  if (g->ranks==1 || g->atoms==0) return;
  const int n=3*g->atoms;
  const int width=n+cell_columns;
  if (n>INT_MAX/width) {
    for (int a=1; a<=g->atoms; ++a)
      for (int j=0; j<3; ++j)
        MPI_Bcast(hessian[3*(a-1)+j+1]+1,width,MPI_DOUBLE,g->atom_owner[a],g->comm);
    return;
  }
  layout(g,3*width);
  const int first=g->offset[g->rank], last=g->offset[g->rank+1];
  const int root=g->rank==g->host;
  const size_t elements=root ? (size_t)n*width : (size_t)g->counts[g->rank];
  double *data=reserve(g,&g->matrix,elements ? elements : 1,sizeof(double));
  double *owned=data+(root ? g->displs[g->rank] : 0);
#pragma omp parallel for schedule(static) if(n>=256)
  for (int i=first; i<last; ++i) {
    const int a=g->order[i];
    for (int j=0; j<3; ++j)
      memcpy(owned+(size_t)(3*(i-first)+j)*width,hessian[3*(a-1)+j+1]+1,(size_t)width*sizeof(double));
  }
  MPI_Gatherv(root ? MPI_IN_PLACE : data,g->counts[g->rank],MPI_DOUBLE,
              data,g->counts,g->displs,MPI_DOUBLE,g->host,g->comm);
  if (!root) return;
#pragma omp parallel for schedule(static) if(n>=256)
  for (int i=0; i<g->atoms; ++i) {
    const int a=g->order[i];
    for (int j=0; j<3; ++j)
      memcpy(hessian[3*(a-1)+j+1]+1,data+(size_t)(3*i+j)*width,(size_t)width*sizeof(double));
  }
}

static double scalar_dot(const double *left, const double *right, size_t n)
{
  double result=0.0;
  for (size_t j=0; j<n; ++j) result+=left[j]*right[j];
  return result;
}

static int schlegel_period(int element)
{
  const int closed_shells[]={2,10,18,36,54,86};
  if (element<=1) return element==1 ? 1 : 2;
  for (int i=0; i<6; ++i)
    if (element<=closed_shells[i]) return element==closed_shells[i] ? 0 : i+1;
  return 7;
}

void OpenMX_Geometry_Schlegel(OpenMXGeometryOpt *g, double **hessian,
    const OpenMXSchlegelModel *model, const double (*cell)[4], const double (*inverse_cell)[4])
{
  /* Schlegel and Wittbrodt parameters, indexed by period. */
  static const double offsets[8][8]={
    {0.5000,  0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000},
    {0.5000, -0.2573, 0.3401, 0.6937, 0.7126, 0.8335, 0.9491, 1.0000},
    {0.5000,  0.3401, 0.9652, 1.2843, 1.4625, 1.6549, 1.7190, 2.0000},
    {0.5000,  0.6937, 1.2843, 1.6925, 1.8238, 2.1164, 2.3185, 2.5000},
    {0.5000,  0.7126, 1.4625, 1.8238, 2.0203, 2.2137, 2.5206, 2.7000},
    {0.5000,  0.8335, 1.6549, 2.1164, 2.2137, 2.3718, 2.5110, 2.7000},
    {0.5000,  0.9491, 1.7190, 2.3185, 2.5206, 2.5110, 2.5200, 2.7000},
    {0.5000,  1.0000, 2.0000, 2.5000, 2.7000, 2.7000, 2.7000, 2.9000}
  };
  const int n=3*g->atoms, cell_columns=cell ? 9 : 0, width=n+cell_columns;
  double diagonal[3][3]={{0.1,0.0,0.0},{0.0,0.1,0.0},{0.0,0.0,0.1}};
  double lattice[81]={0.0};

  if (cell) {
    /* The Cartesian diagonal regularizer becomes 0.1 T T^T. */
    for (int i=0; i<3; ++i)
      for (int j=0; j<3; ++j)
        diagonal[i][j]=0.1*scalar_dot(cell[i+1]+1,cell[j+1]+1,3);
  }

#pragma omp parallel for schedule(static) if(model->local_atoms>=32) reduction(+:lattice[:81])
  for (int a=1; a<=model->local_atoms; ++a) {
    const int center=model->local_to_global[a], row=3*(center-1)+1;
    const double *center_xyz=model->positions[center]+1;
    const int period=schlegel_period(model->elements[model->species[center]]);
    for (int i=0; i<3; ++i) {
      memset(hessian[row+i]+1,0,(size_t)width*sizeof(double));
      memcpy(hessian[row+i]+row,diagonal[i],3*sizeof(double));
    }

    for (int bond=1; bond<=model->neighbors[center]; ++bond) {
      const int neighbor=model->neighbor_atoms[center][bond], col=3*(neighbor-1)+1;
      const int image=model->neighbor_images[center][bond];
      const double r=model->distances[center][bond];
      const int other_period=schlegel_period(model->elements[model->species[neighbor]]);
      const double d=r-offsets[period][other_period], spring=1.734/d/d/d;
      const double *neighbor_xyz=model->positions[neighbor]+1, *shift=model->translations[image]+1;
      double direction[3], displacement[3];
      for (int i=0; i<3; ++i) direction[i]=(center_xyz[i]-neighbor_xyz[i]-shift[i])/r;
      for (int i=0; i<3; ++i)
        displacement[i]=cell ? scalar_dot(cell[i+1]+1,direction,3) : direction[i];
      for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
          const double value=displacement[i]*displacement[j]*spring;
          hessian[row+i][row+j]+=value;
          hessian[row+i][col+j]-=value;
        }
      }
      if (cell) {
        /* Apply the inverse-cell Jacobian to this bond before forming its
           atom-cell and cell-cell outer products. */
        double strain[9];
        for (int i=0; i<3; ++i) {
          double projected=0.0;
          for (int j=0; j<3; ++j) projected+=inverse_cell[j+1][i+1]*direction[j];
          for (int j=0; j<3; ++j) strain[3*i+j]=r*projected*direction[j];
        }
        const double half_spring=0.5*spring;
        for (int i=0; i<3; ++i)
          for (int j=0; j<9; ++j)
            hessian[row+i][n+1+j]+=half_spring*displacement[i]*strain[j];
        for (int i=0; i<9; ++i)
          for (int j=i; j<9; ++j)
            lattice[9*i+j]+=half_spring*strain[i]*strain[j];
      }
    }
  }
  OpenMX_Geometry_Gather_Hessian(g,hessian,cell_columns);
  if (cell) {
    double total[81];
    MPI_Reduce(lattice,total,81,MPI_DOUBLE,MPI_SUM,g->host,g->comm);
    if (g->rank==g->host) {
      for (int i=0; i<9; ++i) {
        double *row=hessian[n+1+i];
        for (int j=1; j<=n; ++j) row[j]=hessian[j][n+1+i];
        for (int j=0; j<i; ++j) row[n+1+j]=total[9*j+i];
        for (int j=i; j<9; ++j) row[n+1+j]=total[9*i+j];
        row[n+1+i]+=0.05;
      }
    }
  }
}

static void residual_gram(OpenMXGeometryOpt *g, double ***residual, int sites, int count, double *gram)
{
  const int n=3*sites, ld=count+1;
  if (!g->dot || n<128) {
    for (int i=0; i<count; ++i) for (int j=0; j<=i; ++j) {
      double sum=0.0;
      for (int a=1; a<=sites; ++a)
        sum+=residual[i][a][1]*residual[j][a][1]+residual[i][a][2]*residual[j][a][2]
             +residual[i][a][3]*residual[j][a][3];
      gram[i*ld+j]=gram[j*ld+i]=sum;
    }
    return;
  }
  double *history=reserve(g,&g->scratch,(size_t)count*n,sizeof(double));
  for (int i=0; i<count; ++i)
    for (int a=1; a<=sites; ++a)
      memcpy(history+(size_t)i*n+3*(a-1),residual[i][a]+1,3*sizeof(double));
#pragma omp parallel for schedule(static,1) if((size_t)count*count*n>=65536)
  for (int i=0; i<count; ++i) for (int j=0; j<=i; ++j)
    gram[i*ld+j]=gram[j*ld+i]=g->dot(history+(size_t)i*n,history+(size_t)j*n,n);
}

void OpenMX_Geometry_Prepare_DIIS(OpenMXGeometryOpt *g, double ***residual, int sites,
    int count, double **matrix, double **weights)
{
  const int n=count+1;
  const size_t dim=(size_t)n;
  double *a=reserve(g,&g->diis,dim*(dim+1),sizeof(double));
  double *b=a+dim*dim;
  residual_gram(g,residual,sites,count,a);
  double maximum=0.0;
  for (int i=0; i<count; ++i)
    for (int j=0; j<count; ++j) maximum=fmax(maximum,fabs(a[i*n+j]));
  const double scale=maximum>0.0 ? 1.0/maximum : 1.0;
  for (int i=0; i<count; ++i) {
    for (int j=0; j<count; ++j) a[i*n+j]*=scale;
    a[i*n+count]=a[count*n+i]=1.0;
    b[i]=0.0;
  }
  a[count*n+count]=0.0;
  b[count]=1.0;
  *matrix=a;
  *weights=b;
}

int OpenMX_Geometry_Solve_DIIS(OpenMXGeometryOpt *g, int count, double *matrix, double *weights)
{
  int n=count+1, nrhs=1, lwork=1, info=0;
  char triangle='L';
  double work;
  int *pivots=reserve(g,&g->iwork,n,sizeof(int));
  /* Use the BLAS-2 solve for this small, single-RHS system. The DSYTRS2
     route in NVHPC's LAPACK can overrun its workspace at a 2x2 pivot. */
  F77_NAME(dsysv,DSYSV)(&triangle,&n,&nrhs,matrix,&n,pivots,weights,&n,&work,&lwork,&info);
  return info;
}

static void matvec(OpenMXGeometryOpt *g, double **a, const double *x, double *y, int n)
{
  const OpenMXHostDot dot=g->dot && n>=64 ? g->dot : scalar_dot;
#pragma omp parallel for schedule(static) if(n>=128)
  for (int i=0; i<n; ++i) y[i]=dot(a[i+1]+1,x,n);
}

void OpenMX_Geometry_BFGS(OpenMXGeometryOpt *g, double **a,
    double ***positions, double ***gradients, int sites, OpenMXGeometryUpdate update)
{
  const int n=3*sites;
  double *s=reserve(g,&g->scratch,2*(size_t)n,sizeof(double));
  double *y=s+n, *product=a[0]+1;
  for (int atom=1; atom<=sites; ++atom) {
    for (int k=1; k<=3; ++k) {
      const int i=3*(atom-1)+k-1;
      s[i]=positions[0][atom][k]-positions[1][atom][k];
      y[i]=gradients[0][atom][k]-gradients[1][atom][k];
    }
  }
  const int inverse=update==OPENMX_GEOMETRY_INVERSE_HESSIAN;
  matvec(g,a,inverse ? y : s,product,n);
  double sy=scalar_dot(s,y,n), curvature=scalar_dot(inverse ? y : s,product,n);
  if (update==OPENMX_GEOMETRY_EF_HESSIAN && sy<=0.0) return;
  if (update==OPENMX_GEOMETRY_RF_HESSIAN) {
    if (sy<1.0e-10) sy=1.0e-10;
    if (curvature<1.0e-10) curvature=1.0e-10;
  }
  const double c0=inverse ? (sy+curvature)/(sy*sy) : 1.0/sy;
  const double c1=inverse ? -1.0/sy : -1.0/curvature;
  const double *left=inverse ? s : y;
  const OpenMXHostUpdateRow update_row=g->update_row;
#pragma omp parallel for schedule(static) if(n>=128)
  for (int i=0; i<n; ++i) {
    const double x=c0*left[i]+(inverse ? c1*product[i] : 0.0);
    const double z=c1*(inverse ? s[i] : product[i]);
    double *row=a[i+1]+1;
    if (update_row) update_row(row,left,product,n,x,z);
    else for (int j=0; j<n; ++j) row[j]+=x*left[j]+z*product[j];
  }
}

void OpenMX_Geometry_Apply_Inverse(OpenMXGeometryOpt *g, double **a, double **xyz)
{
  const int n=3*g->atoms;
  double *x=reserve(g,&g->scratch,n,sizeof(double));
  for (int a=1; a<=g->atoms; ++a) memcpy(x+3*(a-1),xyz[a]+1,3*sizeof(double));
  matvec(g,a,x,a[0]+1,n);
}

/* The native Hessian is row-major. Copy rows into the solver's contiguous
   array; selecting the upper triangle matches the old LAPACK packing. */
static void pack_matrix(double *matrix, double **hessian, int n)
{
#pragma omp parallel for schedule(static) if(n>=256)
  for (int i=0; i<n; ++i) memcpy(matrix+(size_t)i*n,hessian[i+1]+1,(size_t)n*sizeof(double));
}

static int host_eigen(OpenMXGeometryOpt *g, double *matrix, double *values, double *vectors,
    int n, int nev, int syevr)
{
  char job='V', range=nev==n ? 'A' : 'I', triangle='L';
  int found=0, first=1, info=0;
  double lower=0.0, upper=0.0, tolerance=LAPACK_ABSTOL;
  if (syevr) {
    int work_len=-1, iwork_len=-1, iquery=0;
    double query=0.0;
    int *support=reserve(g,&g->scratch,2*(size_t)nev,sizeof(int));
    F77_NAME(dsyevr,DSYEVR)(&job,&range,&triangle,&n,matrix,&n,&lower,&upper,
        &first,&nev,&tolerance,&found,values,vectors,&n,support,&query,&work_len,&iquery,&iwork_len,&info);
    if (info || !(query>=1.0 && query<=INT_MAX) || iquery<1) return 0;
    work_len=(int)query;
    iwork_len=iquery;
    double *work=reserve(g,&g->work,work_len,sizeof(double));
    int *iwork=reserve(g,&g->iwork,iwork_len,sizeof(int));
    F77_NAME(dsyevr,DSYEVR)(&job,&range,&triangle,&n,matrix,&n,&lower,&upper,
        &first,&nev,&tolerance,&found,values,vectors,&n,support,work,&work_len,iwork,&iwork_len,&info);
  }
  else {
    if (n>INT_MAX/8) return 0;
    int work_len=8*n;
    double *work=reserve(g,&g->work,work_len,sizeof(double));
    int *iwork=reserve(g,&g->iwork,6*(size_t)n,sizeof(int));
    F77_NAME(dsyevx,DSYEVX)(&job,&range,&triangle,&n,matrix,&n,&lower,&upper,
        &first,&nev,&tolerance,&found,values,vectors,&n,work,&work_len,iwork,iwork+5*(size_t)n,&info);
  }
  return info==0 && found==nev;
}

static const double *eigensystem(OpenMXGeometryOpt *g, double **hessian, int n, int nev,
    int use_gpu, int stdout_level, const double **eigenvalues)
{
  double *matrix=reserve(g,&g->matrix,(size_t)n*n,sizeof(double));
  double *values=reserve(g,&g->values,n,sizeof(double));
  *eigenvalues=values;
  pack_matrix(matrix,hessian,n);
#if OPENMX_GPU_CUDA || OPENMX_GPU_HIP
  const long requested_min=openmx_env_long_at(OPENMX_ENV_GEOMETRY_GPU_MIN,GPU_CPU_SWITCH_NUM);
  const long gpu_min=requested_min>0 ? requested_min : GPU_CPU_SWITCH_NUM;
  if (use_gpu && n>=gpu_min && gpu_rank_device_usable() && !OpenMX_HostSimdPreferred("Geometry_Opt")) {
    if (gpusolver_Syevdx(matrix,values,n,nev)==0) {
      if (stdout_level>=2) printf("<Geometry_Opt> GPU Hessian eigensolve (%d rows, %d vectors)\n",n,nev);
      return matrix;
    }
    pack_matrix(matrix,hessian,n);
    if (stdout_level>=1) printf("<Geometry_Opt> Retrying the Hessian eigensolve on the CPU.\n");
  }
#else
  (void)use_gpu;
  (void)stdout_level;
#endif
  double *vectors=reserve(g,&g->vectors,(size_t)n*nev,sizeof(double));
  if (n>=192) {
    if (host_eigen(g,matrix,values,vectors,n,nev,1)) return vectors;
    pack_matrix(matrix,hessian,n);
  }
  if (!host_eigen(g,matrix,values,vectors,n,nev,0))
    geometry_abort(g->comm,"Hessian eigensolve failed");
  return vectors;
}

void OpenMX_Geometry_RF_Step(OpenMXGeometryOpt *g, double **hessian, int n,
    int use_gpu, int stdout_level)
{
  const double *values;
  const double *vector=eigensystem(g,hessian,n,1,use_gpu,stdout_level,&values);
  for (int i=0; i<n; ++i) hessian[0][i+1]=vector[i]/vector[n-1];
}

void OpenMX_Geometry_EF_Step(OpenMXGeometryOpt *g, double **hessian, double min_eigenvalue,
    int use_gpu, int stdout_level)
{
  int n=3*g->atoms;
  const double *values;
  const double *vectors=eigensystem(g,hessian,n,n,use_gpu,stdout_level,&values);
  double *gradient=reserve(g,&g->scratch,2*(size_t)n,sizeof(double));
  double *projected=gradient+n;
  for (int i=0; i<n; ++i) gradient[i]=hessian[i+1][0];
  const OpenMXHostDot dot=g->dot && n>=64 ? g->dot : scalar_dot;
#pragma omp parallel for schedule(static) if(n>=128)
  for (int j=0; j<n; ++j)
    projected[j]=dot(vectors+(size_t)j*n,gradient,n)/fmax(values[j],min_eigenvalue);

  /* Back-transform into Cartesian coordinates with the configured BLAS. */
  int inc=1;
  char normal='N';
  double one=1.0, zero=0.0;
  F77_NAME(dgemv,DGEMV)(&normal,&n,&n,&one,(double *)vectors,&n,
                       projected,&inc,&zero,hessian[0]+1,&inc);
}
