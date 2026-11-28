/* Ordered DOS and Fermi-surface output from the packed post-SCF band solver. */
#include "openmx_common.h"
#include "openmx_band_dos_output.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct { int first, last; } DosBands;

static int dos_all_ok(MPI_Comm comm, int local)
{
    int ready;
    MPI_Allreduce(&local,&ready,1,MPI_INT,MPI_MIN,comm);
    return ready;
}

static int solve(const OpenMXBandDosExecution *e, const OpenMXKPathMatrices *m, int count,
                 const OpenMXKPathPoint *points, const OpenMXBandDosProjection *projection,
                 double *eigen, float *weights, size_t budget, int *threads)
{
    if (!count) return 1;
#if OPENMX_GPU_CUDA || OPENMX_GPU_HIP
    if (e->device>=0) return openmx_band_dos_gpu(e->device,m,count,points,projection,eigen,weights);
#endif
    return openmx_band_dos_cpu(m,count,points,projection,eigen,weights,e->simd,budget,threads);
}

/* Keep the DOS energy window referenced to the unshifted Gamma point,
   with the same buffer in band indices as the distributed solver. */
static DosBands dos_window(const double *gamma, int spins, int order, int paos, double lo, double hi)
{
    DosBands bands = {order-1,0};
    for (int spin=0; spin<spins; ++spin) {
        const double *row = gamma+(size_t)spin*order;
        int first = 0, last = order-1;
        for (int b=0; b<order; ++b) {
            if (row[b]>lo) { first = b>0 ? b-1 : 0; break; }
        }
        for (int b=first; b<order; ++b) {
            if (row[b]>hi) { last = b; break; }
        }
        if (first<bands.first) bands.first = first;
        if (last>bands.last) bands.last = last;
    }
    const int padding = paos/(paos==order ? 20 : 10);
    const int buffer = padding>10 ? padding : 10;
    bands.first = bands.first>buffer ? bands.first-buffer : 0;
    bands.last = bands.last<order-1-buffer ? bands.last+buffer : order-1;
    return bands;
}

static int band_boundary(const double *values, int count, double chemical, int include_equal)
{
    int first=0, last=count;
    while (first<last) {
        const int middle=first+(last-first)/2;
        if (values[middle]<chemical || (include_equal && values[middle]==chemical)) first=middle+1;
        else last=middle;
    }
    return first;
}

/* Include the bounding valence and conduction bands in an insulator. */
static DosBands fermi_window(const double *eigen, int points, int spin, int spins, int order, double chemical)
{
    int first=order-1, last=0;
    for (int k=0; k<points; ++k) {
        const double *row=eigen+((size_t)k*spins+spin)*order;
        const int above=band_boundary(row,order,chemical,0);
        const int below=band_boundary(row,order,chemical,1)-1;
        if (above<first) first=above;
        if (below>last) last=below;
    }
    return first<=last ? (DosBands){first,last} : (DosBands){last,first};
}

static FILE *open_output(const OpenMXBandDosExecution *e, const char *suffix)
{
    char path[2*YOUSO10+64];
    snprintf(path,sizeof(path),"%s%s%s",e->directory,e->name,suffix);
    FILE *fp=fopen(path,"w");
    if (!fp) perror(path);
    return fp;
}

static int close_output(FILE *fp)
{
    const int failed=ferror(fp);
    return fclose(fp)==0 && !failed;
}

static double bin_energy(const OpenMXBandDosRequest *r, double chemical, int bin)
{
    const double lo=r->energy_range[0]+chemical, hi=r->energy_range[1]+chemical;
    return lo+(hi-lo)/r->gaussian_bins*bin;
}

static int write_dos_values(const OpenMXBandDosRequest *r, const OpenMXBandDosExecution *e,
    const OpenMXKPathMatrices *m, int order, int points, DosBands bands, const double *eigen)
{
    FILE *fp=open_output(e,".Dos.val");
    if (!fp) return 0;
    const int gaussian=r->gaussian_bins>0;
    const int spin_switch=m->noncollinear ? 1 : m->spin_channels-1;
    fprintf(fp,"mode        %d\nNonCol      %d\nN           %d\nNspin       %d\nErange      %lf %lf\n",
        gaussian ? 7 : 1,m->noncollinear,m->order,spin_switch,r->energy_range[0],r->energy_range[1]);
    if (!gaussian && !m->noncollinear) fprintf(fp,"irange      %d %d\n",bands.first+1,bands.last+1);
    fprintf(fp,"Kgrid       %d %d %d\n",gaussian ? 1 : r->grid[0],gaussian ? 1 : r->grid[1],gaussian ? 1 : r->grid[2]);
    fprintf(fp,"atomnum     %d\n<WhatSpecies\n",r->atom_count);
    for (int a=1; a<=r->atom_count; ++a) fprintf(fp,"%d ",r->species[a]);
    fprintf(fp,"\nWhatSpecies>\nSpeciesNum     %d\n<Spe_Total_CNO\n",r->species_count);
    for (int s=0; s<r->species_count; ++s) fprintf(fp,"%d ",r->orbitals[s]);
    fprintf(fp,"\nSpe_Total_CNO>\nMaxL           %d\n<Spe_Num_CBasis\n",r->max_l);
    for (int s=0; s<r->species_count; ++s) {
        for (int l=0; l<=r->max_l; ++l) fprintf(fp,"%d ",r->species_basis[s][l]);
        fputc('\n',fp);
    }
    fprintf(fp,"Spe_Num_CBasis>\nChemP       %lf\n",e->chemical_potential);
    if (m->noncollinear) {
        fputs("<SpinAngle\n",fp);
        for (int a=1; a<=r->atom_count; ++a) fprintf(fp,"%lf %lf\n",r->theta[a],r->phi[a]);
        fputs("SpinAngle>\n",fp);
    }
    if (gaussian) fprintf(fp,"irange      0 %d\n",r->gaussian_bins-1);
    else if (m->noncollinear) fprintf(fp,"irange      %d %d\n",bands.first+1,bands.last+1);
    fputs("<Eigenvalues\n",fp);
    if (gaussian) {
        for (int s=0; s<=spin_switch; ++s) {
            fputs("0 0 0 ",fp);
            for (int bin=0; bin<r->gaussian_bins; ++bin) fprintf(fp,"%lf ",bin_energy(r,e->chemical_potential,bin));
            fputc('\n',fp);
        }
    } else {
        for (int k=0; k<points; ++k) {
            int index[3];
            openmx_band_dos_grid_index(r->grid,k,index);
            /* DosMain reads a spectrum per output spin component. The
               two noncollinear projections share the same eigenvalues. */
            for (int spin=0; spin<=spin_switch; ++spin) {
                fprintf(fp,"%d %d %d ",index[0],index[1],index[2]);
                const double *row=eigen+((size_t)k*m->spin_channels+(m->noncollinear ? 0 : spin))*order;
                for (int b=bands.first; b<=bands.last; ++b) fprintf(fp,"%lf ",row[b]);
                fputc('\n',fp);
            }
        }
    }
    fputs("Eigenvalues>\n",fp);
    return close_output(fp);
}

static int write_bxsf(const OpenMXBandDosRequest *r, const OpenMXBandDosExecution *e,
    const OpenMXKPathMatrices *m, int spin, int order, DosBands bands, const double *eigen)
{
    char suffix[48];
    if (m->noncollinear) snprintf(suffix,sizeof(suffix),".FermiSurf.bxsf");
    else snprintf(suffix,sizeof(suffix),".FermiSurf%d.bxsf",spin);
    FILE *fp=open_output(e,suffix);
    if (!fp) return 0;
    const int *g=r->grid;
    fprintf(fp,"BEGIN_INFO\nFermi Energy: %15.12f\nEND_INFO\n\n",e->chemical_potential*eV2Hartree);
    fputs("BEGIN_BLOCK_BANDGRID_3D\ncomment_line\nBEGIN_BANDGRID_3D_simple_example\n",fp);
    fprintf(fp,"%d\n%d %d %d\n0.0 0.0 0.0\n",bands.last-bands.first+1,g[0]+1,g[1]+1,g[2]+1);
    for (int axis=1; axis<=3; ++axis) {
        const double *v=e->reciprocal[axis];
        fprintf(fp,"%15.10f %15.10f %15.10f\n",v[1]/BohrR,v[2]/BohrR,v[3]/BohrR);
    }
    for (int b=bands.first; b<=bands.last; ++b) {
        fprintf(fp,"BAND:  %d\n",b+1);
        for (int i=0; i<=g[0]; ++i) {
            for (int j=0; j<=g[1]; ++j) {
                const size_t base=((size_t)(i%g[0])*g[1]+(j%g[1]))*g[2];
                for (int k=0; k<=g[2]; ++k) {
                    const size_t point=base+(k==g[2] ? 0 : k);
                    fprintf(fp,"%10.5f ",eigen[(point*m->spin_channels+spin)*order+b]*eV2Hartree);
                }
                fputc('\n',fp);
            }
            if (i!=g[0]) fputc('\n',fp);
        }
    }
    fputs("END_BANDGRID_3D\nEND_BLOCK_BANDGRID_3D\n",fp);
    return close_output(fp);
}

static int write_frmsf(const OpenMXBandDosRequest *r, const OpenMXBandDosExecution *e,
    const OpenMXKPathMatrices *m, int spin, int atom, int order, int points, DosBands bands,
    const double *eigen, const double *atomic)
{
    char suffix[64];
    if (m->noncollinear) snprintf(suffix,sizeof(suffix),".FermiSurf_a%d.frmsf",atom+1);
    else snprintf(suffix,sizeof(suffix),".FermiSurf_s%d_a%d.frmsf",spin,atom+1);
    FILE *fp=open_output(e,suffix);
    if (!fp) return 0;
    fprintf(fp,"%d %d %d\n1\n%d\n",r->grid[0],r->grid[1],r->grid[2],bands.last-bands.first+1);
    for (int axis=1; axis<=3; ++axis) {
        const double *v=e->reciprocal[axis];
        fprintf(fp,"%e %e %e\n",v[1],v[2],v[3]);
    }
    for (int b=bands.first; b<=bands.last; ++b)
        for (int k=0; k<points; ++k)
            fprintf(fp,"%e\n",eigen[((size_t)k*m->spin_channels+spin)*order+b]-e->chemical_potential);
    for (int b=bands.first; b<=bands.last; ++b) {
        const double *weights=atomic+(((size_t)atom*m->spin_channels+spin)*order+b)*points;
        for (int k=0; k<points; ++k) fprintf(fp,"%e\n",weights[k]);
    }
    return close_output(fp);
}

static int write_surfaces(const OpenMXBandDosRequest *r, const OpenMXBandDosExecution *e,
    const OpenMXKPathMatrices *m, int order, int points, const double *eigen, const double *atomic)
{
    DosBands windows[2];
    for (int s=0; s<m->spin_channels; ++s) {
        windows[s]=r->gaussian_bins>0 ? (DosBands){0,order-1}
            : fermi_window(eigen,points,s,m->spin_channels,order,e->chemical_potential);
    }
    int success=1;
    const int files=atomic ? r->atom_count+1 : 1;
#pragma omp parallel for collapse(2) schedule(dynamic,1) reduction(&:success) if (files>1 || m->spin_channels>1)
    for (int s=0; s<m->spin_channels; ++s) {
        for (int file=0; file<files; ++file) {
            success &= file==0 ? write_bxsf(r,e,m,s,order,windows[s],eigen)
                : write_frmsf(r,e,m,s,file-1,order,points,windows[s],eigen,atomic);
        }
    }
    return success;
}

static int write_vector(FILE *fp, const int index[3], const float *weights, int order)
{
    return fwrite(index,sizeof(*index),3,fp)==3 && fwrite(weights,sizeof(*weights),order,fp)==(size_t)order;
}

static int write_tetra_vectors(FILE *fp, const int grid[3], int first, int count, int order, int spins,
    DosBands bands, const OpenMXBandDosProjection *p, const float *weights)
{
    for (int k=0; k<count; ++k) {
        int index[3];
        openmx_band_dos_grid_index(grid,first+k,index);
        for (int s=0; s<spins; ++s) {
            const float *row=weights+((size_t)k*spins+s)*p->count*order;
            for (int b=bands.first; b<=bands.last; ++b)
                if (!write_vector(fp,index,row+(size_t)(b-p->first)*order,order)) return 0;
        }
    }
    return 1;
}

static void atomic_projections(const OpenMXBandDosRequest *r, const int *orbital,
    int noncollinear, int order, int spins, int points, int first, int count, const float *weights, double *atomic)
{
    const int components=noncollinear ? 2 : 1;
#pragma omp parallel for collapse(2) schedule(static) if (count>8 || r->atom_count>4)
    for (int a=0; a<r->atom_count; ++a) {
        for (int s=0; s<spins; ++s) {
            for (int b=0; b<order; ++b) {
                double *result=atomic+(((size_t)a*spins+s)*order+b)*points+first;
                for (int k=0; k<count; ++k) {
                    const float *row=weights+(((size_t)k*spins+s)*order+b)*order;
                    double sum=0.0;
#pragma omp simd reduction(+:sum)
                    for (int j=components*orbital[a+1]; j<components*orbital[a+2]; ++j) sum+=row[j];
                    result[k]=sum;
                }
            }
        }
    }
}

static void accumulate_gaussian(const OpenMXBandDosRequest *r, double chemical, int points,
    int count, int order, int spins, const double *eigen, const float *weights, float *dos)
{
    if (!count) return;
    const int bins=r->gaussian_bins;
    const double lo=r->energy_range[0]+chemical, hi=r->energy_range[1]+chemical;
    const double width=r->gaussian_width;
    const double spread=floor(width*3.0/(hi-lo)*(bins-1)+3.0);
    const double normalization=1.0/(points*width*sqrt(PI));
#pragma omp parallel for collapse(2) schedule(static) if (bins>8)
    for (int s=0; s<spins; ++s) {
        for (int b=0; b<bins; ++b) {
            float *result=dos+((size_t)s*bins+b)*order;
            const double energy=bin_energy(r,chemical,b);
            for (int k=0; k<count; ++k) {
                const size_t row=((size_t)k*spins+s)*order;
                for (int band=0; band<order; ++band) {
                    const double eg=eigen[row+band];
                    const double center=trunc((eg-lo)/(hi-lo)*(bins-1));
                    if (center+spread<b) continue;
                    if (center-spread>b) break;
                    const double x=(eg-energy)/width;
                    const double factor=normalization*exp(-x*x);
                    const float *weight=weights+(row+band)*order;
#pragma omp simd
                    for (int j=0; j<order; ++j) result[j]+=(float)(factor*weight[j]);
                }
            }
        }
    }
}

/* Distributions and tiles are contiguous in the flattened mesh. Only the
   ranks that have an intersection send data in each collective. */
static void tile_counts(int ranks, const int *partition, int first, int end, int width, int *counts, int *offsets)
{
    for (int rank=0; rank<ranks; ++rank) {
        const int a=partition[rank]>first ? partition[rank] : first;
        const int b=partition[rank+1]<end ? partition[rank+1] : end;
        counts[rank]=b>a ? (b-a)*width : 0;
        offsets[rank]=b>a ? (a-first)*width : 0;
    }
}

int openmx_band_dos_mesh(const OpenMXBandDosRequest *r, const OpenMXBandDosExecution *e,
    const OpenMXKPathMatrices *m, const int *orbital, const OpenMXKPathPoint *points,
    int point_count, int *threads_used)
{
    const MPI_Comm comm=e->comm;
    int rank, ranks;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&ranks);
    const int host=rank==e->host, gaussian=r->gaussian_bins>0;
    const int spins=m->spin_channels, order=m->noncollinear ? 2*m->order : m->order;
    const int bands_per_point=spins*order;
    const int local_points=e->end-e->first;
    const int surf=r->fermi_surfer && !gaussian;
    int success=0;
    double *gamma=NULL, *eigen=NULL, *local_eigen=NULL, *atomic=NULL;
    float *weights=NULL, *local_weights=NULL, *dos=NULL;
    int *layout=NULL;
    OpenMXDosSpinProjection *rotation=NULL;
    FILE *vec=NULL;
    char vec_path[2*YOUSO10+64], temporary[2*YOUSO10+64]={0};
    OpenMXBandDosProjection projection={0,order,!gaussian && !m->noncollinear,NULL};
    DosBands bands={0,order-1};

    if (!gaussian) {
        gamma=malloc((size_t)bands_per_point*sizeof(*gamma));
        if (!dos_all_ok(comm,gamma!=NULL)) goto done;
        const OpenMXKPathPoint point={{0.0,0.0,0.0},0};
        const int ready=rank!=e->gamma_owner || solve(e,m,1,&point,&projection,gamma,NULL,e->workspace_limit,threads_used);
        if (!dos_all_ok(comm,ready)) goto done;
        MPI_Bcast(gamma,bands_per_point,MPI_DOUBLE,e->gamma_owner,comm);
        bands=dos_window(gamma,spins,order,m->order,r->energy_range[0]+e->chemical_potential,
                         r->energy_range[1]+e->chemical_potential);
        if (!surf) {
            projection.first=bands.first;
            projection.count=bands.last-bands.first+1;
        }
        free(gamma);
        gamma=NULL;
    }

    const size_t weight_width=(size_t)bands_per_point*projection.count;
    const size_t eigen_items=(size_t)point_count*bands_per_point;
    const size_t bins_items=gaussian ? (size_t)r->gaussian_bins*bands_per_point : 0;
    if (weight_width>INT_MAX || bins_items>INT_MAX ||
        (surf && eigen_items>SIZE_MAX/sizeof(*atomic)/(size_t)r->atom_count)) goto done;
    const size_t rotation_bytes=m->noncollinear && local_points ? (size_t)m->order*sizeof(*rotation) : 0;
    const size_t atom_items=host && surf ? eigen_items*r->atom_count : 0;
    const size_t host_eigen_items=host ? eigen_items : 0;
    if (!dos_all_ok(comm,atom_items<=SIZE_MAX/sizeof(*atomic)-host_eigen_items)) goto done;
    const size_t metadata_bytes=(host_eigen_items+atom_items)*sizeof(*eigen);
    const size_t aux_bytes=rotation_bytes+bins_items*sizeof(*dos)+((size_t)4*ranks+1)*sizeof(*layout);
    if (!dos_all_ok(comm,aux_bytes<=e->workspace_limit && metadata_bytes<=e->workspace_limit-aux_bytes)) goto done;
    const size_t spare=e->workspace_limit-aux_bytes-metadata_bytes;
    const size_t point_bytes=weight_width*sizeof(*weights);
    const size_t point_storage=point_bytes+(size_t)bands_per_point*sizeof(*local_eigen);
    /* Cap the aggregate MPI payload. Keep the rest of the host budget for
       dense eigensolver workspaces and other SCF state. */
    const size_t buffers=(local_points ? point_storage : 0)+(host && !gaussian ? point_bytes : 0);
    size_t tile=buffers ? spare/2/buffers : (size_t)point_count;
    const size_t payload_limit=(size_t)64<<20;
    const size_t max_by_size=point_storage<payload_limit ? payload_limit/point_storage : 1;
    if (tile>max_by_size) tile=max_by_size;
    if (tile>(size_t)INT_MAX/weight_width) tile=INT_MAX/weight_width;
    if (tile>(size_t)point_count) tile=point_count;
    int my_tile=(int)tile, tile_points;
    MPI_Allreduce(&my_tile,&tile_points,1,MPI_INT,MPI_MIN,comm);
    if (!tile_points) goto done;
    const int local_tile=local_points<tile_points ? local_points : tile_points;
    const size_t solver_budget=spare-(size_t)local_tile*point_storage-(host && !gaussian ? (size_t)tile_points*point_bytes : 0);

    layout=malloc(((size_t)4*ranks+1)*sizeof(*layout));
    if (host) {
        eigen=malloc(eigen_items*sizeof(*eigen));
        if (surf) atomic=malloc(atom_items*sizeof(*atomic));
        if (!gaussian) weights=malloc((size_t)tile_points*point_bytes);
    }
    if (gaussian) dos=calloc(bins_items,sizeof(*dos));
    if (local_tile) {
        local_weights=malloc((size_t)local_tile*point_bytes);
        local_eigen=malloc((size_t)local_tile*bands_per_point*sizeof(*local_eigen));
    }
    if (rotation_bytes) rotation=malloc(rotation_bytes);
    if (!dos_all_ok(comm,layout && (!gaussian || dos) && (!rotation_bytes || rotation) &&
        (!host || (eigen && (gaussian || weights) && (!surf || atomic))) &&
        (!local_tile || (local_weights && local_eigen)))) goto done;

    if (rotation) {
        for (int a=1; a<=r->atom_count; ++a) {
            const int begin=orbital[a], n=orbital[a+1]-begin;
            const double axial=0.5*cos(r->theta[a]);
            const double transverse=sin(r->theta[a]);
            const double x=transverse*cos(r->phi[a]), y=transverse*sin(r->phi[a]);
            for (int i=0; i<n; ++i)
                rotation[begin+i]=(OpenMXDosSpinProjection){2*begin+i,2*begin+n+i,axial,x,y};
        }
        projection.spin=rotation;
    }

    if (host) {
        snprintf(vec_path,sizeof(vec_path),"%s%s.Dos.vec",e->directory,e->name);
        snprintf(temporary,sizeof(temporary),"%s%s.Dos.vec.%ld.tmp",e->directory,e->name,(long)getpid());
        vec=fopen(temporary,"wbx");
        if (!vec) {
            perror(temporary);
            temporary[0]='\0';
        }
    }
    if (!dos_all_ok(comm,!host || vec!=NULL)) goto done;

    int *partition=layout, *sizes=layout+ranks+1, *counts=sizes+ranks, *offsets=counts+ranks;
    MPI_Allgather(&local_points,1,MPI_INT,sizes,1,MPI_INT,comm);
    partition[0]=0;
    for (int i=0; i<ranks; ++i) partition[i+1]=partition[i]+sizes[i];
    int io_ok=1;
    for (int first=0; first<point_count;) {
        const int count=point_count-first<tile_points ? point_count-first : tile_points;
        const int end=first+count;
        const int a=e->first>first ? e->first : first, b=e->end<end ? e->end : end;
        const int local_count=b>a ? b-a : 0;
        const int ready=solve(e,m,local_count,points+a,&projection,local_eigen,local_weights,solver_budget,threads_used);
        if (!dos_all_ok(comm,ready)) goto done;

        tile_counts(ranks,partition,first,end,bands_per_point,counts,offsets);
        MPI_Gatherv(local_eigen,local_count*bands_per_point,MPI_DOUBLE,
            host ? eigen+(size_t)first*bands_per_point : NULL,counts,offsets,MPI_DOUBLE,e->host,comm);

        if (gaussian) {
            accumulate_gaussian(r,e->chemical_potential,point_count,local_count,order,spins,local_eigen,local_weights,dos);
        } else {
            tile_counts(ranks,partition,first,end,(int)weight_width,counts,offsets);
            MPI_Gatherv(local_weights,local_count*(int)weight_width,MPI_FLOAT,weights,counts,offsets,MPI_FLOAT,e->host,comm);
            if (host) {
                io_ok &= write_tetra_vectors(vec,r->grid,first,count,order,spins,bands,&projection,weights);
                if (surf) atomic_projections(r,orbital,m->noncollinear,order,spins,point_count,first,count,weights,atomic);
            }
        }
        first=end;
    }
    if (gaussian) {
        MPI_Reduce(host ? MPI_IN_PLACE : dos,host ? dos : NULL,(int)bins_items,MPI_FLOAT,MPI_SUM,e->host,comm);
        if (host) {
            const int index[3]={0,0,0};
            for (size_t i=0; i<bins_items/(size_t)order; ++i)
                io_ok &= write_vector(vec,index,dos+i*order,order);
        }
    }
    if (host) {
        io_ok &= close_output(vec);
        vec=NULL;
        if (io_ok && rename(temporary,vec_path)==0) {
            temporary[0]='\0';
            io_ok=write_dos_values(r,e,m,order,point_count,bands,eigen) &&
                write_surfaces(r,e,m,order,point_count,eigen,atomic);
        } else {
            perror(vec_path);
            io_ok=0;
        }
    }
    success=dos_all_ok(comm,io_ok);

done:
    if (vec) fclose(vec);
    if (*temporary) unlink(temporary);
    free(rotation);
    free(layout);
    free(dos);
    free(local_weights);
    free(weights);
    free(atomic);
    free(local_eigen);
    free(eigen);
    free(gamma);
    return success;
}
