#!/usr/bin/env python3
"""Run the production band matrix packing and gather with a nonzero GPU root."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

from native_state_test_support import bounded_stack

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "source/Band_DFT_kpath_gpu.c").read_text()
exchange_header = (ROOT / "source/openmx_exact_exchange.h").read_text()
end = exchange_header.index("} OpenMXExactExchangeOutputView;") + len("} OpenMXExactExchangeOutputView;")
exchange_type = exchange_header[exchange_header.rindex("typedef struct {", 0, end):end]
types = source[source.index("typedef struct {"):source.index("static int all_ranks_ok")]
pack = source[source.index("static int values_per_orbital_pair"):source.index("#if OPENMX_GPU_CUDA\nstatic int size_exchange")]
exchange_pack = source[source.index("static void pack_exchange"):source.index("\n#endif\n\nstatic int rank_counts")]
exchange_mpi = source[source.index("    OpenMXKPathBlock *local_block_target"):source.index("    m.blocks = blocks;")]
prefix = r'''
#include <assert.h>
#include <limits.h>
#include <mpi.h>
#include <omp.h>
#include <stdlib.h>
#include "openmx_band_kpath_gpu.h"
#define OPENMX_GPU_CUDA 1
static double element(int channel,int world,int neighbor,int row,int col)
{ return 10000*channel+1000*world+100*neighbor+10*row+col; }
'''
fixture = r'''
static void check(int rank, int mode)
{
    /* Rank zero has no atoms. Rank one contributes a prefix before the first
       GPU owner, and rank three checks that owner's broadcast. */
    enum {COUNT=4, CHANNELS=8, ATOM_MAX=3};
    const int input_root=2, owner_count=2, owner=rank>=input_root, gather_root=rank==input_root;
    MPI_Comm comm=MPI_COMM_WORLD, devices=MPI_COMM_NULL;
    MPI_Comm_split(comm,owner?0:MPI_UNDEFINED,rank,&devices);

    const int local_atoms=rank!=0;
    const int paos[COUNT]={0,2,3,1}, species[COUNT]={0,1,2,3}, counts[COUNT]={0,1,1,1};
    const int to_global[2]={0,rank}, zero_based[COUNT]={0,0,2,5};
    int targets[COUNT][2]={{0,0},{1,2},{2,3},{3,1}}, shift[4]={0,3,-2,1}, shifts[2]={0,0};
    int *neighbors[COUNT]={targets[0],targets[1],targets[2],targets[3]};
    int *cells[COUNT]={shifts,shifts,shifts,shifts}, *translations[1]={shift};
    const KPathBasis basis={3,local_atoms,to_global,NULL,species,paos,NULL,counts,neighbors,cells,translations};
    const OpenMXLocalBasisViews *local_basis=NULL;
    OpenMXKPathMatrices m={.spin_channels=mode==1?2:1,.noncollinear=mode>=2,.imaginary_diagonals=mode==3};
    int offset[COUNT+1];
    assert(orbital_offsets(&m,&basis,offset)==6);

    double data[CHANNELS][2][ATOM_MAX][ATOM_MAX], *rows[CHANNELS][2][ATOM_MAX];
    double **neighbors_data[CHANNELS][2], ***atoms_data[CHANNELS][2];
    for (int ch=0;ch<CHANNELS;++ch) {
        atoms_data[ch][1]=neighbors_data[ch];
        for (int h=0;h<2;++h) {
            neighbors_data[ch][h]=rows[ch][h];
            for (int i=0;i<ATOM_MAX;++i) {
                rows[ch][h][i]=data[ch][h][i];
                for (int j=0;j<ATOM_MAX;++j) data[ch][h][i][j]=element(ch,rank,h,i,j);
            }
        }
    }
    double ****overlap=atoms_data[0], ****hamiltonian[4]={atoms_data[1],atoms_data[2],atoms_data[3],atoms_data[4]};
    double ****imaginary[3]={atoms_data[5],atoms_data[6],atoms_data[7]};

    KPathLocalOffset local_offsets[2];
    assert(local_block_offsets(&m,&basis,offset,local_offsets));
    const KPathLocalOffset end=local_offsets[local_atoms];
    int local_size[2]={end.blocks*(int)sizeof(OpenMXKPathBlock),end.values}, sizes[COUNT][2];
    MPI_Allgather(local_size,2,MPI_INT,sizes,2,MPI_INT,comm);
    int block_sizes[COUNT], block_offsets[COUNT], value_sizes[COUNT], value_offsets[COUNT], block_bytes=0, value_count=0;
    for (int p=0;p<COUNT;++p) {
        block_offsets[p]=block_bytes; block_bytes+=(block_sizes[p]=sizes[p][0]);
        value_offsets[p]=value_count; value_count+=(value_sizes[p]=sizes[p][1]);
    }
    const int semilocal_blocks=block_bytes/(int)sizeof(OpenMXKPathBlock), width=values_per_orbital_pair(&m);
    const int exchange_status=mode<2, exx_atoms[2]={1,2}, exx_shift[3]={-1,0,2}, exx_orbitals[COUNT]={0,2,5,6};
    const size_t exx_offset[2]={0,3};
    const double exx_up[3]={101,103,107}, exx_down[3]={109,113,127};
    const OpenMXExactExchangeOutputView exchange={3,6,m.spin_channels,1,exx_atoms,exx_shift,exx_orbitals,
                                                    exx_offset,{exx_up,exx_down}};
    m.exchange_offset=value_count;
    OpenMXKPathBlock *local=gather_root?NULL:malloc((size_t)(local_size[0]?local_size[0]:1));
    double *local_values=gather_root?NULL:malloc((size_t)(local_size[1]?local_size[1]:1)*sizeof(double));
    OpenMXKPathBlock *blocks=owner?malloc((size_t)(semilocal_blocks+1)*sizeof(*blocks)):NULL;
    double *values=owner?malloc((size_t)(value_count+3*m.spin_channels)*sizeof(*values)):NULL;
    assert(gather_root || (local && local_values));
    assert(!owner || (blocks && values));
'''
verify = r'''
    if (owner) {
        int block=0, value=0;
        for (int world=1;world<COUNT;++world)for (int h=0;h<2;++h) {
            const int n=paos[world], other=targets[world][h], p=paos[other];
            const OpenMXKPathBlock b=blocks[block++];
            assert(b.row==zero_based[world] && b.col==zero_based[other] && b.rows==n && b.cols==p && b.value_offset==value);
            assert(b.translation[0]==3 && b.translation[1]==-2 && b.translation[2]==1);
            for (int j=0;j<p;++j)for (int i=0;i<n;++i)for (int ch=0;ch<width;++ch) {
                double want=element(ch,world,h,i,j);
                if (mode==3 && ch==4) want+=element(7,world,h,i,j);
                assert(values[value++]==want);
            }
        }
        assert(block==semilocal_blocks && value==value_count);
        if (exchange_status) {
            assert(blocks[block].row==2 && blocks[block].col==5 && blocks[block].value_offset==value);
            for (int i=0;i<3;++i)for (int spin=0;spin<m.spin_channels;++spin)
                assert(values[value++]==exchange.values[spin][i]);
        }
    }
    free(blocks);
    free(values);
    if (devices!=MPI_COMM_NULL) MPI_Comm_free(&devices);
}

int main(int argc,char **argv)
{
    MPI_Init(&argc,&argv);
    int rank,ranks;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    assert(ranks==4);
    for (int mode=0;mode<4;++mode) check(rank,mode);
    if (!rank) puts("PASS: band packing with an empty rank, nonzero GPU root and local exchange tails");
    MPI_Finalize();
}
'''

compiler = shlex.split(os.environ.get("MPICC", shutil.which("mpicc") or "mpicc"))
launcher = os.environ.get("MPIEXEC") or str(Path(shutil.which(compiler[0]) or compiler[0]).parent / "mpirun")
env = dict(os.environ)
env.setdefault("OMPI_CC", "cc")
with tempfile.TemporaryDirectory(prefix="openmx-band-mpi-") as directory:
    path = Path(directory)
    cfile, binary = path / "check.c", path / "check"
    cfile.write_text("#include <stdio.h>\n" + "\n".join((prefix, types, exchange_type, pack, exchange_pack,
                                                              fixture, exchange_mpi, verify)))
    subprocess.run(compiler + ["-std=c11", "-O2", "-fopenmp", "-I" + str(ROOT / "source"),
                               str(cfile), "-o", str(binary)], env=env, check=True)
    subprocess.run([launcher, "-np", "4", str(binary)], env=env, check=True, timeout=60, preexec_fn=bounded_stack)
