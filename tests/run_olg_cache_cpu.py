#!/usr/bin/env python3
"""Exercise the production host overlap search and its sizing-pass ownership.

The real C++ cache and host SIMD search are linked to the extracted native C
adapter. A final pass with unavailable temporary grids must reuse the sizing
pass's lists. Native free() then checks the actual ownership-transfer ABI.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
from native_state_test_support import bounded_stack, compiler_command

ROOT = Path(__file__).resolve().parents[1]


def function(source, declaration):
    start = source.rindex(declaration)
    at = source.index('{', start)
    end, depth = at + 1, 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


HARNESS = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "openmx_core_olg.h"
#include "openmx_core_olg_cache.h"
#include "openmx_core_host_simd.h"
static int Matomnum=2,atomnum=3,TCpyCell=2,MD_iter=1;
static int M2G[4]={0,2,1,3},F_G2M[4]={0,2,1,3};
static int FNAN[4]={0,1,2,0},GridN_Atom[4]={0,7,6,5};
static int neighbor0[3]={0,0,0},neighbor1[2]={1,2},neighbor2[3]={2,1,3};
static int *natn[4]={neighbor0,neighbor1,neighbor2,neighbor0};
static int cell1[2]={0,1},cell2[3]={0,2,1},*ncn[4]={neighbor0,cell1,cell2,neighbor0};
static int atv_ijk[3][4]={{0,0,0,0},{0,1,0,0},{0,-1,0,0}};
static int Ngrid1=4,Ngrid2=4,Ngrid3=4,List_YOUSO[40];
static int **NumOLG,***GListTAtoms1,***GListTAtoms2;
static int host_mode;
static int OpenMX_HostSimdMode(void){return host_mode;}
static void OpenMX_HostSimdReport(const char* label){(void)label;}
static int MPI_Abort(int comm,int code){(void)comm;exit(code);}
static int mpi_comm_level1;
static int openmx_periodic_index(int cpy,int a,int b,int c){
  assert(cpy==1);return b||c||a < -1||a>1?-1:a==0?0:a==1?1:2;
}
#include "olg_cache_native.inc"

static const int grids[4][7]={{0},{1,2,2,7,9,12,15},{1,2,2,7,12,16},{1,2,7,12,15}};
static const int cells[4][7]={{0},{0,0,1,2,1,0,0},{0,2,0,1,0,0},{1,2,0,2,0}};
static void outer(void){
  GListTAtoms1=calloc((size_t)Matomnum+1,sizeof(int**));GListTAtoms2=calloc((size_t)Matomnum+1,sizeof(int**));
  assert(GListTAtoms1&&GListTAtoms2);
}
static void clear(void){
  for(int a=0;a<=Matomnum;++a){
    for(int h=0;h<=FNAN[M2G[a]];++h){free(GListTAtoms1[a][h]);free(GListTAtoms2[a][h]);}
    free(GListTAtoms1[a]);free(GListTAtoms2[a]);
  }
  free(GListTAtoms1);free(GListTAtoms2);GListTAtoms1=GListTAtoms2=NULL;
}
// Independent scalar reference: first matching centre entry, in neighbour
// list order, including repeated grid indices and periodic cell shifts.
static void check(int total){
  int expected_total=0;
  for(int a=1;a<=Matomnum;++a){
    const int global=M2G[a];
    for(int h=0;h<=FNAN[global];++h){
      const int other=natn[global][h],shift=ncn[global][h];int count=0;
      for(int p=0;p<GridN_Atom[other];++p){
        const int target=openmx_periodic_index(1,atv_ijk[shift][1]+atv_ijk[cells[other][p]][1],0,0);
        if(target<0)continue;
        for(int q=0;q<GridN_Atom[global];++q)if(grids[global][q]==grids[other][p] && cells[global][q]==target){
          assert(GListTAtoms1[a][h][count]==q && GListTAtoms2[a][h][count]==p);++count;break;
        }
      }
      assert(NumOLG[a][h]==count);expected_total+=count;
    }
  }
  assert(total==expected_total);
}
int main(int argc,char** argv){
  assert(argc==3);host_mode=openmx_core_host_simd_mode(argv[1]);assert(host_mode>=0);
  const int requested=atoi(argv[2]);omp_set_num_threads(requested);
  int observed=0;
#pragma omp parallel reduction(+:observed)
  {observed+=1;}
  assert(observed==requested);List_YOUSO[12]=100;
  int *grid[4]={NULL,(int*)grids[2],(int*)grids[1],(int*)grids[3]};
  int *cell[4]={NULL,(int*)cells[2],(int*)cells[1],(int*)cells[3]};
  NumOLG=calloc(3,sizeof(int*));assert(NumOLG);
  for(int a=0;a<=Matomnum;++a){NumOLG[a]=calloc((size_t)FNAN[M2G[a]]+1,sizeof(int));assert(NumOLG[a]);}
  char e[256];OpenMXCoreContext* core=openmx_core_create_host(e,sizeof(e));assert(core);
  for(int repeat=0;repeat<4;++repeat){
    int total=0;const int step=repeat+1;
    assert(UCellBox_OLG_Host(core,step,2,1,grid,cell,&total));
    const OpenMXOlgCacheInput key={step,1,(size_t)Matomnum,(size_t)atomnum,M2G,FNAN};int hit=0;
    assert(openmx_core_olg_cache_matches(core,&key,&hit,e,sizeof(e))&&hit);
    outer();
    // No temporary grid is available: recomputing the search is an error.
    assert(UCellBox_OLG_Host(core,step,0,1,NULL,NULL,&total));check(total);clear();
    assert(openmx_core_olg_cache_matches(core,&key,&hit,e,sizeof(e))&&!hit);
    // A final pass with no preceding sizing pass must also work.
    total=0;outer();assert(UCellBox_OLG_Host(core,step,0,1,grid,cell,&total));check(total);clear();
  }
  // A second calculation destroys an unconsumed sizing cache. Neither that
  // destruction nor finish may disturb the first calculation's pending one.
  OpenMXCoreContext* other=openmx_core_create_host(e,sizeof(e));assert(other);int total=0;
  assert(UCellBox_OLG_Host(core,11,2,1,grid,cell,&total));
  assert(UCellBox_OLG_Host(other,21,2,1,grid,cell,&total));openmx_core_destroy(other);
  outer();assert(UCellBox_OLG_Host(core,11,0,1,NULL,NULL,&total));check(total);clear();
  assert(UCellBox_OLG_Host(core,12,2,1,grid,cell,&total));openmx_core_finish(core);
  for(int a=0;a<=Matomnum;++a)free(NumOLG[a]);free(NumOLG);
  Matomnum=0;NumOLG=calloc(1,sizeof(int*));NumOLG[0]=calloc(1,sizeof(int));
  total=0;assert(UCellBox_OLG_Host(core,1,2,1,grid,cell,&total));
  outer();assert(UCellBox_OLG_Host(core,1,0,1,NULL,NULL,&total));assert(total==0);clear();
  free(NumOLG[0]);free(NumOLG);openmx_core_destroy(core);
  puts("PASS: native overlap sizing reuse, transfer, uncached pass, empty partition and independent contexts");return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    parser.add_argument('--negative-control', action='store_true')
    parser.add_argument('--openmp-arg', action='append', help='OpenMP compiler/linker argument; repeat as needed')
    parser.add_argument('--serial', action='store_true', help='Exercise one worker without an OpenMP runtime')
    args = parser.parse_args()
    if args.serial and args.openmp_arg:
        parser.error('--serial and --openmp-arg cannot be combined')
    library = args.core_library.resolve(strict=True)
    source = (ROOT / 'source/truncation.c').read_text()
    native = '\n\n'.join(function(source, name) for name in (
        'static void OLG_cache_take(', 'static int UCellBox_OLG_Host('))
    env = {k: v for k, v in os.environ.items() if not k.startswith(('OPENMX_', 'GEMMUL8_'))
           and k not in ('LD_PRELOAD', 'LD_LIBRARY_PATH')}
    env.update(OMP_DYNAMIC='FALSE', OMP_PROC_BIND='false')
    openmp_args = [] if args.serial else (args.openmp_arg or ['-fopenmp'])
    with tempfile.TemporaryDirectory(prefix='openmx-olg-cache-') as directory:
        folder = Path(directory)
        (folder / 'main.c').write_text(HARNESS)
        if args.serial:
            (folder/'omp.h').write_text('#include <assert.h>\n'
                'static inline void omp_set_num_threads(int n){assert(n==1);}\n'
                'static inline int omp_in_parallel(void){return 0;}\n')

        def build(name, body):
            (folder / 'olg_cache_native.inc').write_text(body)
            binary = folder / name
            subprocess.run(compiler_command('c') + ['-I' + str(folder),
                '-I' + str(ROOT / 'core/include'), '-x', 'c', str(folder / 'main.c'),
                '-x', 'none', str(library), '-Wl,-rpath,' + str(library.parent)] +
                openmp_args + ['-o', str(binary)], check=True)
            return binary

        binary = build('native-cache', native)
        for mode in ('scalar', 'auto'):
            for threads in ((1,) if args.serial else (1, 2, 8)):
                subprocess.run([str(binary), mode, str(threads)], cwd=folder, env=env,
                               preexec_fn=bounded_stack, check=True, timeout=60)
        if args.negative_control:
            marker = 'if(estimate_switch==0 && cached)'
            assert native.count(marker) == 1
            broken = build('discarded-cache', native.replace(marker, 'if(0 && cached)'))
            result = subprocess.run([str(broken), 'scalar', '1'], cwd=folder, env=env,
                                    preexec_fn=bounded_stack, capture_output=True, text=True, timeout=60)
            assert result.returncode != 0, 'discarding the sizing cache escaped the native reuse test'
    print('PASS: native overlap cache C ABI and sizing-pass ownership ('+
          ('serial' if args.serial else 'real OpenMP teams: 1, 2, 8')+')')


if __name__ == '__main__':
    main()
