#!/usr/bin/env python3
"""Exercise the production Force3 point-group OpenMP loop with a slow reader.

Both native backends run the extracted control flow. A delayed non-master
thread must consume each group before its shared plan is reused. Removing
the entry barrier is a negative control: it reproduces the lost iteration
and mismatched OpenMP barriers without MPI or a GPU.
"""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ENTRY = '''                    /* Every thread must consume the previous group's h_end
                       and failure flag before the master replaces the plan. */
#pragma omp barrier
'''
HARNESS = r'''
#define _POSIX_C_SOURCE 200809L
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define F3_POINT_BLOCK 1
enum { groups = 16 };
typedef struct { int view, h_end; void *visit_pair; double sum[3]; } Points;
static Points f3_points;
static int plans, calls, failure, fail_at;
static int Force3_points_plan(Points *p,int mc,int gc,int no,int rank,
    void *dm,void *chi,void *pot,int mode,int begin) {
    if (begin != plans) failure = 1;
    p->h_end = begin + 1;
    return plans++ != fail_at;
}
static int openmx_core_force_grid_points(const int *view,int first,int count,
    void *scratch,size_t bytes,int mode) { ++calls; return 1; }
static void Force3_points_sum(Points *p,int mc) {
    p->sum[0] += p->h_end;
    p->sum[1] += p->h_end * p->h_end;
    p->sum[2] -= p->h_end;
}
static void delay_reader(void) {
    if (omp_get_thread_num() == 1) {
        const struct timespec pause = {0, 5000000};
        nanosleep(&pause, NULL);
    }
}
int main(int argc,char **argv) {
    fail_at = argc > 1 ? atoi(argv[1]) : -1;
    int FNAN[1] = {groups-1}, GridN_Atom[1] = {1};
    int f3_points_failed=0, f3_points_ok=0, Mc_AN=1, Gc_AN=0, NO0=1, myid=0;
    int f3_host_mode=0;
    void *DM=NULL,*dChi0=NULL,*Vpot_grid=NULL,*f3_points_scratch=NULL;
    size_t f3_points_scratch_bytes=0;
    double sumx=-1, sumy=-1, sumz=-1;
    int done[3]={0};
    f3_points.visit_pair=&plans;
    omp_set_dynamic(0);
#pragma omp parallel num_threads(3)
    {
        /* PRODUCTION_LOOP */
        done[omp_get_thread_num()]=f3_done;
    }
    const int expected=fail_at<0?groups:fail_at;
    if(failure || calls!=expected || plans!=expected+(fail_at>=0))return 1;
    for(int i=0;i<3;++i)if(done[i]!=(fail_at<0))return 2;
    const double total=fail_at<0?groups*(groups+1)/2:0;
    const double squares=fail_at<0?groups*(groups+1)*(2*groups+1)/6:0;
    if(sumx!=total || sumy!=squares || sumz!=-total)return 3;
    printf("PASS: %d groups, plan failure %d, three synchronized threads\n",calls,fail_at);
    return 0;
}
'''


def main():
    compiler = shlex.split(os.environ.get('CC', 'cc'))
    with tempfile.TemporaryDirectory(prefix='openmx-force3-threads-') as directory:
        folder = Path(directory)
        for backend, path in [('cuda', 'source/Force.c'), ('hip', 'source/backends/hip/src/Force.c')]:
            source = (ROOT/path).read_text()
            start = source.index('int f3_done = 0;')
            end = source.index('if (!f3_done) {', start)
            loop = source[start:end]
            assert loop.count(ENTRY) == 1
            advance = 'h_begin = f3_points.h_end;'
            assert loop.count(advance) == 1
            loop = loop.replace(advance, 'delay_reader();\n' + advance)
            for negative in (False, True):
                text = loop.replace(ENTRY, '') if negative else loop
                name = backend + ('-missing-barrier' if negative else '')
                src, binary = folder/(name+'.c'), folder/name
                src.write_text(HARNESS.replace('/* PRODUCTION_LOOP */', text))
                subprocess.run(compiler+['-std=c11', '-O2', '-fopenmp', str(src), '-o', str(binary)], check=True)
                if negative:
                    try:
                        result = subprocess.run([str(binary)], capture_output=True, timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                    else:
                        assert result.returncode != 0, 'Missing-barrier control unexpectedly passed'
                    print(backend + ': missing-barrier negative control rejected', flush=True)
                else:
                    for fail_at in (-1, 0, 5, 15):
                        subprocess.run([str(binary), str(fail_at)], check=True, timeout=5)


if __name__ == '__main__':
    main()
