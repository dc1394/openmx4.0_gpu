#!/usr/bin/env python3
"""Check first-SCF eigenstate counts in the production spin-union GS2 route.

This dispatch regression uses the production branch and partition functions.
It prevents a zero-state solve when adapting the refined native path's API.
No GPU or external numerical library is needed.
"""
from pathlib import Path
import os
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def block(source, start):
    end = source.index('{', start) + 1
    depth = 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def main():
    source = (ROOT / 'source/Cluster_DFT_Col.c').read_text()
    functions = []
    for name in ('ClusterCol_GpuPathMaxN', 'ClusterCol_SetMaxNAndPartitions'):
        match = re.search(r'^static (?:int|void) ' + name + r'\(', source, re.M)
        functions.append(block(source, match.start()))
    branch = block(source, source.index('if (!gpusolver2_flag && SpinP_switch==1 && numprocs0>1 &&'))
    code = r'''
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
static double system_charge;
static int cal_partial_charge;
static int solved;
#define CHECK(c) do { if(!(c)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c);exit(1); } } while(0)
#define ClusterCol_UnionGs2Path(n,maxn,...) do { CHECK((maxn)>0 && (maxn)<=(n)); solved=(maxn); } while(0)
'''
    code += '\n'.join(functions)
    code += r'''
static void run(int n,int numprocs1,int SCF_iter) {
    int gpusolver2_flag=0,SpinP_switch=1,numprocs0=2*numprocs1;
    int ClusterCol_gs2_fallback_n=n,firsttime=1,MaxN=-1,is2[32],ie2[32];
    struct {int n;} ClusterCol_ugrid={n};
    const char *mode="scf";
    double TZ=8.0;
    solved=0;
'''
    code += branch
    code += r'''
    CHECK(0);
diagonalize_finished:
    CHECK(firsttime==0 && solved==MaxN);
    int expected=SCF_iter==1?n:(n<404?n:404),count=0;
    CHECK(MaxN==expected);
    for(int i=0;i<numprocs1;++i) {
        if(ie2[i]>=is2[i]) {CHECK(is2[i]==count+1); count+=ie2[i]-is2[i]+1;}
    }
    CHECK(count==expected);
}
int main(void) {
    int sizes[]={3,7,23,511},ranks[]={1,2,4,8,16,24},checks=0;
    for(int i=0;i<4;++i)for(int j=0;j<6;++j)for(int scf=1;scf<=2;++scf) {
        run(sizes[i],ranks[j],scf);++checks;
    }
    printf("PASS: %d spin-union eigenstate/partition cases\n",checks);
    return 0;
}
'''
    with tempfile.TemporaryDirectory(prefix='openmx-gs2-states-') as directory:
        tmp = Path(directory)
        (tmp / 'check.c').write_text(code)
        compiler = shlex.split(os.environ.get('CC', 'cc'))
        subprocess.run(compiler + ['-std=c11', '-O1', '-g', '-Wall', '-Wextra',
            '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
            str(tmp / 'check.c'), '-o', str(tmp / 'check')], check=True)
        subprocess.run([str(tmp / 'check')], check=True)


if __name__ == '__main__':
    main()
