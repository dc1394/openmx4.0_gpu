#!/usr/bin/env python3
"""Check the production device-limited schedule against queue invariants."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

# CMake stages these tests beside native/source; also allow direct HIP-overlay runs.
root = Path(__file__).resolve().parents[1]
if not (root/'source/openmx_gpu_k_waves.h').is_file():
    root = Path(__file__).resolve().parents[4]
test = r'''
#include "openmx_gpu_k_waves.h"
#include <stdio.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"line %d: %s\n",__LINE__,#x); exit(1); } } while (0)
static unsigned rng = 1234567;
static unsigned random_value(void) { rng = rng*1664525u+1013904223u; return rng; }
static int checks;
static void verify(int points,int ranks,int devices,const int *owner,const int *map,const int *cap,int interleave) {
    int order[257],offset[258],seen[257]={0},last[32],active[32];
    int waves=openmx_gpu_k_waves(points,ranks,devices,owner,map,cap,interleave,order,offset);
    CHECK(waves>=0 && waves<=points && offset[0]==0 && offset[waves]==points);
    for(int r=0;r<ranks;r++)last[r]=-1;
    for(int w=0;w<waves;w++) {
        CHECK(offset[w]<offset[w+1]);
        memset(active,0,sizeof(active));
        for(int i=offset[w];i<offset[w+1];i++) {
            int p=order[i]; CHECK(p>=0 && p<points && !seen[p]); seen[p]=1;
            int r=owner[p]; CHECK(p>last[r]); last[r]=p; active[r]=1;
        }
        for(int d=0;d<devices;d++) {
            int owners=0;
            for(int r=0;r<ranks;r++)if(active[r] && map[r]==d)owners++;
            CHECK(owners<=cap[d]);
        }
    }
    for(int p=0;p<points;p++)CHECK(seen[p]);
    checks++;
}
int main(void) {
    int owner[257],map[32],cap[8],order[257],offset[258];
    for(int r=0;r<32;r++)map[r]=r/8;
    for(int p=0;p<64;p++)owner[p]=p/2;
    for(int d=0;d<4;d++)cap[d]=8;
    CHECK(openmx_gpu_k_waves(64,32,4,owner,map,cap,1,order,offset)==1);
    CHECK(offset[1]==64);
    for(int d=0;d<4;d++)cap[d]=1;
    CHECK(openmx_gpu_k_waves(64,32,4,owner,map,cap,1,order,offset)==8);
    CHECK(owner[order[0]]==0 && owner[order[1]]==8 && owner[order[2]]==16 && owner[order[3]]==24);
    verify(64,32,4,owner,map,cap,1);
    for(int trial=0;trial<12000;trial++) {
        int ranks=1+random_value()%32,devices=1+random_value()%8,points=random_value()%257;
        for(int r=0;r<ranks;r++)map[r]=random_value()%devices;
        for(int d=0;d<devices;d++)cap[d]=1+random_value()%8;
        for(int p=0;p<points;p++)owner[p]=random_value()%ranks;
        verify(points,ranks,devices,owner,map,cap,trial%2);
    }
    map[0]=0;cap[0]=1;owner[0]=0;
    CHECK(openmx_gpu_k_waves(0,1,1,NULL,map,cap,1,NULL,offset)==0);
    owner[0]=-1;CHECK(openmx_gpu_k_waves(1,1,1,owner,map,cap,1,order,offset)==-1);
    owner[0]=1;CHECK(openmx_gpu_k_waves(1,1,1,owner,map,cap,1,order,offset)==-1);
    owner[0]=0;map[0]=-1;CHECK(openmx_gpu_k_waves(1,1,1,owner,map,cap,1,order,offset)==-1);
    map[0]=0;cap[0]=0;CHECK(openmx_gpu_k_waves(1,1,1,owner,map,cap,1,order,offset)==-1);
    printf("PASS: %d schedules; full-node concurrency, device limits, stable owner order, uneven queues, invalid input\n",checks);
}
'''
with tempfile.TemporaryDirectory(prefix='openmx-mi300a-waves-') as d:
    path = Path(d)
    (path/'test.c').write_text(test)
    compiler = shlex.split(os.environ.get('CC', 'cc'))
    subprocess.run(compiler+shlex.split(os.environ.get('CFLAGS',''))+
                           ['-std=c11','-O1','-Wall','-Wextra','-I'+str(root/'source'),
                            str(path/'test.c'),'-o',str(path/'test')],check=True)
    subprocess.run([str(path/'test')],check=True)
