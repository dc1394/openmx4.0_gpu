#!/usr/bin/env python3
"""Exercise production dense-cache cleanup with host-only HIP mocks.

No MPI, HIP library or GPU is required. The cold mock rejects every HIP
operation with hipErrorNoDevice, reproducing CPU runs with hidden devices.
"""
import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest


def default_source():
    base = Path(__file__).resolve().parents[1]
    for candidate in (base / 'src/Band_DFT_NonCol.c',
                      base / 'source/Band_DFT_NonCol.c'):
        if candidate.is_file():
            return candidate
    return base / 'src/Band_DFT_NonCol.c'


def function(text, name, optional=False):
    match = re.search(r'^static\s+[^\n]+\b' + re.escape(name) +
                      r'\s*\([^)]*\)\s*\{', text, re.M)
    if match is None:
        if optional:
            return ''
        raise ValueError('Missing production function ' + name)
    # Ignore braces inside comments and strings while extracting the body.
    tokens = re.compile(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|'
                        r"'(?:\\.|[^'\\])*'|[{}]", re.S)
    depth = 1
    for token in tokens.finditer(text, match.end()):
        if token[0] == '{':
            depth += 1
        elif token[0] == '}':
            depth -= 1
            if depth == 0:
                return text[match.start():token.end()]
    raise ValueError('Unterminated production function ' + name)


def production(text):
    end = re.search(r'}\s*BandNonColDenseGpuWorkspace\s*;', text)
    if end is None:
        raise ValueError('Missing production dense workspace type')
    start = text.rfind('typedef struct', 0, end.start())
    if start < 0:
        raise ValueError('Missing production dense workspace declaration')
    result = text[start:end.end()] + '\n'
    result += 'static BandNonColDenseGpuWorkspace BandNonCol_dense_gpu_workspace = {0};\n'
    result += function(text, 'BandNonCol_DenseGpu_HasDeviceResources', optional=True) + '\n'
    for name in ('ReleaseHipblas', 'Destroy', 'ProcessCleanup',
                 'ReleaseWaveBuffers', 'ReleaseOverlapMagmaScratch',
                 'ReleasePreMagmaBuffers'):
        result += function(text, 'BandNonCol_DenseGpu_' + name) + '\n'
    return result


MOCK = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct { double r, i; } dcomplex;
typedef void *hipblasHandle_t;
static int available, calls, syncs, device_frees, handle_frees, device_sets;
static int host_frees;
static void *host_allocation;
#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr,"check failed at line %d: %s\n",__LINE__,#condition); exit(2); \
} } while (0)
static int hipDeviceSynchronize(void) {
    ++calls; ++syncs; return available ? 0 : 100;
}
static int hipSetDevice(int device) {
    ++calls; ++device_sets; CHECK(device==3); return available ? 0 : 100;
}
static int hipFree(void *pointer) {
    ++calls; ++device_frees; CHECK(available && pointer); free(pointer); return 0;
}
static int hipblasDestroy(hipblasHandle_t handle) {
    ++calls; ++handle_frees; CHECK(available && handle); free(handle); return 0;
}
static void host_free(void *pointer) {
    if (pointer && pointer==host_allocation) { ++host_frees; host_allocation=NULL; }
    free(pointer);
}
static void checked_hip(int result) {
    if (result) { fprintf(stderr,"HIP mock rejected unused-device call: %d\n",result); exit(3); }
}
#define wait_hipfunc(operation) checked_hip(operation)
#define free(pointer) host_free(pointer)
'''


DRIVER = r'''
#undef free
static void counters(void) {
    calls=syncs=device_frees=handle_frees=device_sets=host_frees=0;
}
static void wave_state(void) {
    BandNonColDenseGpuWorkspace *w=&BandNonCol_dense_gpu_workspace;
    w->d_C2_elems=29; w->d_C2_valid=1; w->d_C2_n2=17; w->d_C2_ld=11;
}
static void wave_reset(void) {
    BandNonColDenseGpuWorkspace *w=&BandNonCol_dense_gpu_workspace;
    CHECK(!w->d_C2 && !w->d_C2_elems && !w->d_C2_valid && !w->d_C2_n2 && !w->d_C2_ld);
}
static void empty_state(void) {
    BandNonColDenseGpuWorkspace expected;
    memset(&expected,0,sizeof(expected)); expected.device_id=-1;
    CHECK(memcmp(&expected,&BandNonCol_dense_gpu_workspace,sizeof(expected))==0);
}
static void allocate_one(int which) {
    BandNonColDenseGpuWorkspace *w=&BandNonCol_dense_gpu_workspace;
    void *pointer=malloc(sizeof(dcomplex)); CHECK(pointer);
    switch(which) {
        case 0:w->d_S=pointer;break;
        case 1:w->d_H11=pointer;break;
        case 2:w->d_H22=pointer;break;
        case 3:w->d_H12=pointer;break;
        case 4:w->d_tmp=pointer;break;
        case 5:w->d_H2=pointer;break;
        case 6:w->d_C2=pointer;break;
        case 7:w->d_scale=pointer;break;
        case 8:w->hipblas=pointer;break;
        default:CHECK(0);
    }
}
static void allocate_host(void) {
    host_allocation=malloc(sizeof(dcomplex)); CHECK(host_allocation);
    BandNonCol_dense_gpu_workspace.h_scale=host_allocation;
}
int main(void) {
    BandNonColDenseGpuWorkspace *w=&BandNonCol_dense_gpu_workspace;
    /* A CPU run visits the helpers at every suite boundary. Stale host
       metadata must still be reset even though no device exists. */
    available=0; counters(); wave_state();
    BandNonCol_DenseGpu_ReleaseOverlapMagmaScratch(); wave_reset();
    wave_state(); allocate_host();
    BandNonCol_DenseGpu_ReleasePreMagmaBuffers(1); wave_reset();
    CHECK(!w->h_scale && host_frees==1 && calls==0);
    allocate_host(); w->initialized=1; w->device_id=3;
    BandNonCol_DenseGpu_ProcessCleanup(); empty_state();
    CHECK(host_frees==2 && calls==0);
    BandNonCol_DenseGpu_Destroy(); empty_state();
    BandNonCol_DenseGpu_ProcessCleanup(); empty_state(); CHECK(calls==0);
    /* Test every owned resource independently while initialized is false,
       as can happen after a partial allocation. */
    available=1;
    for(int resource=0;resource<9;resource++) {
        counters(); w->device_id=3; allocate_one(resource); wave_state();
        BandNonCol_DenseGpu_ReleaseOverlapMagmaScratch(); wave_reset();
        CHECK(syncs>=1);
        CHECK(device_frees==((resource>=1 && resource<=6)?1:0));
        CHECK(handle_frees==(resource==8));
        BandNonCol_DenseGpu_ProcessCleanup(); empty_state();
        CHECK(device_frees==(resource<8)); CHECK(handle_frees==(resource==8));
        CHECK(device_sets==((resource==0 || resource==7)?1:0));
        int previous=calls; BandNonCol_DenseGpu_ProcessCleanup(); CHECK(calls==previous);
    }
    /* A live workspace keeps overlap/scale data after scratch release,
       then progressively releases the remaining resources. */
    counters(); w->initialized=1; w->device_id=3;
    for(int resource=0;resource<9;resource++) allocate_one(resource);
    allocate_host(); wave_state();
    BandNonCol_DenseGpu_ReleaseOverlapMagmaScratch(); wave_reset();
    CHECK(device_frees==6 && handle_frees==1 && syncs==2);
    CHECK(w->d_S && w->d_scale && w->h_scale);
    BandNonCol_DenseGpu_ReleasePreMagmaBuffers(0);
    CHECK(device_frees==7 && host_frees==1 && w->d_S && !w->d_scale && !w->h_scale);
    BandNonCol_DenseGpu_ReleasePreMagmaBuffers(1);
    CHECK(device_frees==8 && syncs==4 && !w->d_S);
    int previous=calls;
    BandNonCol_DenseGpu_ReleaseOverlapMagmaScratch();
    BandNonCol_DenseGpu_ReleasePreMagmaBuffers(1);
    BandNonCol_DenseGpu_Destroy(); empty_state();
    BandNonCol_DenseGpu_ProcessCleanup(); empty_state(); CHECK(calls==previous);
    /* The process-exit path restores the owning device and also handles
       partial allocations, without wait_hipfunc/MPI error handling. */
    counters(); w->device_id=3; allocate_one(2); allocate_one(8); allocate_host();
    BandNonCol_DenseGpu_ProcessCleanup(); empty_state();
    CHECK(syncs==1 && device_sets==1 && device_frees==1 && handle_frees==1 && host_frees==1);
    previous=calls; BandNonCol_DenseGpu_ProcessCleanup(); CHECK(calls==previous);
    puts("PASS: cold CPU, nine partial resources, live releases and process cleanup");
    return 0;
}
'''


class CleanupTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which(shlex.split(os.environ.get('CC', 'cc'))[0]),
                         'A host C compiler is required')
    def test_production_cleanup(self):
        with tempfile.TemporaryDirectory(prefix='openmx-deviceless-cleanup-') as temporary:
            directory = Path(temporary)
            source = directory / 'check.c'
            binary = directory / 'check'
            source.write_text(MOCK + production(SOURCE.read_text()) + DRIVER)
            command = shlex.split(os.environ.get('CC', 'cc'))
            subprocess.run(command + ['-std=c11', '-O1', '-Wall', '-Wextra', '-Werror',
                                      str(source), '-o', str(binary)], check=True, timeout=60)
            result = subprocess.run([str(binary)], check=False, text=True,
                                    capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS: cold CPU', result.stdout)
            print(result.stdout, end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=default_source())
    args, remaining = parser.parse_known_args()
    SOURCE = args.source.resolve()
    unittest.main(argv=[__file__, *remaining])
