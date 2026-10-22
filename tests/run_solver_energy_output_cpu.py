#!/usr/bin/env python3
"""Check real DC/DC-LNO invocation bridges with poisoned unused energy slots.

The numerical kernel is modeled as a spinor solver producing channel zero.
Production C entries, argument packets, callbacks and the C++ invocation owner
run unchanged, including DOS mode and mixed-case SCF requests. No MPI/GPU runs.
"""
import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from run_cluster_col_state_cpu import function, bounded_stack

ROOT = Path(__file__).resolve().parents[1]
PREFIX = r'''
#include "openmx_core_solver_workspace.h"
#include "openmx_core_electronic.h"
#include "openmx_core_local_basis.h"
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
#define CHECK(condition) do { if (!(condition)) { fprintf(stderr,"FAIL: line %d: %s\n",__LINE__,#condition); exit(2); } } while(0)
static int mpi_comm_level1;
static void MPI_Abort(int comm,int code) { (void)comm;(void)code;abort(); }
static int omp_get_max_threads(void) { return 1; }
static int solving,invocations;
'''
KERNEL = r'''
{
    CHECK(solver_owner != NULL && SCF_iter == 1);
    CHECK(Hks == NULL && ImNL == NULL && OLP0 == NULL && CDM == NULL && EDM == NULL);
    CHECK(electronic == NULL && local_basis == NULL);
    if (solving) {
        CHECK(Eele0[0] == 5.5 && Eele1[0] == 7.25);
        CHECK(Eele0[1] == 0.0 && Eele1[1] == 0.0);
        Eele0[0] = -4.0; Eele1[0] = -8.0;
    }
    invocations++;
    return 0.125;
}
'''
DRIVER = r'''
int main(void) {
    const double poison[] = {NAN,INFINITY,-INFINITY,1e300,-1e300,42.0,0.0,-0.0};
    char *modes[] = {"scf","SCF","ScF","dos"};
    for (size_t m=0;m<sizeof(modes)/sizeof(*modes);m++) {
        char *mode=modes[m];solving=strcasecmp(mode,"scf")==0;
        for (size_t rank=0;rank<sizeof(poison)/sizeof(*poison);rank++) {
            double Eele0[2]={5.5,poison[rank]},Eele1[2]={7.25,poison[rank]};
            CHECK(CALL == 0.125);
            if (solving) {
                const double total=Eele1[0]+Eele1[1];
                CHECK(Eele0[0]+Eele0[1]==-4.0 && total==-8.0 && isfinite(total));
                CHECK(fabs(total-(-8.0+1e-9))<1e-7);
            } else {
                CHECK(Eele0[0]==5.5 && Eele1[0]==7.25);
                CHECK((isnan(poison[rank]) && isnan(Eele0[1]) && isnan(Eele1[1])) ||
                      (Eele0[1]==poison[rank] && Eele1[1]==poison[rank]));
            }
        }
    }
    CHECK(invocations==32);
    puts("PASS: 32 poisoned-channel calls; real entry/packet/callback/core, finite identical SCF decisions, DOS outputs retained");
    return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-library', type=Path, required=True)
    parser.add_argument('--negative-control', action='store_true')
    args = parser.parse_args()
    library = args.core_library.resolve(strict=True)
    helpers = (ROOT / 'source/openmx_solver_workspace.h').read_text()
    invocation = '\n'.join(function(helpers, n) for n in ['OpenMX_Solver_Failure', 'OpenMX_Solver_Run'])
    with tempfile.TemporaryDirectory(prefix='openmx-energy-output-') as directory:
        folder = Path(directory)
        for backend in ['cuda', 'hip']:
            for variant in ['Divide_Conquer', 'Divide_Conquer_LNO']:
                path = ROOT / ('source/' + ('' if backend == 'cuda' else 'backends/hip/src/') + variant + '.c')
                text = path.read_text()
                packet = re.search(r'typedef struct\s*\{[^}]*\}\s*DC_NonCol_arguments;', text)[0]
                callback, entry = (function(text, n) for n in ['DC_NonCol_operation', 'DC_NonCol'])
                hit = re.search(r'^static double DC_NonCol_owned\([^;]+?\)\s*\{', text, re.M)
                assert hit is not None
                signature = text[hit.start():hit.end()-1].rstrip()
                call = 'DC_NonCol(mode,' + ('1,1,0,' if variant.endswith('LNO') else '1,')
                call += 'NULL,NULL,NULL,NULL,NULL,Eele0,Eele1,NULL,NULL)'
                fixture = PREFIX + invocation + packet + signature + ';\n' + callback + '\n' + entry + '\n' + signature + KERNEL + DRIVER.replace('CALL', call)
                source, exe = folder / 'check.c', folder / 'check'
                source.write_text(fixture)
                command = shlex.split(os.environ.get('CC', 'cc')) + ['-std=c11', '-O1', '-g',
                    '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-fno-pie', '-no-pie',
                    '-I' + str(ROOT / 'core/include'), str(source), str(library),
                    '-Wl,-rpath,' + str(library.parent), '-lm', '-o', str(exe)]
                subprocess.run(command, check=True)
                subprocess.run([str(exe)], check=True, preexec_fn=bounded_stack)
                if args.negative_control:
                    former = re.sub(r'\n\s*/\* A spinor solve.*?\*/\s*if \(strcasecmp\(mode,"scf"\)==0\) \{\s*Eele0\[1\] = 0.0;\s*Eele1\[1\] = 0.0;\s*\}', '', entry, flags=re.S)
                    assert former != entry, 'missing regression guard'
                    source.write_text(fixture.replace(entry, former))
                    subprocess.run(command, check=True)
                    result = subprocess.run([str(exe)], capture_output=True, text=True, preexec_fn=bounded_stack)
                    assert result.returncode==2 and 'Eele0[1] == 0.0' in result.stderr, result
                print('PASS:', backend, variant, 'and former-entry negative control' if args.negative_control else '', flush=True)


if __name__ == '__main__':
    main()
