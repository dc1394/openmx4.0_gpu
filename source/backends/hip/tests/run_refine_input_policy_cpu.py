#!/usr/bin/env python3
"""Check production refinement input gates and collinear band cycle resets.

Extract the four solver configuration/iteration helpers and BandCol's
BeginStep. Only MPI and device configuration are stubbed; the precision
switch is linked from production. No GPU or MPI job is launched.
--negative-control checks that the former BeginStep ordering fails.
"""
import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def function(source, name):
    match = re.search(r"^static [^;{}\n]*\b" + name + r"\([^;{]*?\)\s*\{", source, re.M)
    if match is None:
        raise ValueError(f"missing function: {name}")
    pos, depth = match.end(), 1
    while depth:
        depth += (source[pos] == "{") - (source[pos] == "}")
        pos += 1
    return source[match.start():pos]


def typedef(source, name):
    end = source.index("} " + name + ";") + len("} " + name + ";")
    return source[source.rfind("typedef struct", 0, end):end]


PREFIX = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_precision.h"
#define CHECK(c, message) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", message); return 2; } } while (0)
enum { MPI_INT, MPI_DOUBLE };
static int mpi_comm_level1, Host_ID, level_stdout=2;
static int config_calls, broadcasts, expected_complex;
static void MPI_Comm_rank(int comm, int *rank) { (void)comm; *rank=0; }
static void MPI_Bcast(void *data, int count, int type, int root, int comm)
{ (void)data; (void)count; (void)type; (void)root; (void)comm; broadcasts++; }
static int openmx_eigen_refine_configure(int cplx, int verbose, int *ratio)
{
    if(cplx!=expected_complex || verbose!=1) abort();
    config_calls++; *ratio=32; return 2;
}
static double openmx_eigen_refine_until(void) { return 0.001; }
'''

POLICY = r'''
static int check_policy(void)
{
    memset(&STATE,0,sizeof(STATE)); STATE.iterations=-1;
    openmx_set_fp32_enabled(0);
    CONFIGURE();
    CHECK(STATE.iterations==-1 && config_calls==0 && broadcasts==0,
          "disabled input was cached or entered configuration collectives");
    CHECK(ITERATIONS()==0,"disabled input allowed refinement");
    openmx_set_fp32_enabled(1);
    CONFIGURE();
    CHECK(config_calls==1 && broadcasts==1 && ITERATIONS()==2,
          "enabled input did not configure after disabled input");
    CHECK(STATE.defaulted && STATE.until==0.001,"configured policy fields lost");
    openmx_set_fp32_enabled(0);
    CONFIGURE();
    CHECK(ITERATIONS()==0 && STATE.iterations==2 && config_calls==1 && broadcasts==1,
          "disabled input failed to override cached preference");
    openmx_set_fp32_enabled(1);
    CONFIGURE();
    CHECK(ITERATIONS()==2 && config_calls==1 && broadcasts==1,
          "enabled input did not restore cached preference");
    return 0;
}
'''

BAND_STUBS = r'''
static int Cnt_switch, DFTSCF_loop=5, xanes_calc, empty_occupation_flag, empty_states_flag;
static struct { int basis_valid; } BandCol_refine_state;
static struct { int refined; } *BandCol_kref;
static int BandCol_kref_slots, reset_calls;
static void BandCol_KRefReset(void) { reset_calls++; BandCol_kref=NULL; BandCol_kref_slots=0; }
'''

BAND_CHECK = r'''
static int check_band_cycle(void)
{
    memset(&BandCol_refine,0,sizeof(BandCol_refine));
    BandCol_refine.iterations=-1;
    openmx_set_fp32_enabled(1);
    BandCol_RefineBeginStep(1,1,1,1);
    CHECK(BandCol_refine.last_refined,"first-step refinement after initial configuration");
    CHECK(BandCol_refine.transient && !BandCol_refine.kdense && !BandCol_refine.staged,
          "first-step resident/two-pass gates");
    BandCol_refine.final_stage=1;
    BandCol_refine.cycle_refined=1;
    BandCol_refine_state.basis_valid=1;
    BandCol_RefineBeginStep(1,1,1,1);
    CHECK(BandCol_refine.last_refined && !BandCol_refine.final_stage,
          "new cycle inherited final FP64 stage");
    CHECK(!BandCol_refine_state.basis_valid && !BandCol_refine.cycle_refined && reset_calls==2,
          "new cycle retained basis or history");
    BandCol_RefineBeginStep(2,1,1,1);
    CHECK(BandCol_refine.last_refined && BandCol_refine.kdense && BandCol_refine.staged,
          "later-step refinement paths");
    openmx_set_fp32_enabled(0);
    BandCol_RefineBeginStep(2,1,1,1);
    CHECK(!BandCol_refine.last_refined && !BandCol_refine.kdense && !BandCol_refine.staged,
          "input off did not suppress all refinement paths");
    openmx_set_fp32_enabled(1);
    BandCol_RefineBeginStep(1,1,1,1);
    CHECK(BandCol_refine.last_refined,"input on did not restore first-step refinement");
    for(int veto=0;veto<6;veto++) {
        BandCol_refine.force_fp64=veto==0; Cnt_switch=veto==1;
        xanes_calc=veto==2; empty_occupation_flag=veto==3; empty_states_flag=veto==4;
        BandCol_RefineBeginStep(veto==5 ? DFTSCF_loop : 2,1,1,1);
        CHECK(!BandCol_refine.last_refined && !BandCol_refine.kdense && !BandCol_refine.staged,
              "explicit refinement veto was ignored");
    }
    return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args()
    wrappers = [
        ("Cluster_DFT_Col.c", "ClusterCol", "ClusterColRefineStage", 0),
        ("Cluster_DFT_NonCol.c", "ClusterNonCol", "ClusterNonColRefineStage", 1),
        ("Band_DFT_Col.c", "BandCol", "BandColRefineState", 1),
        ("Band_DFT_NonCol.c", "BandNonCol", "BandNonColRefineState", 1),
    ]
    compiler = shlex.split(os.environ.get("CC", "cc"))
    flags = ["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror"] + shlex.split(os.environ.get("CFLAGS", ""))
    with tempfile.TemporaryDirectory(prefix="openmx-refine-policy-") as tmp:
        for filename, prefix, typename, cplx in wrappers:
            source = (ROOT / "source" / filename).read_text()
            state = prefix + "_refine"
            policy = POLICY.replace("STATE", state).replace("CONFIGURE", prefix + "_RefineConfigure")
            policy = policy.replace("ITERATIONS", prefix + "_RefineIterations")
            text = PREFIX + typedef(source, typename) + f"\nstatic {typename} {state};\n"
            text += function(source, prefix + "_RefineConfigure") + "\n"
            text += function(source, prefix + "_RefineIterations") + "\n" + policy
            begin = ""
            if prefix == "BandCol":
                begin = function(source, "BandCol_RefineBeginStep")
                text += BAND_STUBS + begin + "\n" + BAND_CHECK
            text += f"\nint main(void) {{ expected_complex={cplx}; int rc=check_policy(); if(rc) return rc;"
            text += "return check_band_cycle(); }\n" if begin else "return 0; }\n"
            src, exe = Path(tmp) / f"{prefix}.c", Path(tmp) / prefix
            src.write_text(text)
            command = compiler + flags + ["-I", str(ROOT / "source"), str(src),
                                          str(ROOT / "source/openmx_precision.c"), "-o", str(exe)]
            subprocess.run(command, check=True)
            subprocess.run([str(exe)], check=True)
            print(f"PASS: {prefix} production input policy" + (" and band cycle resets" if begin else ""))
            if begin and args.negative_control:
                # Reproduce the old order without depending on what HEAD is
                # after this fix has been committed.
                assignment = re.search(r"    common = \(.*?;", begin, re.S).group(0)
                former = begin.replace(assignment, "", 1)
                former = former.replace("    int common;", assignment.replace("common =", "const int common ="), 1)
                src.write_text(text.replace(begin, former))
                subprocess.run(command, check=True)
                result = subprocess.run([str(exe)], capture_output=True, text=True)
                if result.returncode != 2 or "first-step refinement after initial configuration" not in result.stderr:
                    raise AssertionError(f"negative control did not fail as expected: {result}")
                print("PASS: former BandCol BeginStep ordering fails the regression")


if __name__ == "__main__":
    main()
