#!/usr/bin/env python3
"""Check the environment-settings registry (source/openmx_env.c) on the host.

Builds the production module with the system C compiler and a small driver
that exercises the registry ids, the snapshot, the live fallback, the one
parser behind the typed readers, the overrides that keep the process
environment in sync, the rank-0 pack/unpack synchronization, the detection
of unknown OPENMX_* names and invalid values, and the reports.
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DRIVER = r"""
#define _POSIX_C_SOURCE 200809L   /* setenv, unsetenv */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_env.h"
static int checks;
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
static int same(const char *a, const char *b) { return a != NULL && b != NULL && strcmp(a, b) == 0; }
static void set(const char *name, const char *value) { CHECK(setenv(name, value, 1) == 0); }
static char *report(int unknown_only) {
    static char buffer[8192];
    FILE *fp = tmpfile();
    size_t n;
    CHECK(fp != NULL);
    openmx_env_report(fp, unknown_only);
    rewind(fp);
    n = fread(buffer, 1, sizeof buffer - 1, fp);
    buffer[n] = '\0';
    fclose(fp);
    return buffer;
}
int main(void) {
    const OpenMXEnvEntry *entries;
    size_t count = openmx_env_registry(&entries), i;
    const char *const *unknown;
    char *packed = NULL;
    size_t length;
    /* registry: sorted, unique, ids in order, every column present */
    CHECK(count == OPENMX_ENV_COUNT && count > 200);
    for (i = 0; i < count; i++) {
        CHECK(entries[i].name && entries[i].kind && entries[i].scope && entries[i].dflt && entries[i].module && entries[i].doc);
        if (i) CHECK(strcmp(entries[i-1].name, entries[i].name) < 0);
        CHECK(openmx_env_lookup(entries[i].name) == &entries[i]);
        CHECK(openmx_env_id(entries[i].name) == (int)i);
        CHECK(same(openmx_env_name((int)i), entries[i].name));
        CHECK(same(entries[i].scope, "global") || same(entries[i].scope, "local") || same(entries[i].scope, "tool"));
    }
    CHECK(same(openmx_env_name(OPENMX_ENV_BAND_PROFILE), "OPENMX_BAND_PROFILE"));
    CHECK(same(openmx_env_name(OPENMX_ENV_EXT_GEMMUL8_DISABLE), "GEMMUL8_DISABLE"));
    CHECK(same(openmx_env_name(OPENMX_ENV_EXT_OMPI_COMM_WORLD_LOCAL_RANK), "OMPI_COMM_WORLD_LOCAL_RANK"));
    CHECK(same(entries[OPENMX_ENV_EXT_OMPI_COMM_WORLD_LOCAL_RANK].scope, "local"));
    CHECK(same(entries[OPENMX_ENV_BAND_PROFILE].kind, "flag") && same(entries[OPENMX_ENV_SETHAM_STREAM_MB].kind, "mib"));
    /* the default column: a literal fallback, a resolved macro, the reader of another name, a documented text */
    CHECK(same(entries[OPENMX_ENV_BAND_PROFILE].dflt, "0"));
    CHECK(same(entries[OPENMX_ENV_BAND_GPU_SWITCH_NUM].dflt, "800"));
    CHECK(same(entries[OPENMX_ENV_CLUSTER_GEMMUL8_TURN_RELEASE].dflt, "1"));
    CHECK(same(entries[OPENMX_ENV_FORCE4B_CASE1_STREAM].dflt, "auto"));
    CHECK(same(entries[OPENMX_ENV_BAND_GPU_RESERVE_MB].dflt, "max(total/10, 1536)"));
    CHECK(same(openmx_env_name(-1), "") && same(openmx_env_name(OPENMX_ENV_COUNT), ""));
    CHECK(openmx_env_id("OPENMX_NO_SUCH_SETTING") == -1 && openmx_env_id(NULL) == -1);
    /* before the snapshot: live, by name and by id */
    CHECK(!openmx_env_initialized());
    unsetenv("OPENMX_GPU");
    CHECK(openmx_env_get("OPENMX_GPU") == NULL && openmx_env_at(OPENMX_ENV_GPU) == NULL);
    CHECK(openmx_env_flag_at(OPENMX_ENV_GPU, 7) == 7);
    set("OPENMX_GPU", "0");
    CHECK(same(openmx_env_get("OPENMX_GPU"), "0") && same(openmx_env_at(OPENMX_ENV_GPU), "0"));
    CHECK(openmx_env_flag("OPENMX_GPU", 1) == 0 && openmx_env_flag_at(OPENMX_ENV_GPU, 1) == 0);
    CHECK(openmx_env_at(-1) == NULL && openmx_env_at(OPENMX_ENV_COUNT) == NULL);
    /* the parsers (live, before the snapshot) */
    {
        static const struct { const char *value; int flag; } flags[] = {
            {"1", 1}, {"2", 1}, {"0", 0}, {"00", 0}, {"-0", 0}, {"0.0", 0}, {"0.5", 1}, {" 1 ", 1}, {"1\n", 1},
            {"on", 1}, {"OFF", 0}, {"Yes", 1}, {"no", 0}, {"true", 1}, {"False", 0}, {"y", 1}, {"N", 0},
            {"auto", 9}, {"abc", 9}, {"", 9}, {" ", 9}, {"1abc", 9}};
        for (i = 0; i < sizeof flags / sizeof flags[0]; i++) {
            set("OPENMX_BAND_PROFILE", flags[i].value);
            CHECK(openmx_env_flag_at(OPENMX_ENV_BAND_PROFILE, 9) == flags[i].flag);
            CHECK(openmx_env_flag("OPENMX_BAND_PROFILE", 9) == flags[i].flag);
        }
        set("OPENMX_BAND_GPU_SWITCH_NUM", "300");  CHECK(openmx_env_long_at(OPENMX_ENV_BAND_GPU_SWITCH_NUM, -1) == 300);
        set("OPENMX_BAND_GPU_SWITCH_NUM", " 42 "); CHECK(openmx_env_long_at(OPENMX_ENV_BAND_GPU_SWITCH_NUM, -1) == 42);
        set("OPENMX_BAND_GPU_SWITCH_NUM", "-7");   CHECK(openmx_env_long_at(OPENMX_ENV_BAND_GPU_SWITCH_NUM, -1) == -7);
        CHECK(openmx_env_ulong_at(OPENMX_ENV_BAND_GPU_SWITCH_NUM, 5) == 5);              /* negative: not a uint */
        set("OPENMX_BAND_GPU_SWITCH_NUM", "1e3");  CHECK(openmx_env_long_at(OPENMX_ENV_BAND_GPU_SWITCH_NUM, -1) == -1);
        CHECK(openmx_env_double_at(OPENMX_ENV_BAND_GPU_SWITCH_NUM, 0.0) == 1000.0);
        set("OPENMX_BAND_GPU_SWITCH_NUM", "12abc"); CHECK(openmx_env_long_at(OPENMX_ENV_BAND_GPU_SWITCH_NUM, -1) == -1);
        set("OPENMX_BAND_GPU_SWITCH_NUM", "");     CHECK(openmx_env_long_at(OPENMX_ENV_BAND_GPU_SWITCH_NUM, -1) == -1);
        set("OPENMX_SETHAM_STREAM_MB", "300");
        CHECK(openmx_env_ulong_at(OPENMX_ENV_SETHAM_STREAM_MB, 0) == 300ULL);
        CHECK(openmx_env_mib_bytes_at(OPENMX_ENV_SETHAM_STREAM_MB, 0) == (size_t)300 << 20);
        CHECK(openmx_env_ulong("OPENMX_SETHAM_STREAM_MB", 0) == 300ULL);
        set("OPENMX_SETHAM_STREAM_MB", "18446744073709551615");
        CHECK(openmx_env_ulong_at(OPENMX_ENV_SETHAM_STREAM_MB, 0) == 18446744073709551615ULL);
        CHECK(openmx_env_mib_bytes_at(OPENMX_ENV_SETHAM_STREAM_MB, 0) == (size_t)-1);     /* saturates */
        set("OPENMX_SETHAM_STREAM_MB", "18446744073709551616");
        CHECK(openmx_env_ulong_at(OPENMX_ENV_SETHAM_STREAM_MB, 4) == 4);                 /* out of range */
        unsetenv("OPENMX_SETHAM_STREAM_MB");
        CHECK(openmx_env_mib_bytes_at(OPENMX_ENV_SETHAM_STREAM_MB, 64) == (size_t)64 << 20);
        set("OPENMX_EIGEN_REFINE_UNTIL", "2.5");  CHECK(openmx_env_double_at(OPENMX_ENV_EIGEN_REFINE_UNTIL, 0) == 2.5);
        CHECK(openmx_env_double("OPENMX_EIGEN_REFINE_UNTIL", 0) == 2.5);
        set("OPENMX_EIGEN_REFINE_UNTIL", "inf");  CHECK(openmx_env_double_at(OPENMX_ENV_EIGEN_REFINE_UNTIL, 3) == 3);
        set("OPENMX_EIGEN_REFINE_UNTIL", "nan");  CHECK(openmx_env_double_at(OPENMX_ENV_EIGEN_REFINE_UNTIL, 3) == 3);
        /* an unregistered name parses the same way */
        set("SOME_OTHER_VARIABLE", " on ");
        CHECK(openmx_env_flag("SOME_OTHER_VARIABLE", 0) == 1 && openmx_env_long("SOME_OTHER_VARIABLE", 4) == 4);
    }
    /* the snapshot: registered names freeze, unregistered ones stay live */
    set("OPENMX_GPU", "0");
    set("OPENMX_BAND_PROFILE", "1");
    set("OPENMX_SETHAM_STREAM_MB", "300");
    set("OPENMX_TYPO_SETTING", "1");
    set("OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT", "");
    set("OPENMX_EIGEN_REFINE_UNTIL", "abc");         /* not a double: reported */
    set("OPENMX_FORCE_PROFILE", "auto");             /* a flag spelling: not reported */
    unsetenv("OPENMX_CLUSTER_PROFILE");
    set("SOME_OTHER_VARIABLE", "x");
    openmx_env_init();
    CHECK(openmx_env_initialized());
    openmx_env_init();   /* idempotent */
    CHECK(same(openmx_env_at(OPENMX_ENV_GPU), "0") && same(openmx_env_get("OPENMX_GPU"), "0"));
    CHECK(openmx_env_flag_at(OPENMX_ENV_BAND_PROFILE, 0) == 1 && openmx_env_long_at(OPENMX_ENV_SETHAM_STREAM_MB, 0) == 300);
    CHECK(openmx_env_at(OPENMX_ENV_CLUSTER_PROFILE) == NULL && openmx_env_flag_at(OPENMX_ENV_CLUSTER_PROFILE, 7) == 7);
    set("OPENMX_GPU", "1");                              /* a later change of the environment ... */
    CHECK(same(openmx_env_at(OPENMX_ENV_GPU), "0"));     /* ... does not reach the readers */
    set("OPENMX_CLUSTER_PROFILE", "1");
    CHECK(openmx_env_flag_at(OPENMX_ENV_CLUSTER_PROFILE, 7) == 7);   /* unset at the snapshot stays unset */
    CHECK(same(openmx_env_get("SOME_OTHER_VARIABLE"), "x"));        /* not ours: live */
    set("SOME_OTHER_VARIABLE", "y");
    CHECK(same(openmx_env_get("SOME_OTHER_VARIABLE"), "y"));
    CHECK(same(openmx_env_get("OPENMX_TYPO_SETTING"), "1"));         /* unknown OPENMX_* names: snapshotted and reported */
    CHECK(openmx_env_unknown(&unknown) == 1 && same(unknown[0], "OPENMX_TYPO_SETTING"));
    CHECK(openmx_env_flag_at(OPENMX_ENV_GEMMUL8_MAX_WORKSPACE_PERCENT, 5) == 5);          /* empty: fallback */
    CHECK(openmx_env_double_at(OPENMX_ENV_EIGEN_REFINE_UNTIL, 1.5) == 1.5);
    CHECK(openmx_env_flag_at(OPENMX_ENV_FORCE_PROFILE, 3) == 3);                           /* auto: fallback */
    /* overrides: the snapshot and the process environment agree afterwards */
    CHECK(openmx_env_set_if_unset("OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT", "50") == 0);
    CHECK(same(openmx_env_at(OPENMX_ENV_GEMMUL8_MAX_WORKSPACE_PERCENT), ""));             /* set (empty) counts as set */
    CHECK(openmx_env_set_if_unset("OPENMX_CLUSTER_PROFILE", "2") == 0);
    CHECK(same(openmx_env_at(OPENMX_ENV_CLUSTER_PROFILE), "2") && same(getenv("OPENMX_CLUSTER_PROFILE"), "2"));
    CHECK(openmx_env_flag_at(OPENMX_ENV_CLUSTER_PROFILE, 0) == 1);                         /* the slot is re-parsed */
    CHECK(openmx_env_set("OPENMX_CLUSTER_PROFILE", "off") == 0 && openmx_env_flag_at(OPENMX_ENV_CLUSTER_PROFILE, 1) == 0);
    CHECK(openmx_env_unset("OPENMX_CLUSTER_PROFILE") == 0);
    CHECK(openmx_env_at(OPENMX_ENV_CLUSTER_PROFILE) == NULL && getenv("OPENMX_CLUSTER_PROFILE") == NULL);
    CHECK(openmx_env_set("OPENMX_NEW_AT_RUNTIME", "v") == 0 && same(openmx_env_get("OPENMX_NEW_AT_RUNTIME"), "v"));
    CHECK(openmx_env_unset("OPENMX_NEW_AT_RUNTIME") == 0 && openmx_env_get("OPENMX_NEW_AT_RUNTIME") == NULL);
    CHECK(openmx_env_set(NULL, "v") != 0 && openmx_env_set("X", NULL) != 0 && openmx_env_set("A=B", "v") != 0);
    CHECK(openmx_env_unset("") != 0);
    /* reports */
    {
        const char *text = report(1);
        CHECK(strstr(text, "OPENMX_TYPO_SETTING is set, but no setting") != NULL);
        CHECK(strstr(text, "OPENMX_EIGEN_REFINE_UNTIL=abc is not a valid ") != NULL);
        CHECK(strstr(text, "OPENMX_FORCE_PROFILE") == NULL);                                  /* auto is fine */
        text = report(0);
        CHECK(strstr(text, "OPENMX_BAND_PROFILE=1 (default 0)") != NULL && strstr(text, "OPENMX_GPU=0 (default 1)") != NULL);
        CHECK(strstr(text, "OPENMX_CLUSTER_PROFILE") == NULL && strstr(text, "SOME_OTHER_VARIABLE") == NULL);
    }
    /* rank 0 -> every rank: pack here, apply on a differently configured snapshot */
    length = openmx_env_pack(&packed);
    CHECK(packed != NULL && length > (size_t)OPENMX_ENV_COUNT);
    CHECK(openmx_env_unpack(packed, length) == 0);                   /* same values: nothing changes */
    openmx_env_reset();
    CHECK(!openmx_env_initialized());
    set("OPENMX_GPU", "1");                                           /* differs from rank 0 ("0") */
    unsetenv("OPENMX_BAND_PROFILE");                                  /* unset here, "1" on rank 0 */
    set("OPENMX_CLUSTER_PROFILE", "1");                               /* set here, unset on rank 0 */
    set("OMPI_COMM_WORLD_LOCAL_RANK", "3");                           /* local scope: never synchronized */
    set("OPENMX_TYPO_SETTING", "2");
    openmx_env_init();
    {
        int changed = openmx_env_unpack(packed, length);
        CHECK(changed == 3);
        CHECK(same(openmx_env_at(OPENMX_ENV_GPU), "0") && same(getenv("OPENMX_GPU"), "0"));
        CHECK(openmx_env_flag_at(OPENMX_ENV_BAND_PROFILE, 0) == 1 && same(getenv("OPENMX_BAND_PROFILE"), "1"));
        CHECK(openmx_env_at(OPENMX_ENV_CLUSTER_PROFILE) == NULL && getenv("OPENMX_CLUSTER_PROFILE") == NULL);
        CHECK(same(openmx_env_at(OPENMX_ENV_EXT_OMPI_COMM_WORLD_LOCAL_RANK), "3"));
        CHECK(same(openmx_env_get("OPENMX_TYPO_SETTING"), "2"));     /* unknown names stay local */
        CHECK(openmx_env_unpack(packed, length) == 0);               /* now identical */
        CHECK(openmx_env_unpack(packed, length - 1) == -1);          /* truncated */
        CHECK(openmx_env_unpack(NULL, length) == -1);
        packed[0] = 'X';
        CHECK(openmx_env_unpack(packed, length) == -1);              /* corrupt tag */
    }
    free(packed);
    /* HIP-only controls participate in the same snapshot and rank-0 wire
       format; launcher ranks remain local. These names were previously
       absent from the generated registry. */
    CHECK(same(entries[OPENMX_ENV_SETHAM_GPU].kind, "flag"));
    CHECK(same(entries[OPENMX_ENV_GPU_EIGENSOLVER].kind, "string"));
    CHECK(same(entries[OPENMX_ENV_SETPRO_HVNA_GPU].kind, "flag"));
    CHECK(same(entries[OPENMX_ENV_EXT_MPI_LOCALRANKID].scope, "local"));
    CHECK(openmx_env_set("OPENMX_SETHAM_GPU", "on") == 0);
    CHECK(openmx_env_set("OPENMX_GPU_EIGENSOLVER", "magma") == 0);
    CHECK(openmx_env_set("MAGMA_OZAKI2_MAX_WS_MB", "768") == 0);
    length = openmx_env_pack(&packed);
    openmx_env_reset();
    set("OPENMX_SETHAM_GPU", "off");
    set("OPENMX_GPU_EIGENSOLVER", "hipsolver");
    set("MAGMA_OZAKI2_MAX_WS_MB", "256");
    set("MPI_LOCALRANKID", "7");
    openmx_env_init();
    CHECK(openmx_env_unpack(packed, length) == 3);
    CHECK(openmx_env_flag_at(OPENMX_ENV_SETHAM_GPU, 0) == 1);
    CHECK(same(openmx_env_at(OPENMX_ENV_GPU_EIGENSOLVER), "magma"));
    CHECK(openmx_env_double_at(OPENMX_ENV_EXT_MAGMA_OZAKI2_MAX_WS_MB, 0) == 768);
    CHECK(same(getenv("OPENMX_GPU_EIGENSOLVER"), "magma"));
    CHECK(openmx_env_long_at(OPENMX_ENV_EXT_MPI_LOCALRANKID, -1) == 7);
    free(packed);
    /* reset: back to live reads */
    openmx_env_reset();
    CHECK(!openmx_env_initialized());
    set("OPENMX_GPU", "1");
    CHECK(same(openmx_env_at(OPENMX_ENV_GPU), "1"));
    CHECK(openmx_env_unknown(&unknown) == 0);
    openmx_env_init();
    CHECK(same(openmx_env_get("OPENMX_GPU"), "1"));
    openmx_env_reset();
    printf("PASS: environment registry (%d checks)\n", checks);
    return 0;
}
"""


def main():
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        print("SKIP: no C compiler", file=sys.stderr)
        return 0
    with tempfile.TemporaryDirectory(prefix="openmx-env-registry-") as directory:
        tmp = Path(directory)
        (tmp / "check.c").write_text(DRIVER)
        exe = tmp / "check"
        subprocess.run([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-O1", "-I", str(ROOT / "source"),
                        str(tmp / "check.c"), str(ROOT / "source" / "openmx_env.c"), "-o", str(exe)], check=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith("OPENMX_")}
        result = subprocess.run([str(exe)], env=env, text=True, capture_output=True)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode != 0:
            return result.returncode
    # the registry header must be what the readers in source/ imply
    check = subprocess.run([sys.executable, str(ROOT / "tools" / "update_env_registry.py"), "--check"],
                           text=True, capture_output=True)
    if check.returncode != 0:
        sys.stderr.write(check.stdout + check.stderr)
        print("FAIL: source/openmx_env_registry.h is out of step with the readers; run tools/update_env_registry.py", file=sys.stderr)
        return 1
    print("PASS: environment registry header in step with the readers")
    # the documentation tables must be what the registry and doc/environment_variables.json imply
    check = subprocess.run([sys.executable, str(ROOT / "tools" / "env_registry_doc.py"), "--check"],
                           text=True, capture_output=True)
    if check.returncode != 0:
        sys.stderr.write(check.stdout + check.stderr)
        print("FAIL: the environment-variable tables are out of date; run tools/env_registry_doc.py", file=sys.stderr)
        return 1
    print("PASS: environment-variable tables in step with the registry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
