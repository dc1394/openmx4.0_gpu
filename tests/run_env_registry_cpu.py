#!/usr/bin/env python3
"""Check the environment-settings registry (source/openmx_env.c) on the host.

Builds the production module with the system C compiler and a small driver
that exercises the snapshot, the live fallback, the typed readers, the
overrides that keep the process environment in sync, the detection of
unknown OPENMX_* names, the reports, and the registry's ordering.
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DRIVER = r'''
#define _POSIX_C_SOURCE 200809L   /* setenv, unsetenv */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openmx_env.h"
static int checks;
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
static int same(const char *a, const char *b) { return a != NULL && b != NULL && strcmp(a, b) == 0; }
int main(void) {
    const OpenMXEnvEntry *entries;
    size_t count = openmx_env_registry(&entries), i;
    const char *const *unknown;
    /* registry: sorted, unique, every field present, a known name resolves */
    CHECK(count > 100);
    for (i = 0; i < count; i++) {
        CHECK(entries[i].name && entries[i].kind && entries[i].module && entries[i].doc);
        if (i) CHECK(strcmp(entries[i-1].name, entries[i].name) < 0);
        CHECK(openmx_env_lookup(entries[i].name) == &entries[i]);
    }
    CHECK(openmx_env_lookup("OPENMX_GPU") != NULL);
    CHECK(openmx_env_lookup("OPENMX_NO_SUCH_SETTING") == NULL);
    /* before the snapshot: live, like getenv */
    CHECK(!openmx_env_initialized());
    unsetenv("OPENMX_GPU");
    CHECK(openmx_env_get("OPENMX_GPU") == NULL);
    setenv("OPENMX_GPU", "0", 1);
    CHECK(same(openmx_env_get("OPENMX_GPU"), "0"));
    CHECK(openmx_env_flag("OPENMX_GPU", 1) == 0);
    /* the snapshot: registered names freeze, unregistered ones stay live */
    setenv("OPENMX_FORCE_PROFILE", "1", 1);
    setenv("OPENMX_SETHAM_STREAM_MB", "300", 1);
    setenv("OPENMX_TYPO_SETTING", "1", 1);
    setenv("OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT", "", 1);
    unsetenv("OPENMX_BAND_PROFILE");
    setenv("SOME_OTHER_VARIABLE", "x", 1);
    openmx_env_init();
    CHECK(openmx_env_initialized());
    openmx_env_init();   /* idempotent */
    CHECK(same(openmx_env_get("OPENMX_GPU"), "0"));
    CHECK(same(openmx_env_get("OPENMX_FORCE_PROFILE"), "1"));
    CHECK(openmx_env_get("OPENMX_BAND_PROFILE") == NULL);
    setenv("OPENMX_GPU", "1", 1);                       /* a later change of the environment ... */
    CHECK(same(openmx_env_get("OPENMX_GPU"), "0"));       /* ... does not reach the readers */
    setenv("OPENMX_BAND_PROFILE", "1", 1);
    CHECK(openmx_env_get("OPENMX_BAND_PROFILE") == NULL);  /* unset at the snapshot stays unset */
    CHECK(same(openmx_env_get("SOME_OTHER_VARIABLE"), "x"));  /* not ours: live */
    setenv("SOME_OTHER_VARIABLE", "y", 1);
    CHECK(same(openmx_env_get("SOME_OTHER_VARIABLE"), "y"));
    CHECK(same(openmx_env_get("OPENMX_TYPO_SETTING"), "1"));  /* unknown OPENMX_* names are snapshotted and reported */
    CHECK(openmx_env_unknown(&unknown) == 1 && same(unknown[0], "OPENMX_TYPO_SETTING"));
    /* typed readers */
    CHECK(openmx_env_flag("OPENMX_FORCE_PROFILE", 0) == 1);
    CHECK(openmx_env_flag("OPENMX_BAND_PROFILE", 7) == 7);
    CHECK(openmx_env_flag("OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT", 5) == 5);   /* empty: fallback */
    CHECK(openmx_env_long("OPENMX_SETHAM_STREAM_MB", -1) == 300);
    CHECK(openmx_env_long("OPENMX_FORCE_PROFILE", -1) == 1);
    CHECK(openmx_env_long("OPENMX_BAND_PROFILE", -1) == -1);
    CHECK(openmx_env_mib("OPENMX_SETHAM_STREAM_MB", 0) == 300ULL);
    CHECK(openmx_env_double("OPENMX_SETHAM_STREAM_MB", 0.0) == 300.0);
    /* overrides: the snapshot and the process environment agree afterwards */
    CHECK(openmx_env_set_if_unset("OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT", "50") == 0);
    CHECK(same(openmx_env_get("OPENMX_GEMMUL8_MAX_WORKSPACE_PERCENT"), ""));   /* set (empty) counts as set */
    CHECK(openmx_env_set_if_unset("OPENMX_BAND_PROFILE", "2") == 0);
    CHECK(same(openmx_env_get("OPENMX_BAND_PROFILE"), "2"));
    CHECK(same(getenv("OPENMX_BAND_PROFILE"), "2"));
    CHECK(openmx_env_set("OPENMX_BAND_PROFILE", "3") == 0);
    CHECK(same(openmx_env_get("OPENMX_BAND_PROFILE"), "3") && same(getenv("OPENMX_BAND_PROFILE"), "3"));
    CHECK(openmx_env_set("OPENMX_NEW_AT_RUNTIME", "v") == 0);
    CHECK(same(openmx_env_get("OPENMX_NEW_AT_RUNTIME"), "v"));
    CHECK(openmx_env_set(NULL, "v") != 0 && openmx_env_set("X", NULL) != 0);
    /* reports */
    {
        FILE *fp = tmpfile();
        long n;
        char buffer[4096];
        CHECK(fp != NULL);
        openmx_env_report(fp, 1);
        openmx_env_report(fp, 0);
        n = ftell(fp);
        CHECK(n > 0 && n < (long)sizeof buffer);
        rewind(fp);
        buffer[fread(buffer, 1, sizeof buffer - 1, fp)] = '\0';
        CHECK(strstr(buffer, "OPENMX_TYPO_SETTING is set, but no setting") != NULL);
        CHECK(strstr(buffer, "OPENMX_FORCE_PROFILE=1") != NULL);
        CHECK(strstr(buffer, "OPENMX_GPU=0") != NULL);
        CHECK(strstr(buffer, "SOME_OTHER_VARIABLE") == NULL);
        fclose(fp);
    }
    /* reset: back to live reads */
    openmx_env_reset();
    CHECK(!openmx_env_initialized());
    CHECK(same(openmx_env_get("OPENMX_GPU"), "1"));
    CHECK(openmx_env_unknown(&unknown) == 0);
    openmx_env_init();
    CHECK(same(openmx_env_get("OPENMX_GPU"), "1"));
    openmx_env_reset();
    printf("PASS: environment registry (%d checks)\n", checks);
    return 0;
}
'''


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
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
