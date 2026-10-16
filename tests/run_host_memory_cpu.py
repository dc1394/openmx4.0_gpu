#!/usr/bin/env python3
"""Check production host-memory admission against deterministic proc/cgroup files."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DRIVER = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct { const char *path, *text; } File;
static File files[24];
static size_t count;
static int checks;
static FILE *fixture_open(const char *path, const char *mode) {
    (void)mode;
    for (size_t i = 0; i < count; ++i) if (strcmp(files[i].path, path) == 0) {
        FILE *fp = tmpfile();
        if (!fp) abort();
        fputs(files[i].text, fp); rewind(fp); return fp;
    }
    return NULL;
}
#define fopen fixture_open
#include "openmx_host_memory.c"
#undef fopen
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
static void add(const char *path, const char *text) {
    files[count++] = (File){path, text};
}
static void fixture(const char *meminfo, int v1, const char *limit, const char *usage, const char *cache) {
    count = 0;
    if (meminfo) add("/proc/meminfo", meminfo);
    if (v1) {
        add("/proc/self/cgroup", "3:cpuset:/job\n7:cpu,memory:/job\n0::/ignored\n");
        if (limit) add("/sys/fs/cgroup/memory/job/memory.limit_in_bytes", limit);
        if (usage) add("/sys/fs/cgroup/memory/job/memory.usage_in_bytes", usage);
        if (cache) add("/sys/fs/cgroup/memory/job/memory.stat", cache);
    } else {
        add("/proc/self/cgroup", "0::/job\n");
        if (limit) add("/sys/fs/cgroup/job/memory.max", limit);
        if (usage) add("/sys/fs/cgroup/job/memory.current", usage);
        if (cache) add("/sys/fs/cgroup/job/memory.stat", cache);
    }
}
int main(void) {
    const char *mem = "MemTotal: 16 kB\nMemFree: 2 kB\nMemAvailable: 8 kB\n";
    for (int v1 = 0; v1 <= 1; ++v1) {
        fixture(mem, v1, "4096\n", "2048\n", NULL);
        CHECK(openmx_host_mem_total() == 4096);
        CHECK(openmx_host_mem_available() == 2048);
        fixture(mem, v1, "4096\n", "4096\n", v1 ? "total_inactive_file 1024\n" : "inactive_file 1024\n");
        CHECK(openmx_host_mem_available() == 1024);
        fixture(mem, v1, "4096\n", "8192\n", NULL);
        CHECK(openmx_host_mem_available() == 1); /* known exhaustion, not unknown */
        fixture(mem, v1, "4096\n", NULL, NULL);
        CHECK(openmx_host_mem_available() <= 4096); /* never exceed a readable limit */
        fixture("MemTotal: 16 kB\nMemAvailable: 0 kB\n", v1, NULL, NULL, NULL);
        CHECK(openmx_host_mem_available() == 1);
        fixture("MemTotal: 16 kB\n", v1, "4096\n", "1024\n", NULL);
        CHECK(openmx_host_mem_available() == 3072); /* cgroup information is still usable */
        fixture(NULL, v1, "4096\n", "1024\n", NULL);
        CHECK(openmx_host_mem_available() == 3072);
        CHECK(openmx_host_mem_total() == 4096);
        fixture(mem, v1, "4096\n", "1024\n", NULL);
        if (v1) {
            add("/sys/fs/cgroup/memory//memory.limit_in_bytes", "2048\n");
            add("/sys/fs/cgroup/memory//memory.usage_in_bytes", "1536\n");
        } else {
            add("/sys/fs/cgroup//memory.max", "2048\n");
            add("/sys/fs/cgroup//memory.current", "1536\n");
        }
        CHECK(openmx_host_mem_total() == 2048);
        CHECK(openmx_host_mem_available() == 512); /* parent usage includes other jobs */
    }
    fixture(mem, 0, "max\n", "4096\n", NULL);
    CHECK(openmx_host_mem_available() == 8192);
    CHECK(openmx_host_mem_total() == 16384);
    fixture(mem, 1, "9223372036854771712\n", "4096\n", NULL);
    CHECK(openmx_host_mem_available() == 8192);
    count = 0;
    CHECK(openmx_host_mem_available() == 0);
    CHECK(openmx_host_mem_total() == 0);
    printf("PASS: host memory, cgroup v1/v2, limits, cache, zero and missing counters (%d checks)\n", checks);
    return 0;
}
'''

with tempfile.TemporaryDirectory(prefix="openmx-host-memory-") as directory:
    tmp = Path(directory)
    (tmp / "check.c").write_text(DRIVER)
    exe = tmp / "check"
    subprocess.run(shlex.split(os.environ.get("CC", "cc")) + [
        "-std=gnu11", "-O1", "-g", "-Wall", "-Wextra", "-fsanitize=address,undefined",
        "-fno-sanitize-recover=all", "-I" + str(ROOT / "source"), str(tmp / "check.c"),
        "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
