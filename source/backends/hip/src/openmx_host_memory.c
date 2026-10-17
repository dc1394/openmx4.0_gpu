/**********************************************************************
  openmx_host_memory.c

  Host memory of this node as a batch job sees it; see
  openmx_host_memory.h.
***********************************************************************/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "openmx_host_memory.h"

#include <stdio.h>
#include <string.h>
#include <limits.h>

static int read_ull_file(const char *path, unsigned long long *value)
{
    FILE *fp = fopen(path, "r");
    int ok;

    if (fp == NULL) return 0;
    ok = (fscanf(fp, "%llu", value) == 1); /* "max" (no limit) fails */
    fclose(fp);
    return ok;
}

/* the inactive file cache charged to a cgroup (memory.stat), which the
   kernel reclaims before it would kill: usage minus it is what the job
   really holds (the "working set"); 0 when unreadable */
static unsigned long long cgroup_inactive_file(const char *stat_path, int v1)
{
    FILE *fp = fopen(stat_path, "r");
    char line[256], key[64];
    unsigned long long value, found = 0;

    if (fp == NULL) return 0;
    while (fgets(line, sizeof(line), fp) != NULL) {
        if (sscanf(line, "%63s %llu", key, &value) != 2) continue;
        /* v1: the hierarchical total; v2: the cgroup's own counters are
           already hierarchical */
        if (strcmp(key, v1 ? "total_inactive_file" : "inactive_file") == 0) {
            found = value;
            break;
        }
    }
    fclose(fp);
    return found;
}

/* the smallest memory limit of this process's cgroup and its parents and
   the room left under it (limit - usage, with the reclaimable inactive
   file cache not counted as used), cgroup v1 or v2; ULLONG_MAX when there
   is no readable limit */
static void cgroup_mem_limits(unsigned long long *limit_min, unsigned long long *room_min)
{
    FILE *fp = fopen("/proc/self/cgroup", "r");
    char line[1024], path[1024] = "", file[1400], stat[1400];
    unsigned long long limit, usage, cache;
    int v1 = 0;

    *limit_min = ULLONG_MAX;
    *room_min = ULLONG_MAX;
    if (fp == NULL) return;
    while (fgets(line, sizeof(line), fp) != NULL) {
        char *a = strchr(line, ':'), *b = a ? strchr(a + 1, ':') : NULL, *nl;
        if (b == NULL) continue;
        nl = strchr(b + 1, '\n');
        if (nl) *nl = '\0';
        *b = '\0';
        if (strcmp(a + 1, "") == 0 && path[0] == '\0') {
            snprintf(path, sizeof(path), "%s", b + 1);          /* v2: "0::/path" */
        }
        else {
            for (const char *tok=a+1; *tok;) {
                const char *end=strchr(tok,',');
                size_t len=end?(size_t)(end-tok):strlen(tok);
                if (len==6 && memcmp(tok,"memory",6)==0) {
                    snprintf(path,sizeof(path),"%s",b+1);
                    v1=1;
                }
                if (!end) break;
                tok=end+1;
            }
        }
    }
    fclose(fp);
    if (path[0] == '\0') return;

    for (;;) {
        int got;
        if (v1) {
            snprintf(file, sizeof(file), "/sys/fs/cgroup/memory%s/memory.limit_in_bytes", path);
            got = read_ull_file(file, &limit);
            snprintf(file, sizeof(file), "/sys/fs/cgroup/memory%s/memory.usage_in_bytes", path);
            snprintf(stat, sizeof(stat), "/sys/fs/cgroup/memory%s/memory.stat", path);
        }
        else {
            snprintf(file, sizeof(file), "/sys/fs/cgroup%s/memory.max", path);
            got = read_ull_file(file, &limit);
            snprintf(file, sizeof(file), "/sys/fs/cgroup%s/memory.current", path);
            snprintf(stat, sizeof(stat), "/sys/fs/cgroup%s/memory.stat", path);
        }
        if (got && limit < (1ULL << 60)) {
            if (limit < *limit_min) *limit_min = limit;
            if (read_ull_file(file, &usage)) {
                unsigned long long left;
                cache = cgroup_inactive_file(stat, v1);
                usage = (cache < usage) ? usage - cache : 0;
                left = (usage < limit) ? limit - usage : 0;
                if (left < *room_min) *room_min = left;
            }
        }
        {
            char *slash = strrchr(path, '/');
            if (slash == NULL || slash == path) {
                if (strcmp(path, "/") == 0 || path[0] == '\0') break;
                strcpy(path, "/");
            }
            else {
                *slash = '\0';
            }
        }
    }
}

/* Distinguish an unreadable counter from a known zero: the latter must
   reject allocations, not disable admission as an unknown measurement. */
static int meminfo_bytes(const char *name, unsigned long long *bytes)
{
    FILE *fp = fopen("/proc/meminfo", "r");
    char line[256], format[64];
    unsigned long long kb = 0;
    int found = 0;

    if (fp == NULL) return 0;
    snprintf(format, sizeof(format), "%s: %%llu kB", name);
    while (fgets(line, sizeof(line), fp) != NULL) {
        if (sscanf(line, format, &kb) == 1) { found = 1; break; }
    }
    fclose(fp);
    if (found) *bytes = kb > ULLONG_MAX / 1024ULL ? ULLONG_MAX : kb * 1024ULL;
    return found;
}

unsigned long long openmx_host_mem_available(void)
{
    unsigned long long bytes = 0, limit, room;
    int known = meminfo_bytes("MemAvailable", &bytes);

    cgroup_mem_limits(&limit, &room);
    /* Even if usage is unreadable, the available bytes cannot exceed the
       job's total limit. Conversely a cgroup can provide a useful bound
       when /proc/meminfo has no MemAvailable counter. */
    if (limit < room) room = limit;
    if (room != ULLONG_MAX && (!known || room < bytes)) { bytes = room; known = 1; }
    if (!known) return 0;
    return (bytes > 0) ? bytes : 1;  /* known, but nothing left */
}

unsigned long long openmx_host_mem_total(void)
{
    unsigned long long bytes = ULLONG_MAX, limit, room;

    (void)meminfo_bytes("MemTotal", &bytes);
    cgroup_mem_limits(&limit, &room);
    if (limit < bytes) bytes = limit;
    return bytes == ULLONG_MAX ? 0 : bytes;
}
