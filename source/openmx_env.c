#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L   /* setenv, unsetenv */
#endif
/**********************************************************************
  openmx_env.c

  The environment settings of this build, read once.  See openmx_env.h;
  the names are in openmx_env_registry.h.
***********************************************************************/
#include "openmx_env.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

extern char **environ;

/* --- the registry table, in id order (sorted by name) ------------------- */
#define OPENMX_ENV_ENTRY_(id, name, kind, scope, dflt, module, doc) {name, kind, scope, dflt, module, doc},
static const OpenMXEnvEntry registry[] = { OPENMX_ENV_LIST(OPENMX_ENV_ENTRY_) };
#undef OPENMX_ENV_ENTRY_
#define REGISTRY_COUNT ((int)(sizeof registry / sizeof registry[0]))
typedef char openmx_env_count_matches_enum[(sizeof registry / sizeof registry[0]) == OPENMX_ENV_COUNT ? 1 : -1];

/* --- the snapshot --------------------------------------------------------- */
typedef struct {
    char  *value;                 /* NULL: unset */
    int    flag;                  /* -1: unset, empty or not a flag */
    int    has_long, has_ulong, has_double;
    long   long_value;
    unsigned long long ulong_value;
    double double_value;
} Slot;

typedef struct {                  /* a name the registry does not list */
    char *name;
    char *value;
} Extra;

static struct {
    Slot   slots[OPENMX_ENV_COUNT];
    Extra *extra;                 /* sorted by name */
    size_t extra_count, extra_capacity;
    char **unknown;               /* OPENMX_* names of the environment that the registry lacks */
    size_t unknown_count, unknown_capacity;
    int    initialized;
} state;

static char *dup_string(const char *s)
{
    size_t n;
    char *copy;

    if (s == NULL) return NULL;
    n = strlen(s) + 1;
    copy = (char *)malloc(n);
    if (copy != NULL) memcpy(copy, s, n);
    return copy;
}

/* --- the parsers ----------------------------------------------------------- */
#define NUMBER_MAX 63

/* the value without surrounding blanks, in buffer; 0 when empty or too long */
static int trimmed(const char *value, char *buffer)
{
    size_t n;

    while (*value == ' ' || *value == '\t') value++;
    n = strlen(value);
    while (n > 0 && (value[n - 1] == ' ' || value[n - 1] == '\t' || value[n - 1] == '\n' || value[n - 1] == '\r')) n--;
    if (n == 0 || n > NUMBER_MAX) return 0;
    memcpy(buffer, value, n);
    buffer[n] = '\0';
    return 1;
}

static int parse_long(const char *value, long *out)
{
    char buffer[NUMBER_MAX + 1], *end;
    long x;

    if (!trimmed(value, buffer)) return 0;
    errno = 0;
    x = strtol(buffer, &end, 10);
    if (end == buffer || *end != '\0' || errno == ERANGE) return 0;
    *out = x;
    return 1;
}

static int parse_ulong(const char *value, unsigned long long *out)
{
    char buffer[NUMBER_MAX + 1], *end;
    unsigned long long x;

    if (!trimmed(value, buffer) || buffer[0] == '-' || (buffer[0] == '+' && buffer[1] == '-')) return 0;
    errno = 0;
    x = strtoull(buffer, &end, 10);
    if (end == buffer || *end != '\0' || errno == ERANGE) return 0;
    *out = x;
    return 1;
}

static int parse_double(const char *value, double *out)
{
    char buffer[NUMBER_MAX + 1], *end;
    double x;

    if (!trimmed(value, buffer)) return 0;
    errno = 0;
    x = strtod(buffer, &end);
    if (end == buffer || *end != '\0' || errno == ERANGE || !isfinite(x)) return 0;
    *out = x;
    return 1;
}

static int word(const char *a, const char *b)
{
    for (; *a != '\0' && *b != '\0'; a++, b++) {
        int ca = *a, cb = *b;

        if ('A' <= ca && ca <= 'Z') ca += 'a' - 'A';
        if (ca != cb) return 0;
    }
    return *a == '\0' && *b == '\0';
}

/* 1, 0, or -1 when the value is not a flag spelling ("auto" included) */
static int parse_flag(const char *value)
{
    char buffer[NUMBER_MAX + 1];
    double x;

    if (!trimmed(value, buffer)) return -1;
    if (parse_double(buffer, &x)) return x != 0.0;
    if (word(buffer, "on") || word(buffer, "yes") || word(buffer, "true") || word(buffer, "y")) return 1;
    if (word(buffer, "off") || word(buffer, "no") || word(buffer, "false") || word(buffer, "n")) return 0;
    return -1;
}

static int flag_spelling(const char *value)
{
    char buffer[NUMBER_MAX + 1];

    return parse_flag(value) >= 0 || (trimmed(value, buffer) && word(buffer, "auto"));
}

static int empty(const char *value)
{
    char buffer[NUMBER_MAX + 1];

    return value == NULL || value[0] == '\0' || (strlen(value) <= NUMBER_MAX && !trimmed(value, buffer));
}

/* --- slots ------------------------------------------------------------------ */
static void slot_parse(Slot *s)
{
    s->flag = -1;
    s->has_long = s->has_ulong = s->has_double = 0;
    if (empty(s->value)) return;
    s->flag = parse_flag(s->value);
    s->has_long = parse_long(s->value, &s->long_value);
    s->has_ulong = parse_ulong(s->value, &s->ulong_value);
    s->has_double = parse_double(s->value, &s->double_value);
}

static int slot_set(Slot *s, const char *value)
{
    char *copy = dup_string(value);

    if (value != NULL && copy == NULL) return -1;
    free(s->value);
    s->value = copy;
    slot_parse(s);
    return 0;
}

static int valid_id(int id)
{
    return 0 <= id && id < REGISTRY_COUNT;
}

/* --- unregistered names ------------------------------------------------------ */
static int extra_compare(const void *a, const void *b)
{
    return strcmp(((const Extra *)a)->name, ((const Extra *)b)->name);
}

static Extra *find_extra(const char *name)
{
    Extra key;

    if (state.extra_count == 0) return NULL;
    key.name = (char *)name;
    key.value = NULL;
    return (Extra *)bsearch(&key, state.extra, state.extra_count, sizeof *state.extra, extra_compare);
}

/* append unsorted; the caller sorts */
static int append_extra(const char *name, const char *value)
{
    Extra *e;

    if (state.extra_count == state.extra_capacity) {
        size_t capacity = state.extra_capacity ? 2 * state.extra_capacity : 16;
        Extra *grown = (Extra *)realloc(state.extra, capacity * sizeof *grown);

        if (grown == NULL) return -1;
        state.extra = grown;
        state.extra_capacity = capacity;
    }
    e = &state.extra[state.extra_count];
    e->name = dup_string(name);
    e->value = dup_string(value);
    if (e->name == NULL || (value != NULL && e->value == NULL)) {
        free(e->name);
        free(e->value);
        return -1;
    }
    state.extra_count++;
    return 0;
}

static int remember_unknown(const char *name)
{
    char *copy;

    if (state.unknown_count == state.unknown_capacity) {
        size_t capacity = state.unknown_capacity ? 2 * state.unknown_capacity : 8;
        char **grown = (char **)realloc(state.unknown, capacity * sizeof *grown);

        if (grown == NULL) return -1;
        state.unknown = grown;
        state.unknown_capacity = capacity;
    }
    copy = dup_string(name);
    if (copy == NULL) return -1;
    state.unknown[state.unknown_count++] = copy;
    return 0;
}

/* --- the registry ------------------------------------------------------------ */
static int registry_compare(const void *a, const void *b)
{
    return strcmp((const char *)a, ((const OpenMXEnvEntry *)b)->name);
}

const OpenMXEnvEntry *openmx_env_lookup(const char *name)
{
    if (name == NULL) return NULL;
    return (const OpenMXEnvEntry *)bsearch(name, registry, (size_t)REGISTRY_COUNT, sizeof registry[0], registry_compare);
}

int openmx_env_id(const char *name)
{
    const OpenMXEnvEntry *entry = openmx_env_lookup(name);

    return entry != NULL ? (int)(entry - registry) : -1;
}

size_t openmx_env_registry(const OpenMXEnvEntry **entries)
{
    if (entries != NULL) *entries = registry;
    return (size_t)REGISTRY_COUNT;
}

const char *openmx_env_name(int id)
{
    return valid_id(id) ? registry[id].name : "";
}

/* --- the snapshot ------------------------------------------------------------- */
void openmx_env_init(void)
{
    int id;
    char **e;

    if (state.initialized) return;
    for (id = 0; id < REGISTRY_COUNT; id++) {
        (void)slot_set(&state.slots[id], getenv(registry[id].name));
    }
    /* every OPENMX_* present in the environment: snapshotted too, and
       reported when the registry does not know it (a misspelt setting) */
    for (e = environ; e != NULL && *e != NULL; e++) {
        const char *equals = strchr(*e, '=');
        size_t n;
        char name[256];

        if (strncmp(*e, "OPENMX_", 7) != 0 || equals == NULL) continue;
        n = (size_t)(equals - *e);
        if (n == 0 || n >= sizeof name) continue;
        memcpy(name, *e, n);
        name[n] = '\0';
        if (openmx_env_lookup(name) != NULL) continue;
        (void)remember_unknown(name);
        (void)append_extra(name, equals + 1);
    }
    qsort(state.extra, state.extra_count, sizeof *state.extra, extra_compare);
    state.initialized = 1;
}

int openmx_env_initialized(void)
{
    return state.initialized;
}

void openmx_env_reset(void)
{
    size_t i;
    int id;

    for (id = 0; id < REGISTRY_COUNT; id++) free(state.slots[id].value);
    for (i = 0; i < state.extra_count; i++) {
        free(state.extra[i].name);
        free(state.extra[i].value);
    }
    free(state.extra);
    for (i = 0; i < state.unknown_count; i++) free(state.unknown[i]);
    free(state.unknown);
    memset(&state, 0, sizeof state);
}

/* --- readers by id -------------------------------------------------------------- */
const char *openmx_env_at(int id)
{
    if (!valid_id(id)) return NULL;
    if (!state.initialized) return getenv(registry[id].name);
    return state.slots[id].value;
}

static int flag_of(const char *value, int fallback)
{
    int f;

    if (empty(value)) return fallback;
    f = parse_flag(value);
    return f < 0 ? fallback : f;
}

static long long_of(const char *value, long fallback)
{
    long x;

    return !empty(value) && parse_long(value, &x) ? x : fallback;
}

static unsigned long long ulong_of(const char *value, unsigned long long fallback)
{
    unsigned long long x;

    return !empty(value) && parse_ulong(value, &x) ? x : fallback;
}

static double double_of(const char *value, double fallback)
{
    double x;

    return !empty(value) && parse_double(value, &x) ? x : fallback;
}

int openmx_env_flag_at(int id, int fallback)
{
    if (!valid_id(id)) return fallback;
    if (!state.initialized) return flag_of(getenv(registry[id].name), fallback);
    return state.slots[id].flag < 0 ? fallback : state.slots[id].flag;
}

long openmx_env_long_at(int id, long fallback)
{
    if (!valid_id(id)) return fallback;
    if (!state.initialized) return long_of(getenv(registry[id].name), fallback);
    return state.slots[id].has_long ? state.slots[id].long_value : fallback;
}

unsigned long long openmx_env_ulong_at(int id, unsigned long long fallback)
{
    if (!valid_id(id)) return fallback;
    if (!state.initialized) return ulong_of(getenv(registry[id].name), fallback);
    return state.slots[id].has_ulong ? state.slots[id].ulong_value : fallback;
}

double openmx_env_double_at(int id, double fallback)
{
    if (!valid_id(id)) return fallback;
    if (!state.initialized) return double_of(getenv(registry[id].name), fallback);
    return state.slots[id].has_double ? state.slots[id].double_value : fallback;
}

size_t openmx_env_mib_bytes_at(int id, unsigned long long fallback_mib)
{
    unsigned long long mib = openmx_env_ulong_at(id, fallback_mib);
    const unsigned long long limit = (unsigned long long)((size_t)-1) >> 20;

    return mib > limit ? (size_t)-1 : (size_t)mib << 20;
}

/* --- readers by name ------------------------------------------------------------ */
char *openmx_env_get(const char *name)
{
    int id;
    Extra *extra;

    if (name == NULL) return NULL;
    if (!state.initialized) return getenv(name);
    id = openmx_env_id(name);
    if (id >= 0) return state.slots[id].value;
    extra = find_extra(name);
    return extra != NULL ? extra->value : getenv(name);
}

int openmx_env_flag(const char *name, int fallback)
{
    int id = openmx_env_id(name);

    return id >= 0 ? openmx_env_flag_at(id, fallback) : flag_of(openmx_env_get(name), fallback);
}

long openmx_env_long(const char *name, long fallback)
{
    int id = openmx_env_id(name);

    return id >= 0 ? openmx_env_long_at(id, fallback) : long_of(openmx_env_get(name), fallback);
}

unsigned long long openmx_env_ulong(const char *name, unsigned long long fallback)
{
    int id = openmx_env_id(name);

    return id >= 0 ? openmx_env_ulong_at(id, fallback) : ulong_of(openmx_env_get(name), fallback);
}

double openmx_env_double(const char *name, double fallback)
{
    int id = openmx_env_id(name);

    return id >= 0 ? openmx_env_double_at(id, fallback) : double_of(openmx_env_get(name), fallback);
}

/* --- overrides ------------------------------------------------------------------- */
static int store(const char *name, const char *value)
{
    int id;
    Extra *extra;

    if (!state.initialized) return 0;
    id = openmx_env_id(name);
    if (id >= 0) return slot_set(&state.slots[id], value);
    extra = find_extra(name);
    if (extra != NULL) {
        char *copy = dup_string(value);

        if (value != NULL && copy == NULL) return -1;
        free(extra->value);
        extra->value = copy;
        return 0;
    }
    if (value == NULL) return 0;
    if (append_extra(name, value) != 0) return -1;
    qsort(state.extra, state.extra_count, sizeof *state.extra, extra_compare);
    return 0;
}

int openmx_env_set(const char *name, const char *value)
{
    if (name == NULL || value == NULL || name[0] == '\0' || strchr(name, '=') != NULL) return -1;
    if (setenv(name, value, 1) != 0) return -1;
    return store(name, value);
}

int openmx_env_set_if_unset(const char *name, const char *value)
{
    if (openmx_env_get(name) != NULL) return 0;
    return openmx_env_set(name, value);
}

int openmx_env_unset(const char *name)
{
    if (name == NULL || name[0] == '\0' || strchr(name, '=') != NULL) return -1;
    if (unsetenv(name) != 0) return -1;
    return store(name, NULL);
}

/* --- rank 0 -> every rank ---------------------------------------------------------- */
/* one record per registry entry, in id order: 'L' (local, no value),
   'U' (unset) or 'S' followed by the value and its terminator */
static int synchronized(int id)
{
    return strcmp(registry[id].scope, "global") == 0;
}

size_t openmx_env_pack(char **buffer)
{
    size_t length = 0, at = 0;
    int id;
    char *out;

    if (buffer == NULL) return 0;
    *buffer = NULL;
    for (id = 0; id < REGISTRY_COUNT; id++) {
        const char *value = synchronized(id) ? openmx_env_at(id) : NULL;

        length += 1 + (value != NULL ? strlen(value) + 1 : 0);
    }
    out = (char *)malloc(length);
    if (out == NULL) return 0;
    for (id = 0; id < REGISTRY_COUNT; id++) {
        const char *value = synchronized(id) ? openmx_env_at(id) : NULL;

        if (!synchronized(id)) {
            out[at++] = 'L';
        } else if (value == NULL) {
            out[at++] = 'U';
        } else {
            size_t n = strlen(value) + 1;

            out[at++] = 'S';
            memcpy(out + at, value, n);
            at += n;
        }
    }
    *buffer = out;
    return length;
}

int openmx_env_unpack(const char *buffer, size_t length)
{
    size_t at = 0;
    int id, changed = 0;

    if (buffer == NULL) return -1;
    for (id = 0; id < REGISTRY_COUNT; id++) {
        const char *mine;
        char tag;

        if (at >= length) return -1;
        tag = buffer[at++];
        if (tag == 'L') continue;
        if (!synchronized(id)) return -1;
        mine = openmx_env_at(id);
        if (tag == 'U') {
            if (mine != NULL) {
                if (openmx_env_unset(registry[id].name) != 0) return -1;
                changed++;
            }
        } else if (tag == 'S') {
            const char *value = buffer + at;
            const char *end = (const char *)memchr(value, '\0', length - at);

            if (end == NULL) return -1;
            at = (size_t)(end - buffer) + 1;
            if (mine == NULL || strcmp(mine, value) != 0) {
                if (openmx_env_set(registry[id].name, value) != 0) return -1;
                changed++;
            }
        } else {
            return -1;
        }
    }
    return at == length ? changed : -1;
}

/* --- diagnostics --------------------------------------------------------------------- */
size_t openmx_env_unknown(const char *const **names)
{
    if (names != NULL) *names = (const char *const *)state.unknown;
    return state.unknown_count;
}

/* 0 when the value does not parse for the registered kind */
static int value_fits_kind(int id)
{
    const Slot *s = &state.slots[id];
    const char *kind = registry[id].kind;

    if (empty(s->value)) return 1;
    if (strcmp(kind, "flag") == 0) return flag_spelling(s->value);
    if (strcmp(kind, "int") == 0) return s->has_long;
    if (strcmp(kind, "uint") == 0 || strcmp(kind, "mib") == 0) return s->has_ulong;
    if (strcmp(kind, "double") == 0) return s->has_double;
    return 1;
}

void openmx_env_report(FILE *out, int unknown_only)
{
    size_t i;
    int id, printed = 0;

    if (out == NULL) return;
    if (unknown_only) {
        for (i = 0; i < state.unknown_count; i++) {
            fprintf(out, "<OpenMX> environment: %s is set, but no setting of this build has that name"
                         " (check the spelling; section 10.2 of doc/OpenMX40_GPU_implementation_en.html lists them).\n",
                    state.unknown[i]);
        }
        if (state.initialized) {
            for (id = 0; id < REGISTRY_COUNT; id++) {
                if (value_fits_kind(id)) continue;
                fprintf(out, "<OpenMX> environment: %s=%s is not a valid %s value; the default applies.\n",
                        registry[id].name, state.slots[id].value, registry[id].kind);
            }
        }
        fflush(out);
        return;
    }
    for (id = 0; id < REGISTRY_COUNT; id++) {
        const char *value = openmx_env_at(id);

        if (value == NULL) continue;
        if (!printed) {
            fprintf(out, "<OpenMX> environment settings in effect:\n");
            printed = 1;
        }
        if (registry[id].dflt[0] != '\0')
            fprintf(out, "  %s=%s (default %s)\n", registry[id].name, value, registry[id].dflt);
        else
            fprintf(out, "  %s=%s\n", registry[id].name, value);
    }
    if (printed) fflush(out);
}
