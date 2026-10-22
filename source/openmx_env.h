/**********************************************************************
  openmx_env.h

  The environment settings of this build, read once.

  Every environment name the code reads is listed in openmx_env_registry.h
  with an id, its kind (what the readers parse), its scope, its default
  where the readers agree on one literal, the units that read it and a
  one-line description.  openmx_env_init() takes one
  snapshot of those names, and of every OPENMX_* name present in the
  environment, right after MPI_Init; openmx.c then copies rank 0's values
  of the global-scope names to every rank (openmx_env_pack/unpack), so a
  setting has one value for the whole run no matter which ranks the
  launcher passed it to.  Readers see that snapshot for the whole run,
  tests inject values without touching the process environment, a name
  that no setting of this build uses is reported once (a typo in a job
  script used to be silent), a value that does not parse for its kind is
  reported once, and the active settings can be printed with the output.

  Reading:
    openmx_env_flag_at(OPENMX_ENV_BAND_PROFILE, 0)   O(1), no parsing
    openmx_env_long_at / _ulong_at / _double_at / _mib_bytes_at / _at
  The id of OPENMX_X is OPENMX_ENV_X; a name outside the OPENMX_ namespace
  (GEMMUL8_DISABLE, OMPI_COMM_WORLD_LOCAL_RANK, ...) is OPENMX_ENV_EXT_<name>.
  openmx_env_get(name) remains a drop-in replacement for getenv() for names
  that are only known at run time: a registered or snapshotted name comes
  from the snapshot, any other name from the live environment, and before
  openmx_env_init() everything is live.

  One parser for every reader:
    flag    unset, empty or "auto" -> fallback; a number -> nonzero;
            on/yes/true/y -> 1; off/no/false/n -> 0 (any case);
            anything else -> fallback (and reported)
    int     a whole decimal integer (strtol), else fallback
    uint    a whole non-negative decimal integer, else fallback
    mib     uint, a size in MiB; openmx_env_mib_bytes_at() returns bytes,
            saturating at SIZE_MAX
    double  a whole finite decimal number (strtod), else fallback
  Surrounding blanks are ignored.
***********************************************************************/
#ifndef OPENMX_ENV_H
#define OPENMX_ENV_H

#include <stddef.h>
#include <stdio.h>

#include "openmx_env_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *name;    /* environment variable */
    const char *kind;    /* "flag", "int", "uint", "mib", "double", "string" or "tool" */
    const char *scope;   /* "global" (rank 0's value for every rank), "local" or "tool" */
    const char *dflt;    /* the readers' fallback when it is one literal ("auto", "unlimited"), else "" */
    const char *module;  /* units that read it, comma separated */
    const char *doc;     /* one line, or "" */
} OpenMXEnvEntry;

/* OPENMX_ENV_<id>: index of each registered name, in registry order. */
enum {
#define OPENMX_ENV_ID_(id, name, kind, scope, dflt, module, doc) OPENMX_ENV_##id,
    OPENMX_ENV_LIST(OPENMX_ENV_ID_)
#undef OPENMX_ENV_ID_
    OPENMX_ENV_COUNT
};

/* Take the snapshot (idempotent).  Safe to call from every rank. */
void openmx_env_init(void);
int  openmx_env_initialized(void);
/* Drop the snapshot and every override (tests). */
void openmx_env_reset(void);

/* --- readers by id (O(1); before openmx_env_init() they read the live
       environment and parse on the spot) ---------------------------------- */
const char        *openmx_env_at(int id);                       /* NULL when unset */
int                openmx_env_flag_at(int id, int fallback);
long               openmx_env_long_at(int id, long fallback);
unsigned long long openmx_env_ulong_at(int id, unsigned long long fallback);
double             openmx_env_double_at(int id, double fallback);
size_t             openmx_env_mib_bytes_at(int id, unsigned long long fallback_mib);
const char        *openmx_env_name(int id);                     /* "" for an invalid id */

/* --- readers by name (run-time names; a registered name is served from
       its slot) ------------------------------------------------------------ */
char              *openmx_env_get(const char *name);            /* getenv() replacement */
int                openmx_env_flag(const char *name, int fallback);
long               openmx_env_long(const char *name, long fallback);
unsigned long long openmx_env_ulong(const char *name, unsigned long long fallback);
double             openmx_env_double(const char *name, double fallback);
int                openmx_env_id(const char *name);             /* -1 when not registered */

/* --- overrides: the snapshot and, so that libraries and the HIP copies
       agree, the process environment.  0 on success. --------------------- */
int openmx_env_set(const char *name, const char *value);
int openmx_env_set_if_unset(const char *name, const char *value);
int openmx_env_unset(const char *name);

/* --- rank 0 -> every rank: pack the global-scope values into a malloc'd
       buffer (length returned, *buffer freed by the caller), apply a packed
       buffer (number of values changed, -1 when malformed). ---------------- */
size_t openmx_env_pack(char **buffer);
int    openmx_env_unpack(const char *buffer, size_t length);

/* --- the registry ------------------------------------------------------- */
size_t openmx_env_registry(const OpenMXEnvEntry **entries);     /* sorted by name */
const OpenMXEnvEntry *openmx_env_lookup(const char *name);       /* NULL when not registered */
/* OPENMX_* names found in the environment at openmx_env_init() that the
   registry does not list; names valid until openmx_env_reset(). */
size_t openmx_env_unknown(const char *const **names);

/* Print the set variables with their values (level 2 of the implementation
   document's diagnostics), or only the warnings (unknown_only != 0: the
   unknown names and the values that do not parse for their kind).  Nothing
   is printed when there is nothing to say. */
void openmx_env_report(FILE *out, int unknown_only);

#ifdef __cplusplus
}
#endif

#endif /* OPENMX_ENV_H */
