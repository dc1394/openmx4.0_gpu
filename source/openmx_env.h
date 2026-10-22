/**********************************************************************
  openmx_env.h

  The environment settings of this build, read once.

  Every OPENMX_* (and the few GEMMUL8_* / CUDA_MPS_* names) that the code
  reads is listed in a registry with its kind, the units that read it and
  a one-line description.  openmx_env_init() takes one snapshot of those
  names, and of every OPENMX_* name present in the environment, right
  after MPI_Init.  Readers then see the same value for the whole run,
  tests can inject values without touching the process environment, a
  name that no setting of this build uses is reported once (a typo in a
  job script used to be silent), and the active settings can be printed
  with the output for reproducibility.

  openmx_env_get() is a drop-in replacement for getenv(): a registered or
  snapshotted name comes from the snapshot, any other name from the live
  environment, and before openmx_env_init() everything is live.  The
  parsing conventions of the existing readers (what counts as "on") are
  theirs; the typed helpers here are for new code.
***********************************************************************/
#ifndef OPENMX_ENV_H
#define OPENMX_ENV_H

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *name;    /* environment variable */
    const char *kind;    /* "flag", "int", "mib", "double", "string" or "mode": a hint, not the parser */
    const char *module;  /* translation units that read it, comma separated */
    const char *doc;     /* one line, or "" where the implementation document says nothing */
} OpenMXEnvEntry;

/* Take the snapshot (idempotent).  Safe to call from every rank. */
void openmx_env_init(void);
int  openmx_env_initialized(void);
/* Drop the snapshot and every override (tests). */
void openmx_env_reset(void);

/* getenv() replacement; NULL when unset.  The string stays valid until the
   next openmx_env_set() of the same name or openmx_env_reset(). */
char *openmx_env_get(const char *name);

/* Typed readers for new code.  An unset or empty variable yields the
   fallback; "0", "off", "no" and "false" (any case) are false for a flag. */
int                openmx_env_flag(const char *name, int fallback);
long               openmx_env_long(const char *name, long fallback);
double             openmx_env_double(const char *name, double fallback);
unsigned long long openmx_env_mib(const char *name, unsigned long long fallback);

/* Set a value for the rest of the run: the snapshot and, so that libraries
   and not yet migrated readers agree, the process environment.  Return 0
   on success.  openmx_env_set_if_unset() keeps an existing value. */
int openmx_env_set(const char *name, const char *value);
int openmx_env_set_if_unset(const char *name, const char *value);

/* The registry of this build's settings, sorted by name. */
size_t openmx_env_registry(const OpenMXEnvEntry **entries);
/* Registered? (binary search by name) */
const OpenMXEnvEntry *openmx_env_lookup(const char *name);
/* OPENMX_* names found in the environment at openmx_env_init() that the
   registry does not list; count returned, names valid until openmx_env_reset(). */
size_t openmx_env_unknown(const char *const **names);

/* Print the set variables of the registry with their values (level 2 of
   the implementation document's diagnostics), or only the unknown names
   (unknown_only != 0, a warning).  Nothing is printed when there is nothing. */
void openmx_env_report(FILE *out, int unknown_only);

#ifdef __cplusplus
}
#endif

#endif /* OPENMX_ENV_H */
