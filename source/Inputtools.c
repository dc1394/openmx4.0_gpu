/***************************************************
   Copyright (C), 2002-2026, Hiori Kino.
   This library follows the terms of GNU GPL.

   The keyword-file reader now lives in the independent OpenMX::io package
   (modules/io); this file keeps the original handle-less API of the
   application and its post-processing tools over it.  The current file,
   the nesting stack and the error counter that used to be five file-scope
   statics are one owned object created by the first input_open().
***************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "Inputtools.h"
#include "openmx_io.h"

#define MAXNEST 14
#define BUFSIZE 4096   /* the longest value the original reader could hold */

static OpenMXKeywordFiles *files = NULL;

static void release_files(void)
{
  openmx_io_keyword_files_destroy(files);
  files = NULL;
}

static OpenMXKeywordFiles *stack(void)
{
  if (files==NULL) {
    files = openmx_io_keyword_files_create(MAXNEST, stdout);
    if (files!=NULL) atexit(release_files);
  }
  return files;
}

int input_open(const char *fname)
{
  return openmx_io_open(stack(), fname);
}

int input_close()
{
  return openmx_io_close(stack());
}

int input_cmpstring(char *str, int *ret, int nvals, char **strval, int *ivals)
{
  return openmx_io_cmpstring(str, ret, nvals, (const char *const *)strval, ivals);
}

int input_errorCount()
{
  return openmx_io_error_count(stack());
}

OpenMXKeywordFiles *input_keyword_files(void)
{
  return stack();
}

int input_logical(const char *key, int *ret, const int defval)
{
  return openmx_io_logical(stack(), key, ret, defval);
}

int input_int(const char *key, int *ret, const int defval)
{
  return openmx_io_int(stack(), key, ret, defval);
}

int input_double(const char *key, double *ret, const double defval)
{
  /* the original returned 0, not -1, for a value it could not read */
  int result = openmx_io_double(stack(), key, ret, defval);
  return result<0 ? 0 : result;
}

int input_string(const char *key, char *ret, const char *defval)
{
  return openmx_io_string(stack(), key, ret, BUFSIZE, defval);
}

int input_string2int(const char *key, int *ret, int nvals, char **strval, int *ivals)
{
  return openmx_io_string2int(stack(), key, ret, nvals, (const char *const *)strval, ivals);
}

int input_stringv(const char *key, const int nret, char **ret, char **defval)
{
  return openmx_io_stringv(stack(), key, nret, ret, BUFSIZE, (const char *const *)defval);
}

int input_doublev(const char *key, const int nret, double *ret, double *defval)
{
  return openmx_io_doublev(stack(), key, nret, ret, defval);
}

int input_intv(const char *key, const int nret, int *ret, int *defval)
{
  return openmx_io_intv(stack(), key, nret, ret, defval);
}

FILE *input_find(const char *key)
{
  return openmx_io_find(stack(), key);
}

int input_last(const char *key)
{
  return openmx_io_last(stack(), key);
}
