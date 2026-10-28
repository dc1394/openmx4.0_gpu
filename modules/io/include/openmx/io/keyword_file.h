#ifndef OPENMX_IO_KEYWORD_FILE_H
#define OPENMX_IO_KEYWORD_FILE_H

/* C interface of the OpenMX keyword-file reader (see keyword_file.hpp for
   the rules).  It exposes the nesting model of the original Inputtools:
   a stack of open files, lookups answered by the top one, an error counter
   reset by each open.  Return codes of the lookups are those of the original
   API: 1 found, 0 missing (the value is the fallback), -1 invalid (an error
   is recorded).  -1 is also returned when no file is open or a required
   pointer is NULL.  Every function tolerates a NULL stack. */

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OpenMXKeywordFiles OpenMXKeywordFiles;

/* capacity: the nesting limit (at least 1).  diagnostics, when not NULL,
   receives every error message as it is recorded.  NULL on allocation failure. */
OpenMXKeywordFiles* openmx_io_keyword_files_create(int capacity, FILE* diagnostics);
void openmx_io_keyword_files_destroy(OpenMXKeywordFiles* files);

/* 1 when the file is open and on top of the stack, 0 when the stack is full
   or the file cannot be read (a message is recorded either way). */
int openmx_io_open(OpenMXKeywordFiles* files, const char* path);
/* fclose's result for the top file; EOF when nothing is open. */
int openmx_io_close(OpenMXKeywordFiles* files);
int openmx_io_depth(const OpenMXKeywordFiles* files);
/* errors recorded (in any file) since the last open */
int openmx_io_error_count(const OpenMXKeywordFiles* files);

int openmx_io_logical(OpenMXKeywordFiles* files, const char* key, int* value, int fallback);
int openmx_io_int(OpenMXKeywordFiles* files, const char* key, int* value, int fallback);
int openmx_io_double(OpenMXKeywordFiles* files, const char* key, double* value, double fallback);
/* value receives at most capacity-1 characters and a terminator; a value
   that does not fit is an error (-1) and leaves the fallback. */
int openmx_io_string(OpenMXKeywordFiles* files, const char* key, char* value, size_t capacity, const char* fallback);
/* value must be one of count spellings in texts (case-insensitively), and
   receives the matching element of values; the fallback is values[0]. */
int openmx_io_string2int(OpenMXKeywordFiles* files, const char* key, int* value,
                         int count, const char* const* texts, const int* values);
/* count tokens after the keyword into values[i] (each of capacity bytes),
   fallback[i] first. */
int openmx_io_stringv(OpenMXKeywordFiles* files, const char* key, int count,
                      char* const* values, size_t capacity, const char* const* fallback);
int openmx_io_doublev(OpenMXKeywordFiles* files, const char* key, int count,
                      double* values, const double* fallback);
int openmx_io_intv(OpenMXKeywordFiles* files, const char* key, int count,
                   int* values, const int* fallback);

/* The top file's stream positioned after the line whose first token is key,
   or NULL when absent (or nothing is open). */
FILE* openmx_io_find(OpenMXKeywordFiles* files, const char* key);
/* Consumes the rest of the current line and one more line; 1 when that
   line's first token is key. */
int openmx_io_last(OpenMXKeywordFiles* files, const char* key);

/* The matching rule alone: 1 and *value = values[i] for the first spelling
   texts[i] equal to text (case-insensitively, same length), else 0. */
int openmx_io_cmpstring(const char* text, int* value, int count, const char* const* texts, const int* values);

#ifdef __cplusplus
}
#endif

#endif
