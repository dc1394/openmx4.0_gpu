#pragma once
#ifndef LEAKDETECT_H
#define LEAKDETECT_H

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void leak_detect_init(void);
void *leak_detect_malloc(size_t, const char *, unsigned int);
void leak_detect_free(void *);
void leak_detect_check(void);
void leak_detect_finalize(void);

#ifdef __cplusplus
}
#endif

#ifdef LEAK_DETECT
#define malloc(s) leak_detect_malloc((s), __FILE__, __LINE__)
#define free      leak_detect_free
#endif

#endif
