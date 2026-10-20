#ifndef OPENMX_COMPLEX_H
#define OPENMX_COMPLEX_H

/* The legacy C ABI uses two consecutive doubles. Keep the same type in
   standalone numerical modules without importing application globals. */
#ifndef ___dcomplex_definition___
typedef struct { double r,i; } dcomplex;
#define ___dcomplex_definition___
#endif

#endif
