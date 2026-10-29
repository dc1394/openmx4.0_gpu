#ifndef OPENMX_BASIS_COMPLEX_H
#define OPENMX_BASIS_COMPLEX_H

/* Shared C representation: real followed by imaginary double. The historic
   name is retained so native C callers and C++ owners use the very same type,
   without aliasing std::complex storage or importing application headers. */
#ifndef ___dcomplex_definition___
typedef struct { double r,i; } dcomplex;
#define ___dcomplex_definition___
#endif

#endif
