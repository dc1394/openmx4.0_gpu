#ifndef OPENMX_TEST_REFERENCE_ARITHMETIC_H
#define OPENMX_TEST_REFERENCE_ARITHMETIC_H
#include <math.h>

/* Exercise binary64 reference arithmetic on hosts with a wider long double.
   Apple Silicon's long double has the same precision as double. These types
   belong only to the independent numerical test oracles. */
#ifdef OPENMX_TEST_REFERENCE_BINARY64
typedef double OpenMXTestReference;
static inline OpenMXTestReference OpenMX_TestReferenceSqrt(OpenMXTestReference x) { return sqrt(x); }
static inline OpenMXTestReference OpenMX_TestReferenceAbs(OpenMXTestReference x) { return fabs(x); }
static inline OpenMXTestReference OpenMX_TestReferencePow(OpenMXTestReference x,OpenMXTestReference y) { return pow(x,y); }
#else
typedef long double OpenMXTestReference;
static inline OpenMXTestReference OpenMX_TestReferenceSqrt(OpenMXTestReference x) { return sqrtl(x); }
static inline OpenMXTestReference OpenMX_TestReferenceAbs(OpenMXTestReference x) { return fabsl(x); }
static inline OpenMXTestReference OpenMX_TestReferencePow(OpenMXTestReference x,OpenMXTestReference y) { return powl(x,y); }
#endif
#endif
