#ifndef OPENMX_CORE_QUADRATURE_H
#define OPENMX_CORE_QUADRATURE_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int count;
  const double *nodes, *weights;
} OpenMXGaussRuleData;

/* The borrowed rule is valid only during the synchronous consumer call.
   Generation/allocation failure does not invoke the consumer. A
   consumer failure is reported, but its side effects are not rolled back. */
typedef int (*OpenMXGaussRuleConsumer)(void *user, const OpenMXGaussRuleData *rule,
    char *error, size_t error_capacity);
int openmx_core_with_gauss_rule(int count, OpenMXGaussRuleConsumer consumer,
    void *user, char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
