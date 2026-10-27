#ifndef OPENMX_CORE_QUADRATURE_H
#define OPENMX_CORE_QUADRATURE_H
#include "openmx_core.h"
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

/* The calculation owns this immutable rule until finish/destroy. Repeating
   initialization with the same count reuses its stable addresses; a different
   count is rejected rather than invalidating outstanding borrowed views.
   Failure leaves the owner and output unchanged. Initialize and finish require
   quiescent users; concurrent reads of an initialized rule are permitted. */
int openmx_core_radial_quadrature_initialize(OpenMXCoreContext *context, int count,
    OpenMXGaussRuleData *out, char *error, size_t error_capacity);
int openmx_core_radial_quadrature_get(const OpenMXCoreContext *context,
    OpenMXGaussRuleData *out, char *error, size_t error_capacity);

/* Independent rule for neutral-atom energy, stress and reference XC
   integration. The ownership and concurrency contract above also applies;
   initializing either rule never changes the other rule's borrowed views. */
int openmx_core_neutral_quadrature_initialize(OpenMXCoreContext *context, int count,
    OpenMXGaussRuleData *out, char *error, size_t error_capacity);
int openmx_core_neutral_quadrature_get(const OpenMXCoreContext *context,
    OpenMXGaussRuleData *out, char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
