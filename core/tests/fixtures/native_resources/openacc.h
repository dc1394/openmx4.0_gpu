#ifndef OPENMX_TEST_NATIVE_RESOURCE_OPENACC_H
#define OPENMX_TEST_NATIVE_RESOURCE_OPENACC_H
#include <stddef.h>
typedef enum { acc_device_nvidia=5 } acc_device_t;
static inline int acc_get_device_num(acc_device_t) { return openmx_test_acc_device(); }
static inline void acc_set_device_num(int device,acc_device_t) { openmx_test_acc_set(device); }
static inline void *acc_malloc(size_t bytes) { return openmx_test_acc_malloc(bytes); }
static inline void acc_free(void *pointer) { openmx_test_acc_free(pointer); }
#endif
