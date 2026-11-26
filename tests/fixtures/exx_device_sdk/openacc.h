#ifndef OPENMX_TEST_EXX_DEVICE_SDK_OPENACC_H
#define OPENMX_TEST_EXX_DEVICE_SDK_OPENACC_H

/* The headers of the main program name this type; the unit under test does
   not use it. */
typedef int acc_device_t;

/* The two routines of OpenACC that the whole calls of
   source/Exact_Exchange_Device.c use: the CUDA stream of a queue, and the
   wait for the work of a queue. tests/exx_device_product_harness.c defines
   them. */
void *acc_get_cuda_stream(int async);
void acc_wait(int wait_arg);

#endif
