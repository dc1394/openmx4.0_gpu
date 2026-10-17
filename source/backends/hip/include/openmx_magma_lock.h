#ifndef OPENMX_MAGMA_LOCK_H
#define OPENMX_MAGMA_LOCK_H
#include <mutex>

/* MAGMA's logical-device map is process-global. Single-device retries and
   multi-device solves must use the same lock while calling the library. */
inline std::mutex &openmx_magma_runtime_mutex()
{
    static std::mutex lock;
    return lock;
}
#endif
