#ifndef OPENMX_GPU_ARCHITECTURE_H
#define OPENMX_GPU_ARCHITECTURE_H
#include <string.h>

/* Throughput classification only. FP64 instruction support must be queried
   from the runtime; an unlisted architecture is not assumed to lack FP64. */
static inline int openmx_amd_high_throughput_fp64(const char *architecture)
{
    static const char *const known[] = {
        "gfx906", "gfx908", "gfx90a", "gfx940", "gfx941", "gfx942", "gfx950"
    };
    if (!architecture) return 0;
    for (unsigned i=0; i<sizeof(known)/sizeof(known[0]); ++i) {
        const size_t n=strlen(known[i]);
        if (strncmp(architecture,known[i],n)==0 &&
            (architecture[n]=='\0' || architecture[n]==':')) return 1;
    }
    return 0;
}
#endif
