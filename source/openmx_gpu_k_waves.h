#ifndef OPENMX_GPU_K_WAVES_H
#define OPENMX_GPU_K_WAVES_H

#include <stdint.h>
#include <stdlib.h>

/* A wave admits distinct owners, rather than k points. Each admitted owner
   completes its ascending queue before its buffers are released. This keeps
   the MPI/NUMA assignment intact and bounds live owners on every device,
   including uneven queues, partially occupied GPUs and multiple nodes.
   Output arrays have points and points+1 entries. -1 declines invalid input
   or an allocation failure; no partially built plan may then be consumed. */
static inline int openmx_gpu_k_waves(int points, int ranks, int devices,
                                    const int *owner, const int *rank_device,
                                    const int *device_limit, int interleave,
                                    int *order, int *offset)
{
    int *head = NULL, *next = NULL, *selected = NULL;
    int position = 0, waves = 0;
    if (points < 0 || ranks < 1 || devices < 1 || !offset ||
        (points && (!owner || !order)) || !rank_device || !device_limit ||
        (size_t)ranks > SIZE_MAX / sizeof(int) ||
        (size_t)points > SIZE_MAX / sizeof(int)) return -1;
    for (int d = 0; d < devices; ++d) if (device_limit[d] < 1) return -1;
    offset[0] = 0;
    if (!points) return 0;
    head = (int *)malloc((size_t)ranks * sizeof(int));
    next = (int *)malloc((size_t)points * sizeof(int));
    selected = (int *)malloc((size_t)ranks * sizeof(int));
    if (!head || !next || !selected) goto failed;
    for (int r = 0; r < ranks; ++r) head[r] = -1;
    for (int p = points - 1; p >= 0; --p) {
        int r = owner[p];
        if (r < 0 || r >= ranks || rank_device[r] < 0 || rank_device[r] >= devices)
            goto failed;
        next[p] = head[r];
        head[r] = p;
    }
    while (position < points) {
        int before = position;
        for (int r = 0; r < ranks; ++r) selected[r] = 0;
        for (int d = 0; d < devices; ++d) {
            int admitted = 0;
            for (int r = 0; r < ranks && admitted < device_limit[d]; ++r) {
                if (rank_device[r] == d && head[r] >= 0) {
                    selected[r] = 1;
                    ++admitted;
                }
            }
        }
        if (interleave) {
            int progress;
            do {
                progress = 0;
                for (int r = 0; r < ranks; ++r) if (selected[r] && head[r] >= 0) {
                    int p = head[r];
                    order[position++] = p;
                    head[r] = next[p];
                    progress = 1;
                }
            } while (progress);
        } else {
            for (int p = 0; p < points; ++p) if (selected[owner[p]])
                order[position++] = p;
            for (int r = 0; r < ranks; ++r) if (selected[r]) head[r] = -1;
        }
        if (position == before) goto failed;
        offset[++waves] = position;
    }
    free(selected);
    free(next);
    free(head);
    return waves;
failed:
    free(selected);
    free(next);
    free(head);
    return -1;
}

#endif
