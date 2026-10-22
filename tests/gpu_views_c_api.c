#include "openmx_gpu_views.h"
#include <stddef.h>
#include <string.h>

extern int gpu_views_test_select(int);
extern int gpu_views_test_current(void);
int gpu_views_c_layout(size_t size,size_t allocated,size_t device)
{
    return sizeof(OpenMXGpuReadView)==size && offsetof(OpenMXGpuReadView,allocated)==allocated &&
        offsetof(OpenMXGpuReadView,device)==device;
}
int gpu_views_c_lifetime(const void *source,size_t bytes,int next)
{
    /* The C factory initializes the output, including an uninitialized C
       object. C has explicit release; C++ adds scope-based ownership. */
    OpenMXGpuReadView view;
    if(!openmx_gpu_read_view(&view,source,bytes))
        return (!view.pointer && !view.allocated && view.device==-1)?0:-1;
    const int copied=!bytes || !memcmp(view.pointer,source,bytes);
    if(gpu_views_test_select(next))return -2;
    openmx_gpu_read_view_release(&view);
    const int released=!view.pointer && !view.allocated && view.device==-1 && gpu_views_test_current()==next;
    openmx_gpu_read_view_release(&view);
    return copied && released;
}
