#include "openmx_precision.h"

static int fp32_enabled = 1;

int openmx_fp32_enabled(void)
{
    return fp32_enabled;
}

void openmx_set_fp32_enabled(int enabled)
{
    fp32_enabled = (enabled != 0);
}
