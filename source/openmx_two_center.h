#ifndef OPENMX_TWO_CENTER_H
#define OPENMX_TWO_CENTER_H
#include "openmx_core_two_center.h"
#include <stdio.h>
#include <stdlib.h>

/* Synchronous host adapter. The enclosing C++ workspace owns all buffers;
   output rows keep the original native ordering and stabilization calls. */
static inline void OpenMX_TwoCenterTransform(const OpenMXTwoCenterBuffers *buffers,
    const dcomplex *angular,size_t rows,size_t columns,size_t row,size_t column,
    int l0,int l1,const dcomplex *conversion,size_t orders,size_t width,
    const OpenMXTwoCenterGeometry *geometry)
{
  const OpenMXTwoCenterInput input={angular,rows,columns,row,column,l0,l1,
      conversion,orders,width,*geometry};
  char error[256];
  if (!openmx_core_two_center_transform(&input,buffers,error,sizeof(error))) {
    fprintf(stderr,"OpenMX two-center transform: %s\n",error);
    abort();
  }
}
static inline double OpenMX_TwoCenterValue(const OpenMXTwoCenterBuffers *buffers,
    int channel,int row,int column)
{ return buffers->values[((size_t)channel*buffers->rows+row)*buffers->columns+column]; }
static inline double OpenMX_TwoCenterMomentum(const OpenMXTwoCenterBuffers *buffers,
    int axis,int row,int column)
{ return buffers->momenta[((size_t)axis*buffers->rows+row)*buffers->columns+column]; }
#endif
