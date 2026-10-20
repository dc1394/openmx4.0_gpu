#ifndef OPENMX_PROJECTOR_EXCHANGE_H
#define OPENMX_PROJECTOR_EXCHANGE_H
#include "openmx_core_projector_exchange.h"
#include "mpi.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

/* Explicit communicator: no application globals and no hidden default owner.
   MPI sends must be waited for before buffer reuse or destruction. */
typedef struct {
  OpenMXProjectorExchange *owner;
  OpenMXProjectorExchangeBuffers buffers;
  MPI_Comm communicator;
} OpenMXProjectorTransfer;

static inline void OpenMX_TransferFail(MPI_Comm communicator,const char *error)
{
  fprintf(stderr,"Projector exchange: %s\n",error);
  fflush(stderr);
  MPI_Abort(communicator,EXIT_FAILURE);
  abort();
}

static inline OpenMXProjectorTransfer OpenMX_TransferBegin(MPI_Comm communicator,int peers,size_t scalar_bytes)
{
  OpenMXProjectorTransfer transfer;
  char error[256];
  const int scalar=scalar_bytes==sizeof(float)?OPENMX_EXCHANGE_FLOAT:
                   scalar_bytes==sizeof(double)?OPENMX_EXCHANGE_DOUBLE:0;
  transfer.communicator=communicator;
  transfer.owner=openmx_core_projector_exchange_create(peers,scalar,&transfer.buffers,error,sizeof(error));
  if (!transfer.owner) OpenMX_TransferFail(communicator,error);
  return transfer;
}

static inline void *OpenMX_TransferBuffer(OpenMXProjectorTransfer *transfer,int direction,int elements)
{
  void *buffer;
  char error[256];
  if (!openmx_core_projector_exchange_buffer(transfer->owner,direction,elements,&buffer,error,sizeof(error)))
    OpenMX_TransferFail(transfer->communicator,error);
  return buffer;
}

/* Keep MPI's signed-int count representable before multiplying dimensions. */
static inline int OpenMX_TransferCount(OpenMXProjectorTransfer *transfer,int blocks,int rows,int columns)
{
  if (blocks<0 || rows<0 || columns<0)
    OpenMX_TransferFail(transfer->communicator,"negative projector message dimension");
  if (!blocks || !rows || !columns) return 0;
  if (rows>INT_MAX/columns || blocks>INT_MAX/(rows*columns))
    OpenMX_TransferFail(transfer->communicator,"projector message exceeds MPI int count");
  return blocks*rows*columns;
}

static inline void OpenMX_TransferEnd(OpenMXProjectorTransfer *transfer)
{
  openmx_core_projector_exchange_destroy(transfer->owner);
  transfer->owner=NULL;
}
#endif
