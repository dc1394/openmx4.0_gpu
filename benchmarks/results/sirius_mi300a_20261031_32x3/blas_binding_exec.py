#!/usr/bin/env python3
import os,sys
if os.environ.get('OMPI_COMM_WORLD_RANK')=='0':
    os.environ['LD_DEBUG']='bindings'
    os.environ['LD_DEBUG_OUTPUT']=os.environ['OMX_BLAS_AUDIT_OUTPUT']
os.execv(sys.argv[1],sys.argv[1:])
