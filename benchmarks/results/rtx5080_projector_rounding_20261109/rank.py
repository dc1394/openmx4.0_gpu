import json,os,sys
from pathlib import Path
rank=int(next(os.environ[k] for k in ['OMPI_COMM_WORLD_LOCAL_RANK','MPI_LOCALRANKID','PMI_RANK'] if k in os.environ))
cpus={rank,rank+18}
os.sched_setaffinity(0,cpus)
folder=Path.cwd()
if os.environ.get('OMX_PRECISION_AUDIT_LIBRARY'):
    os.environ['LD_PRELOAD']=os.environ['OMX_PRECISION_AUDIT_LIBRARY']
    os.environ['OMX_PRECISION_AUDIT_LOG']=str(folder/f'rank-{rank:02d}-cusolver.jsonl')
(folder/f'rank-{rank:02d}.json').write_text(json.dumps(dict(rank=rank,pid=os.getpid(),cpus=sorted(os.sched_getaffinity(0)),
    requested_cpus=sorted(cpus),executable=sys.argv[1]),indent=2)+'\n')
os.execv(sys.argv[1],sys.argv[1:])
