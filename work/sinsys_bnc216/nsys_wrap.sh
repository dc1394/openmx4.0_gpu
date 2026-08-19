#!/bin/bash
# nsys toolchain GO/NO-GO (plan v2.6 sec. 9.2): profile ranks 0 and 1 of a
# 48-rank MPS band run; all other ranks exec openmx directly.  Profiling
# runs are NEVER pooled with production timing (plan separates them).
NSYS=/system/apps/ubuntu/22.04-202604/nvhpc/26.5/Linux_x86_64/26.5/profilers/13.2/Nsight_Systems/target-linux-x64/nsys
R=${OMPI_COMM_WORLD_RANK:-0}
if [ "$R" -le 1 ]; then
  exec "$NSYS" profile -t cuda,osrt --stats=false --cuda-memory-usage=false \
       -o "sinsys_bnc216.rank${R}" ../openmx sinsys_bnc216.dat -nt 1
else
  exec ../openmx sinsys_bnc216.dat -nt 1
fi
