#!/bin/bash

################################
### Apps, input, output info ###
################################

PATH_OPENMX="~/program/OpenMX/source"
PATH_FLPQ="~/program/FLPQ/source"
PATH_FLPQV="~/program/FLPQViewer"
EXE_O="${PATH_OPENMX}/openmx"
EXE_F="${PATH_FLPQ}/flpq"

INP_O=INPUT.dat
LOG_O=RESULT.std
INP_F=flpq.inp
LOG_F=flpq.log

RUN_O="mpirun -np 4"
RUN_F="mpirun -np 4"

#RUN_O="srun --exclusive --mem-per-cpu=${MEMORY_PER_CPU_CORE} -n ${NUM_PROCS} -c ${NUM_THREADS} -N ${NUM_NODES}"
#RUN_F="srun --exclusive --mem-per-cpu=${MEMORY_PER_CPU_CORE} -n 64 -c 4 -N 2"

################
### Workflow ###
################

NAMEDIR=$PWD

### For flpq ### 
cd ${NAMEDIR}
mkdir ${NAMEDIR}/flpq
cp ${INP_O}\# ${NAMEDIR}/flpq/${INP_O}
cd ${NAMEDIR}/flpq

cat << 'EOF' >> ${INP_O}
DM.export   on
EOF
sed -i 's/^MD\.Type[[:space:]]\+.*/MD.type    nomd/' ${INP_O}

$RUN_O $EXE_O $INP_O -nt ${OMP_NUM_THREADS} > $LOG_O

cd ${NAMEDIR}
cp ${INP_F} ${NAMEDIR}/flpq/${INP_F}
cd ${NAMEDIR}/flpq
$RUN_F $EXE_F $INP_F > $LOG_F

### End flpq ###

### For flpq_ew ###
cd ${NAMEDIR}
mkdir ${NAMEDIR}/flpq_ew
cp ${INP_O}\# ${NAMEDIR}/flpq_ew/${INP_O}
cd ${NAMEDIR}/flpq_ew

cat << 'EOF' >> ${INP_O}
DM.export   window
DM.specify.energy.range   on
DM.energy.range  -3.0  0.0
EOF
sed -i 's/^MD\.Type[[:space:]]\+.*/MD.type    nomd/' ${INP_O}

$RUN_O $EXE_O $INP_O -nt ${OMP_NUM_THREADS} > $LOG_O

cd ${NAMEDIR}
cp ${INP_F} ${NAMEDIR}/flpq_ew/${INP_F}
cd ${NAMEDIR}/flpq_ew
$RUN_F $EXE_F $INP_F > $LOG_F

conda run -n flpqv_env python3 ${PATH_FLPQV}/flpqv/main_flpqv.py chempot.toml