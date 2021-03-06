#!/bin/bash
#PBS -l nodes=1:ppn=8
#PBS -l walltime=00:15:00
#PBS -A rck-371-aa
#PBS -o poisson_100-fgpilu-1t.log
#PBS -e poisson-fgpilu.err
#PBS -N poisson-fgpilu

module load ifort_icc/14.0.4 openmpi/1.6.3-intel MKL/11.2
module load petsc/3.5.3-openmpi-1.6.3-intel

cd $PBS_O_WORKDIR/../build
export IPATH_NO_CPUAFFINITY=1
export OMP_NUM_THREADS=1
mpiexec -n 1 -npernode 1 ./poisson3d ../test.control ../petscrc
