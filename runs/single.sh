#!/bin/bash
#PBS -l nodes=1:ppn=8
#PBS -l walltime=00:01:00
#PBS -A rck-371-aa
#PBS -o poisson-fgpilu-8.log
#PBS -e poisson-fgpilu.err
#PBS -N poisson-fgpilu

module load iomkl/2016.02

cd $PBS_O_WORKDIR/../build
export OMP_NUM_THREADS=8
export IPATH_NO_CPUAFFINITY=1
mpiexec -n 1 -npernode 1 ./poisson3d ../test.control ../petscrc
