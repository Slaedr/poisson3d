PaCaPoisson3D
=============

Parallel Cartesian-grid 3D Poisson solver. Currently assumes homogeneous Dirichlet boundaries.

The code is MPI-parallel, using PETSc as the parallel computation framework. PETSc's DMDA is used to handle the vectors and matrix.
