#ifndef __CARTMESH_H
#include "cartmesh.hpp"
#endif

/// Set RHS = 3sin(x)sin(y)sin(z) for u_exact = sin(x)sin(y)sin(z)
void computeRHS(Vec f, Vec uexact, const CartMesh *const m)
{
	PetscReal *valuesrhs = std::malloc(m->gnpointotal()*sizeof(PetscReal));
	PetscReal *valuesuexact = std::malloc(m->gnpointotal()*sizeof(PetscReal));
	PetscInt *indices = std::malloc(m->gnpointotal()*sizeof(PetscInt));

	// point ordering index
	PetscInt l = 0;

	for(PetscInt k = 0; k < m->gnpoind(2); k++)
		for(PetscInt j = 0; j < m->gnpoind(1); j++)
			for(PetscInt i = 0; i < m->gnpoind(0); i++)
			{
				indices[l] = i + m->gnpoind(0)*j + m->gnpoind(1)*m->gnpoind(0)*k;

				valuesrhs[l] = 3.0*std::sin(m->gcoords(i,0))*std::sin(m->gcoords(j,1))*std::sin(m->gcoords(k,2));
				valuesuexact[l] = std::sin(m->gcoords(i,0))*std::sin(m->gcoords(j,1))*std::sin(m->gcoords(k,2));
			}

	VecSetValues(f, m->gnpointotal(), indices, valuesrhs, INSERT_VALUES);
	VecSetValues(uexact, m->gnpointotal(), indices, valuesuexact, INSERT_VALUES);

	free(valuesrhs);
	free(valuesexact);
	free(indices);
}

int main(int argc, char* argv[])
{
	char help[] = "Solves 3D Poisson equation by finite differences.\n\n";
	char * optfile;
	PetscMPIInt size;

	PetscInitialize(&argc, &argv, optfile, help);
	MPI_Comm_size(PETSC_COMM_WORLD,&size);
	if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"Currently single processor only!");

	PetscFinalize();
	return 0;
}
