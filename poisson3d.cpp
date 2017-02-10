#ifndef __CARTMESH_H
#include "cartmesh.hpp"
#endif

/// Generate a non-uniform mesh in a cuboid
void generateCartMesh(PetscInt npdim[NDIM], PetscReal rmin[NDIM], PetscReal rmax[NDIM], CartMesh *const m);

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
