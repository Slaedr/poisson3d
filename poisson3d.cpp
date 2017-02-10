#ifndef __CARTMESH_H
#include "cartmesh.hpp"
#endif

inline PetscInt getLinearIndex(const CartMesh *const m, const PetscInt i, const PetscInt j, const PetscInt k)
{
	return i + m->gnpoind(0)*j + m->gnpoind(0)*m->gnpoind(1)*k;
}

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

				l++;
			}

	VecSetValues(f, m->gnpointotal(), indices, valuesrhs, INSERT_VALUES);
	VecSetValues(uexact, m->gnpointotal(), indices, valuesuexact, INSERT_VALUES);

	free(valuesrhs);
	free(valuesexact);
	free(indices);
}

/// Set stiffness matrix
/** Inserts entries rowwise into the matrix.
 */
void computeLHS(Mat A, const CartMesh *const m)
{
	PetscReal values[NSTENCIL];
	PetscInt cindices[NSTENCIL];
	PetscInt rindices[1];
	PetscInt n = NSTENCIL;
	PetscInt m = 1;
	PetscInt l = 0;	// linear node numbering
	
	// nodes that don't have a Dirichlet node as a neighbor
	for(PetscInt k = 2; k < m->gnpoind(2)-2; k++)
		for(PetscInt j = 2; j < m->gnpoind(1)-2; j++)
			for(PetscInt i = 2; i < m->gnpoind(0)-2; i++)
			{
				rindices[0] = getLinearIndex(m,i,j,k);

				cindices[0] = getLinearIndex(m,i-1,j,k);
				cindices[1] = getLinearIndex(m,i,j-1,k);
				cindices[2] = getLinearIndex(m,i,j,k-1);
				cindices[3] = rindices[0];
				cindices[4] = getLinearIndex(m,i+1,j,k);
				cindices[5] = getLinearIndex(m,i,j+1,k);
				cindices[7] = getLinearIndex(m,i,j,k+1);
				
				values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
				values[1] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
				values[2] = -1.0/( (m->gcoords(2,k)-m->gcoords(2,k-1)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

				values[3] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
				values[3] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
				values[3] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

				values[4] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
				values[5] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
				values[6] = -1.0/( (m->gcoords(2,k+1)-m->gcoords(2,k)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

				MatSetValues(A, m, rindices, n, cindices, values, INSERT_VALUES);
			}

	// next, nodes with Dirichlet nodes as neighbors
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
