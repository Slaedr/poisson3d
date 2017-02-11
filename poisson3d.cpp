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
 * Since we don't solve for the boundary DOFs, we need to subtract 1 from each index computed by getLinearIndex.
 */
void computeLHS(Mat A, const CartMesh *const m)
{
	/*PetscReal values[NSTENCIL];
	PetscInt cindices[NSTENCIL];
	PetscInt rindices[1];
	PetscInt n = NSTENCIL;
	PetscInt m = 1;*/
	
	// nodes that don't have a Dirichlet node as a neighbor
	for(PetscInt k = 2; k < m->gnpoind(2)-2; k++)
		for(PetscInt j = 2; j < m->gnpoind(1)-2; j++)
			for(PetscInt i = 2; i < m->gnpoind(0)-2; i++)
			{
				PetscReal values[NSTENCIL];
				PetscInt cindices[NSTENCIL];
				PetscInt rindices[1];
				PetscInt n = NSTENCIL;
				PetscInt m = 1;

				rindices[0] = getLinearIndex(m,i,j,k)-1;

				cindices[0] = getLinearIndex(m,i-1,j,k)-1;
				cindices[1] = getLinearIndex(m,i,j-1,k)-1;
				cindices[2] = getLinearIndex(m,i,j,k-1)-1;
				cindices[3] = rindices[0]-1;
				cindices[4] = getLinearIndex(m,i+1,j,k)-1;
				cindices[5] = getLinearIndex(m,i,j+1,k)-1;
				cindices[7] = getLinearIndex(m,i,j,k+1)-1;
				
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
	
	k = 1;
	for(PetscInt j = 1; j < m->gnpoind(1)-1; j++)
		for(PetscInt i = 1; i < m->gnpoind(0)-1; i++)
		{
			if(j == 1)
			{
				if(i == 1)
				{
					PetscReal values[NSTENCIL-3];
					PetscInt cindices[NSTENCIL-3];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-3;
					PetscInt m = 1;
					
					rindices[0] = getLinearIndex(m,i,j,k)-1;

					cindices[0] = rindices[0]-1;
					cindices[1] = getLinearIndex(m,i+1,j,k)-1;
					cindices[2] = getLinearIndex(m,i,j+1,k)-1;
					cindices[3] = getLinearIndex(m,i,j,k+1)-1;
					
					values[0] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[0] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[0] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[1] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[2] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
					values[3] = -1.0/( (m->gcoords(2,k+1)-m->gcoords(2,k)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					MatSetValues(A, m, rindices, n, cindices, values, INSERT_VALUES);
				}
				else if(i == m->gnpoind(0)-2)
				{
					PetscReal values[NSTENCIL-3];
					PetscInt cindices[NSTENCIL-3];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-3;
					PetscInt m = 1;
					
					rindices[0] = getLinearIndex(m,i,j,k)-1;

					cindices[0] = getLinearIndex(m,i-1,j,k)-1;
					cindices[1] = rindices[0]-1;
					cindices[2] = getLinearIndex(m,i,j+1,k)-1;
					cindices[3] = getLinearIndex(m,i,j,k+1)-1;
					
					values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );

					values[1] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[1] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[1] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[2] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
					values[3] = -1.0/( (m->gcoords(2,k+1)-m->gcoords(2,k)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					MatSetValues(A, m, rindices, n, cindices, values, INSERT_VALUES);
				}
				else
				{
					PetscReal values[NSTENCIL-2];
					PetscInt cindices[NSTENCIL-2];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-2;
					PetscInt m = 1;
					
					rindices[0] = getLinearIndex(m,i,j,k)-1;

					cindices[0] = getLinearIndex(m,i-1,j,k)-1;
					cindices[1] = rindices[0]-1;
					cindices[2] = getLinearIndex(m,i+1,j,k)-1;
					cindices[3] = getLinearIndex(m,i,j+1,k)-1;
					cindices[4] = getLinearIndex(m,i,j,k+1)-1;
					
					values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );

					values[1] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[1] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[1] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[2] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[3] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
					values[4] = -1.0/( (m->gcoords(2,k+1)-m->gcoords(2,k)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					MatSetValues(A, m, rindices, n, cindices, values, INSERT_VALUES);
				}
			}
			else if(j == m->gnpoind(1)-2)
			{
				if(i == 1)
				{
					PetscReal values[NSTENCIL-3];
					PetscInt cindices[NSTENCIL-3];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-3;
					PetscInt m = 1;
					
					rindices[0] = getLinearIndex(m,i,j,k)-1;

					cindices[0] = getLinearIndex(m,i,j-1,k)-1;
					cindices[1] = rindices[0]-1;
					cindices[2] = getLinearIndex(m,i+1,j,k)-1;
					cindices[3] = getLinearIndex(m,i,j,k+1)-1;
					
					values[0] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );

					values[1] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[1] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[1] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[2] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[3] = -1.0/( (m->gcoords(2,k+1)-m->gcoords(2,k)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					MatSetValues(A, m, rindices, n, cindices, values, INSERT_VALUES);
				}
				else if(i == m->gnpoind(0)-2)
				{
					PetscReal values[NSTENCIL-3];
					PetscInt cindices[NSTENCIL-3];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-3;
					PetscInt m = 1;
					
					rindices[0] = getLinearIndex(m,i,j,k)-1;

					cindices[0] = getLinearIndex(m,i-1,j,k)-1;
					cindices[1] = getLinearIndex(m,i,j-1,k)-1;
					cindices[2] = rindices[0]-1;
					cindices[3] = getLinearIndex(m,i,j,k+1)-1;
					
					values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[1] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );

					values[2] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[2] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[2] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[3] = -1.0/( (m->gcoords(2,k+1)-m->gcoords(2,k)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					MatSetValues(A, m, rindices, n, cindices, values, INSERT_VALUES);
				}
				else
				{
					PetscReal values[NSTENCIL-2];
					PetscInt cindices[NSTENCIL-2];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-2;
					PetscInt m = 1;
					
					rindices[0] = getLinearIndex(m,i,j,k)-1;

					cindices[0] = getLinearIndex(m,i-1,j,k)-1;
					cindices[1] = getLinearIndex(m,i,j-1,k)-1;
					cindices[2] = rindices[0]-1;
					cindices[3] = getLinearIndex(m,i+1,j,k)-1;
					cindices[4] = getLinearIndex(m,i,j,k+1)-1;
					
					values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[1] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );

					values[2] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[2] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[2] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[3] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[4] = -1.0/( (m->gcoords(2,k+1)-m->gcoords(2,k)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					MatSetValues(A, m, rindices, n, cindices, values, INSERT_VALUES);
				}
			}
			else
			{
				if(i == 1)
				{
					PetscReal values[NSTENCIL-2];
					PetscInt cindices[NSTENCIL-2];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-2;
					PetscInt m = 1;
					
					rindices[0] = getLinearIndex(m,i,j,k)-1;

					cindices[0] = getLinearIndex(m,i,j-1,k)-1;
					cindices[1] = rindices[0]-1;
					cindices[2] = getLinearIndex(m,i+1,j,k)-1;
					cindices[3] = getLinearIndex(m,i,j+1,k)-1;
					cindices[4] = getLinearIndex(m,i,j,k+1)-1;
					
					values[0] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );

					values[1] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[1] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[1] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[2] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[3] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
					values[4] = -1.0/( (m->gcoords(2,k+1)-m->gcoords(2,k)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					MatSetValues(A, m, rindices, n, cindices, values, INSERT_VALUES);
				}
				else if(i == m->gnpoind(0)-2)
				{
					PetscReal values[NSTENCIL-2];
					PetscInt cindices[NSTENCIL-2];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-2;
					PetscInt m = 1;
					
					rindices[0] = getLinearIndex(m,i,j,k)-1;

					cindices[0] = getLinearIndex(m,i-1,j,k)-1;
					cindices[1] = getLinearIndex(m,i,j-1,k)-1;
					cindices[2] = rindices[0]-1;
					cindices[3] = getLinearIndex(m,i,j+1,k)-1;
					cindices[4] = getLinearIndex(m,i,j,k+1)-1;
					
					values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[1] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );

					values[2] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[2] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[2] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[3] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
					values[4] = -1.0/( (m->gcoords(2,k+1)-m->gcoords(2,k)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					MatSetValues(A, m, rindices, n, cindices, values, INSERT_VALUES);
				}
				else
				{
					PetscReal values[NSTENCIL-1];
					PetscInt cindices[NSTENCIL-1];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-1;
					PetscInt m = 1;
					
					rindices[0] = getLinearIndex(m,i,j,k);

					cindices[0] = getLinearIndex(m,i-1,j,k)-1;
					cindices[1] = getLinearIndex(m,i,j-1,k)-1;
					cindices[2] = rindices[0]-1;
					cindices[3] = getLinearIndex(m,i+1,j,k)-1;
					cindices[4] = getLinearIndex(m,i,j+1,k)-1;
					cindices[5] = getLinearIndex(m,i,j,k+1)-1;
					
					values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[1] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );

					values[2] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[2] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[2] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[3] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[4] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
					values[5] = -1.0/( (m->gcoords(2,k+1)-m->gcoords(2,k)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					MatSetValues(A, m, rindices, n, cindices, values, INSERT_VALUES);
				}
			}

		} // end loop
	
	k = n->gnpoind(2)-1;
	for(PetscInt j = 1; j < m->gnpoind(1)-1; j++)
		for(PetscInt i = 1; i < m->gnpoind(0)-1; i++)
		{
			if(j == 1)
			{
				if(i == 1)
				{
					PetscReal values[NSTENCIL-3];
					PetscInt cindices[NSTENCIL-3];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-3;
					PetscInt m = 1;
					
					rindices[0] = getLinearIndex(m,i,j,k);

					cindices[0] = getLinearIndex(m,i,j,k-1)-1;
					cindices[1] = rindices[0]-1;
					cindices[2] = getLinearIndex(m,i+1,j,k)-1;
					cindices[3] = getLinearIndex(m,i,j+1,k)-1;
					
					values[0] = -1.0/( (m->gcoords(2,k)-m->gcoords(2,k-1)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					values[1] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[1] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[1] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[2] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[3] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );

					MatSetValues(A, m, rindices, n, cindices, values, INSERT_VALUES);
				}
				else if(i == m->gnpoind(0)-2)
				{
					PetscReal values[NSTENCIL-3];
					PetscInt cindices[NSTENCIL-3];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-3;
					PetscInt m = 1;
					
					rindices[0] = getLinearIndex(m,i,j,k);

					cindices[0] = getLinearIndex(m,i-1,j,k)-1;
					cindices[1] = getLinearIndex(m,i,j,k-1)-1;
					cindices[2] = rindices[0]-1;
					cindices[3] = getLinearIndex(m,i,j+1,k)-1;
					
					values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[1] = -1.0/( (m->gcoords(2,k)-m->gcoords(2,k-1)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					values[2] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[2] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[2] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[3] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );

					MatSetValues(A, m, rindices, n, cindices, values, INSERT_VALUES);
				}
				else
				{
					PetscReal values[NSTENCIL-2];
					PetscInt cindices[NSTENCIL-2];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-2;
					PetscInt m = 1;
					
					rindices[0] = getLinearIndex(m,i,j,k);

					cindices[0] = getLinearIndex(m,i-1,j,k)-1;
					cindices[1] = getLinearIndex(m,i,j,k-1)-1;
					cindices[2] = rindices[0]-1;
					cindices[3] = getLinearIndex(m,i+1,j,k)-1;
					cindices[4] = getLinearIndex(m,i,j+1,k)-1;
					
					values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[1] = -1.0/( (m->gcoords(2,k)-m->gcoords(2,k-1)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					values[2] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[2] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[2] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[3] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[4] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );

					MatSetValues(A, m, rindices, n, cindices, values, INSERT_VALUES);
				}
			}
			else if(j == m->gnpoind(1)-2)
			{
				if(i == 1)
				{
					PetscReal values[NSTENCIL-3];
					PetscInt cindices[NSTENCIL-3];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-3;
					PetscInt m = 1;
					
					rindices[0] = getLinearIndex(m,i,j,k);

					cindices[0] = getLinearIndex(m,i,j-1,k)-1;
					cindices[1] = getLinearIndex(m,i,j,k-1)-1;
					cindices[2] = rindices[0]-1;
					cindices[3] = getLinearIndex(m,i+1,j,k)-1;
					
					values[0] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
					values[1] = -1.0/( (m->gcoords(2,k)-m->gcoords(2,k-1)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					values[2] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[2] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[2] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[3] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );

					MatSetValues(A, m, rindices, n, cindices, values, INSERT_VALUES);
				}
				else if(i == m->gnpoind(0)-2)
				{
					PetscReal values[NSTENCIL-3];
					PetscInt cindices[NSTENCIL-3];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-3;
					PetscInt m = 1;
					
					rindices[0] = getLinearIndex(m,i,j,k);

					cindices[0] = getLinearIndex(m,i-1,j,k)-1;
					cindices[1] = getLinearIndex(m,i,j-1,k)-1;
					cindices[2] = getLinearIndex(m,i,j,k-1)-1;
					cindices[3] = rindices[0]-1;
					
					values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[1] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
					values[2] = -1.0/( (m->gcoords(2,k)-m->gcoords(2,k-1)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					values[3] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[3] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[3] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					MatSetValues(A, m, rindices, n, cindices, values, INSERT_VALUES);
				}
				else
				{
					PetscReal values[NSTENCIL-2];
					PetscInt cindices[NSTENCIL-2];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-2;
					PetscInt m = 1;
					
					rindices[0] = getLinearIndex(m,i,j,k);

					cindices[0] = getLinearIndex(m,i-1,j,k)-1;
					cindices[1] = getLinearIndex(m,i,j-1,k)-1;
					cindices[2] = getLinearIndex(m,i,j,k-1)-1;
					cindices[3] = rindices[0]-1;
					cindices[4] = getLinearIndex(m,i+1,j,k)-1;
					
					values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[1] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
					values[2] = -1.0/( (m->gcoords(2,k)-m->gcoords(2,k-1)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					values[3] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[3] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[3] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[4] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );

					MatSetValues(A, m, rindices, n, cindices, values, INSERT_VALUES);
				}
			}
			else
			{
				if(i == 1)
				{
					PetscReal values[NSTENCIL-2];
					PetscInt cindices[NSTENCIL-2];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-2;
					PetscInt m = 1;
					
					rindices[0] = getLinearIndex(m,i,j,k);

					cindices[0] = getLinearIndex(m,i,j-1,k)-1;
					cindices[1] = getLinearIndex(m,i,j,k-1)-1;
					cindices[2] = rindices[0]-1;
					cindices[3] = getLinearIndex(m,i+1,j,k)-1;
					cindices[4] = getLinearIndex(m,i,j+1,k)-1;
					
					values[0] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
					values[1] = -1.0/( (m->gcoords(2,k)-m->gcoords(2,k-1)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					values[2] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[2] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[2] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[3] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[4] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );

					MatSetValues(A, m, rindices, n, cindices, values, INSERT_VALUES);
				}
				else if(i == m->gnpoind(0)-2)
				{
					PetscReal values[NSTENCIL-2];
					PetscInt cindices[NSTENCIL-2];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-2;
					PetscInt m = 1;
					
					rindices[0] = getLinearIndex(m,i,j,k);

					cindices[0] = getLinearIndex(m,i-1,j,k)-1;
					cindices[1] = getLinearIndex(m,i,j-1,k)-1;
					cindices[2] = getLinearIndex(m,i,j,k-1)-1;
					cindices[3] = rindices[0]-1;
					cindices[4] = getLinearIndex(m,i,j+1,k)-1;
					
					values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[1] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
					values[2] = -1.0/( (m->gcoords(2,k)-m->gcoords(2,k-1)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					values[3] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[3] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[3] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[4] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );

					MatSetValues(A, m, rindices, n, cindices, values, INSERT_VALUES);
				}
				else
				{
					PetscReal values[NSTENCIL-1];
					PetscInt cindices[NSTENCIL-1];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-1;
					PetscInt m = 1;
					
					rindices[0] = getLinearIndex(m,i,j,k);

					cindices[0] = getLinearIndex(m,i-1,j,k)-1;
					cindices[1] = getLinearIndex(m,i,j-1,k)-1;
					cindices[2] = getLinearIndex(m,i,j,k-1)-1;
					cindices[3] = rindices[0]-1;
					cindices[4] = getLinearIndex(m,i+1,j,k)-1;
					cindices[5] = getLinearIndex(m,i,j+1,k)-1;
					
					values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[1] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
					values[2] = -1.0/( (m->gcoords(2,k)-m->gcoords(2,k-1)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					values[3] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[3] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[3] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[4] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[5] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );

					MatSetValues(A, m, rindices, n, cindices, values, INSERT_VALUES);
				}
			}

		} // end loop
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
