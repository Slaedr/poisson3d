#ifndef __CARTMESH_H
#include "cartmesh.hpp"
#endif

/// Gives the index of a point in the point grid collapsed to 1D
inline PetscInt getFlattenedIndex(const CartMesh *const m, const PetscInt i, const PetscInt j, const PetscInt k)
{
	return i + m->gnpoind(0)*j + m->gnpoind(0)*m->gnpoind(1)*k;
}

/// Gives the index of a point in the point grid collapsed to 1D, assuming boundary points don't exist
/** Make sure there's at least one interior point, or Bad Things (TM) may happen.
 */
inline PetscInt getFlattenedInteriorIndex(const CartMesh *const m, const PetscInt i, const PetscInt j, const PetscInt k)
{
#if DEBUG==1
	if(i == 0 || i == m->gnpoind(0)-1 || j == 0 || j == m->gnpoind(1)-1 || k == 0 || k == m->gnpoind(2)-1) {
		std::printf("! getFlattenedInteriorIndex(): Invalid i, j, or k index!\n");
		return 0;
	}
#endif
	return i-1 + (m->gnpoind(0)-2)*(j-1) + (m->gnpoind(0)-2)*(m->gnpoind(1)-2)*(k-1);
}

/// Set RHS = 12*pi^2*sin(2pi*x)sin(2pi*y)sin(2pi*z) for u_exact = sin(2pi*x)sin(2pi*y)sin(2pi*z)
/** Note that the values are only set for interior points.
 * \param f is the rhs vector
 * \param uexact is the exact solution
 */
void computeRHS(const CartMesh *const m, Vec f, Vec uexact)
{
	printf("ComputeRHS: Starting\n");
	PetscReal *valuesrhs = (PetscReal*)std::malloc(m->gninpoin()*sizeof(PetscReal));
	PetscReal *valuesuexact = (PetscReal*)std::malloc(m->gninpoin()*sizeof(PetscReal));
	PetscInt *indices = (PetscInt*)std::malloc(m->gninpoin()*sizeof(PetscInt));

	// point ordering index
	PetscInt l = 0;

	// iterate over interior nodes
	for(PetscInt k = 1; k < m->gnpoind(2)-1; k++)
		for(PetscInt j = 1; j < m->gnpoind(1)-1; j++)
			for(PetscInt i = 1; i < m->gnpoind(0)-1; i++)
			{
				indices[l] = getFlattenedInteriorIndex(m,i,j,k);

				valuesrhs[l] = 12.0*PI*PI*std::sin(2*PI*m->gcoords(i,0))*std::sin(2*PI*m->gcoords(j,1))*std::sin(2*PI*m->gcoords(k,2));
				valuesuexact[l] = std::sin(2*PI*m->gcoords(i,0))*std::sin(2*PI*m->gcoords(j,1))*std::sin(2*PI*m->gcoords(k,2));

				l++;
			}

	VecSetValues(f, m->gninpoin(), indices, valuesrhs, INSERT_VALUES);
	VecSetValues(uexact, m->gninpoin(), indices, valuesuexact, INSERT_VALUES);

	std::free(valuesrhs);
	std::free(valuesuexact);
	std::free(indices);
	printf("ComputeRHS: Done\n");
}

/// Set stiffness matrix corresponding to interior points
/** Inserts entries rowwise into the matrix.
 */
void computeLHS(const CartMesh *const m, Mat A)
{
	printf("ComputeLHS: For interior nodes...\n");

	// nodes that don't have a Dirichlet node as a neighbor
	for(PetscInt k = 2; k < m->gnpoind(2)-2; k++)
		for(PetscInt j = 2; j < m->gnpoind(1)-2; j++)
			for(PetscInt i = 2; i < m->gnpoind(0)-2; i++)
			{
				PetscReal values[NSTENCIL];
				PetscInt cindices[NSTENCIL];
				PetscInt rindices[1];
				PetscInt n = NSTENCIL;
				PetscInt mm = 1;

				rindices[0] = getFlattenedInteriorIndex(m,i,j,k);

				cindices[0] = getFlattenedInteriorIndex(m,i-1,j,k);
				cindices[1] = getFlattenedInteriorIndex(m,i,j-1,k);
				cindices[2] = getFlattenedInteriorIndex(m,i,j,k-1);
				cindices[3] = rindices[0];
				cindices[4] = getFlattenedInteriorIndex(m,i+1,j,k);
				cindices[5] = getFlattenedInteriorIndex(m,i,j+1,k);
				cindices[7] = getFlattenedInteriorIndex(m,i,j,k+1);
				
				values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
				values[1] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
				values[2] = -1.0/( (m->gcoords(2,k)-m->gcoords(2,k-1)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

				values[3] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
				values[3] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
				values[3] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

				values[4] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
				values[5] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
				values[6] = -1.0/( (m->gcoords(2,k+1)-m->gcoords(2,k)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

				MatSetValues(A, mm, rindices, n, cindices, values, INSERT_VALUES);
			}

	// next, nodes with Dirichlet nodes as neighbors
	printf("ComputeLHS: For boundary nodes...\n");
	
	PetscInt k = 1;
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
					PetscInt mm = 1;
					
					rindices[0] = getFlattenedInteriorIndex(m,i,j,k);

					cindices[0] = rindices[0];
					cindices[1] = getFlattenedInteriorIndex(m,i+1,j,k);
					cindices[2] = getFlattenedInteriorIndex(m,i,j+1,k);
					cindices[3] = getFlattenedInteriorIndex(m,i,j,k+1);
					
					values[0] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[0] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[0] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[1] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[2] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
					values[3] = -1.0/( (m->gcoords(2,k+1)-m->gcoords(2,k)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					MatSetValues(A, mm, rindices, n, cindices, values, INSERT_VALUES);
				}
				else if(i == m->gnpoind(0)-2)
				{
					PetscReal values[NSTENCIL-3];
					PetscInt cindices[NSTENCIL-3];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-3;
					PetscInt mm = 1;
					
					rindices[0] = getFlattenedInteriorIndex(m,i,j,k);

					cindices[0] = getFlattenedInteriorIndex(m,i-1,j,k);
					cindices[1] = rindices[0];
					cindices[2] = getFlattenedInteriorIndex(m,i,j+1,k);
					cindices[3] = getFlattenedInteriorIndex(m,i,j,k+1);
					
					values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );

					values[1] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[1] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[1] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[2] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
					values[3] = -1.0/( (m->gcoords(2,k+1)-m->gcoords(2,k)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					MatSetValues(A, mm, rindices, n, cindices, values, INSERT_VALUES);
				}
				else
				{
					PetscReal values[NSTENCIL-2];
					PetscInt cindices[NSTENCIL-2];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-2;
					PetscInt mm = 1;
					
					rindices[0] = getFlattenedInteriorIndex(m,i,j,k);

					cindices[0] = getFlattenedInteriorIndex(m,i-1,j,k);
					cindices[1] = rindices[0];
					cindices[2] = getFlattenedInteriorIndex(m,i+1,j,k);
					cindices[3] = getFlattenedInteriorIndex(m,i,j+1,k);
					cindices[4] = getFlattenedInteriorIndex(m,i,j,k+1);
					
					values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );

					values[1] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[1] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[1] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[2] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[3] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
					values[4] = -1.0/( (m->gcoords(2,k+1)-m->gcoords(2,k)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					MatSetValues(A, mm, rindices, n, cindices, values, INSERT_VALUES);
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
					PetscInt mm = 1;
					
					rindices[0] = getFlattenedInteriorIndex(m,i,j,k);

					cindices[0] = getFlattenedInteriorIndex(m,i,j-1,k);
					cindices[1] = rindices[0];
					cindices[2] = getFlattenedInteriorIndex(m,i+1,j,k);
					cindices[3] = getFlattenedInteriorIndex(m,i,j,k+1);
					
					values[0] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );

					values[1] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[1] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[1] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[2] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[3] = -1.0/( (m->gcoords(2,k+1)-m->gcoords(2,k)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					MatSetValues(A, mm, rindices, n, cindices, values, INSERT_VALUES);
				}
				else if(i == m->gnpoind(0)-2)
				{
					PetscReal values[NSTENCIL-3];
					PetscInt cindices[NSTENCIL-3];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-3;
					PetscInt mm = 1;
					
					rindices[0] = getFlattenedInteriorIndex(m,i,j,k);

					cindices[0] = getFlattenedInteriorIndex(m,i-1,j,k);
					cindices[1] = getFlattenedInteriorIndex(m,i,j-1,k);
					cindices[2] = rindices[0];
					cindices[3] = getFlattenedInteriorIndex(m,i,j,k+1);
					
					values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[1] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );

					values[2] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[2] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[2] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[3] = -1.0/( (m->gcoords(2,k+1)-m->gcoords(2,k)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					MatSetValues(A, mm, rindices, n, cindices, values, INSERT_VALUES);
				}
				else
				{
					PetscReal values[NSTENCIL-2];
					PetscInt cindices[NSTENCIL-2];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-2;
					PetscInt mm = 1;
					
					rindices[0] = getFlattenedInteriorIndex(m,i,j,k);

					cindices[0] = getFlattenedInteriorIndex(m,i-1,j,k);
					cindices[1] = getFlattenedInteriorIndex(m,i,j-1,k);
					cindices[2] = rindices[0];
					cindices[3] = getFlattenedInteriorIndex(m,i+1,j,k);
					cindices[4] = getFlattenedInteriorIndex(m,i,j,k+1);
					
					values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[1] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );

					values[2] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[2] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[2] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[3] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[4] = -1.0/( (m->gcoords(2,k+1)-m->gcoords(2,k)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					MatSetValues(A, mm, rindices, n, cindices, values, INSERT_VALUES);
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
					PetscInt mm = 1;
					
					rindices[0] = getFlattenedInteriorIndex(m,i,j,k);

					cindices[0] = getFlattenedInteriorIndex(m,i,j-1,k);
					cindices[1] = rindices[0];
					cindices[2] = getFlattenedInteriorIndex(m,i+1,j,k);
					cindices[3] = getFlattenedInteriorIndex(m,i,j+1,k);
					cindices[4] = getFlattenedInteriorIndex(m,i,j,k+1);
					
					values[0] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );

					values[1] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[1] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[1] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[2] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[3] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
					values[4] = -1.0/( (m->gcoords(2,k+1)-m->gcoords(2,k)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					MatSetValues(A, mm, rindices, n, cindices, values, INSERT_VALUES);
				}
				else if(i == m->gnpoind(0)-2)
				{
					PetscReal values[NSTENCIL-2];
					PetscInt cindices[NSTENCIL-2];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-2;
					PetscInt mm = 1;
					
					rindices[0] = getFlattenedInteriorIndex(m,i,j,k);

					cindices[0] = getFlattenedInteriorIndex(m,i-1,j,k);
					cindices[1] = getFlattenedInteriorIndex(m,i,j-1,k);
					cindices[2] = rindices[0];
					cindices[3] = getFlattenedInteriorIndex(m,i,j+1,k);
					cindices[4] = getFlattenedInteriorIndex(m,i,j,k+1);
					
					values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[1] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );

					values[2] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[2] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[2] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[3] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
					values[4] = -1.0/( (m->gcoords(2,k+1)-m->gcoords(2,k)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					MatSetValues(A, mm, rindices, n, cindices, values, INSERT_VALUES);
				}
				else
				{
					PetscReal values[NSTENCIL-1];
					PetscInt cindices[NSTENCIL-1];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-1;
					PetscInt mm = 1;
					
					rindices[0] = getFlattenedInteriorIndex(m,i,j,k);

					cindices[0] = getFlattenedInteriorIndex(m,i-1,j,k);
					cindices[1] = getFlattenedInteriorIndex(m,i,j-1,k);
					cindices[2] = rindices[0];
					cindices[3] = getFlattenedInteriorIndex(m,i+1,j,k);
					cindices[4] = getFlattenedInteriorIndex(m,i,j+1,k);
					cindices[5] = getFlattenedInteriorIndex(m,i,j,k+1);
					
					values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[1] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );

					values[2] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[2] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[2] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[3] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[4] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
					values[5] = -1.0/( (m->gcoords(2,k+1)-m->gcoords(2,k)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					MatSetValues(A, mm, rindices, n, cindices, values, INSERT_VALUES);
				}
			}

		} // end loop
	
	k = m->gnpoind(2)-2;
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
					PetscInt mm = 1;
					
					rindices[0] = getFlattenedInteriorIndex(m,i,j,k);

					cindices[0] = getFlattenedInteriorIndex(m,i,j,k-1);
					cindices[1] = rindices[0];
					cindices[2] = getFlattenedInteriorIndex(m,i+1,j,k);
					cindices[3] = getFlattenedInteriorIndex(m,i,j+1,k);
					
					values[0] = -1.0/( (m->gcoords(2,k)-m->gcoords(2,k-1)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					values[1] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[1] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[1] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[2] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[3] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );

					MatSetValues(A, mm, rindices, n, cindices, values, INSERT_VALUES);
				}
				else if(i == m->gnpoind(0)-2)
				{
					PetscReal values[NSTENCIL-3];
					PetscInt cindices[NSTENCIL-3];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-3;
					PetscInt mm = 1;
					
					rindices[0] = getFlattenedInteriorIndex(m,i,j,k);

					cindices[0] = getFlattenedInteriorIndex(m,i-1,j,k);
					cindices[1] = getFlattenedInteriorIndex(m,i,j,k-1);
					cindices[2] = rindices[0];
					cindices[3] = getFlattenedInteriorIndex(m,i,j+1,k);
					
					values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[1] = -1.0/( (m->gcoords(2,k)-m->gcoords(2,k-1)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					values[2] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[2] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[2] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[3] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );

					MatSetValues(A, mm, rindices, n, cindices, values, INSERT_VALUES);
				}
				else
				{
					PetscReal values[NSTENCIL-2];
					PetscInt cindices[NSTENCIL-2];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-2;
					PetscInt mm = 1;
					
					rindices[0] = getFlattenedInteriorIndex(m,i,j,k);

					cindices[0] = getFlattenedInteriorIndex(m,i-1,j,k);
					cindices[1] = getFlattenedInteriorIndex(m,i,j,k-1);
					cindices[2] = rindices[0];
					cindices[3] = getFlattenedInteriorIndex(m,i+1,j,k);
					cindices[4] = getFlattenedInteriorIndex(m,i,j+1,k);
					
					values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[1] = -1.0/( (m->gcoords(2,k)-m->gcoords(2,k-1)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					values[2] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[2] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[2] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[3] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[4] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );

					MatSetValues(A, mm, rindices, n, cindices, values, INSERT_VALUES);
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
					PetscInt mm = 1;
					
					rindices[0] = getFlattenedInteriorIndex(m,i,j,k);

					cindices[0] = getFlattenedInteriorIndex(m,i,j-1,k);
					cindices[1] = getFlattenedInteriorIndex(m,i,j,k-1);
					cindices[2] = rindices[0];
					cindices[3] = getFlattenedInteriorIndex(m,i+1,j,k);
					
					values[0] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
					values[1] = -1.0/( (m->gcoords(2,k)-m->gcoords(2,k-1)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					values[2] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[2] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[2] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[3] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );

					MatSetValues(A, mm, rindices, n, cindices, values, INSERT_VALUES);
				}
				else if(i == m->gnpoind(0)-2)
				{
					PetscReal values[NSTENCIL-3];
					PetscInt cindices[NSTENCIL-3];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-3;
					PetscInt mm = 1;
					
					rindices[0] = getFlattenedInteriorIndex(m,i,j,k);

					cindices[0] = getFlattenedInteriorIndex(m,i-1,j,k);
					cindices[1] = getFlattenedInteriorIndex(m,i,j-1,k);
					cindices[2] = getFlattenedInteriorIndex(m,i,j,k-1);
					cindices[3] = rindices[0];
					
					values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[1] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
					values[2] = -1.0/( (m->gcoords(2,k)-m->gcoords(2,k-1)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					values[3] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[3] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[3] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					MatSetValues(A, mm, rindices, n, cindices, values, INSERT_VALUES);
				}
				else
				{
					PetscReal values[NSTENCIL-2];
					PetscInt cindices[NSTENCIL-2];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-2;
					PetscInt mm = 1;
					
					rindices[0] = getFlattenedInteriorIndex(m,i,j,k);

					cindices[0] = getFlattenedInteriorIndex(m,i-1,j,k);
					cindices[1] = getFlattenedInteriorIndex(m,i,j-1,k);
					cindices[2] = getFlattenedInteriorIndex(m,i,j,k-1);
					cindices[3] = rindices[0];
					cindices[4] = getFlattenedInteriorIndex(m,i+1,j,k);
					
					values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[1] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
					values[2] = -1.0/( (m->gcoords(2,k)-m->gcoords(2,k-1)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					values[3] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[3] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[3] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[4] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );

					MatSetValues(A, mm, rindices, n, cindices, values, INSERT_VALUES);
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
					PetscInt mm = 1;
					
					rindices[0] = getFlattenedInteriorIndex(m,i,j,k);

					cindices[0] = getFlattenedInteriorIndex(m,i,j-1,k);
					cindices[1] = getFlattenedInteriorIndex(m,i,j,k-1);
					cindices[2] = rindices[0];
					cindices[3] = getFlattenedInteriorIndex(m,i+1,j,k);
					cindices[4] = getFlattenedInteriorIndex(m,i,j+1,k);
					
					values[0] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
					values[1] = -1.0/( (m->gcoords(2,k)-m->gcoords(2,k-1)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					values[2] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[2] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[2] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[3] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[4] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );

					MatSetValues(A, mm, rindices, n, cindices, values, INSERT_VALUES);
				}
				else if(i == m->gnpoind(0)-2)
				{
					PetscReal values[NSTENCIL-2];
					PetscInt cindices[NSTENCIL-2];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-2;
					PetscInt mm = 1;
					
					rindices[0] = getFlattenedInteriorIndex(m,i,j,k);

					cindices[0] = getFlattenedInteriorIndex(m,i-1,j,k);
					cindices[1] = getFlattenedInteriorIndex(m,i,j-1,k);
					cindices[2] = getFlattenedInteriorIndex(m,i,j,k-1);
					cindices[3] = rindices[0];
					cindices[4] = getFlattenedInteriorIndex(m,i,j+1,k);
					
					values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[1] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
					values[2] = -1.0/( (m->gcoords(2,k)-m->gcoords(2,k-1)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					values[3] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[3] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[3] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[4] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );

					MatSetValues(A, mm, rindices, n, cindices, values, INSERT_VALUES);
				}
				else
				{
					PetscReal values[NSTENCIL-1];
					PetscInt cindices[NSTENCIL-1];
					PetscInt rindices[1];
					PetscInt n = NSTENCIL-1;
					PetscInt mm = 1;
					
					rindices[0] = getFlattenedInteriorIndex(m,i,j,k);

					cindices[0] = getFlattenedInteriorIndex(m,i-1,j,k);
					cindices[1] = getFlattenedInteriorIndex(m,i,j-1,k);
					cindices[2] = getFlattenedInteriorIndex(m,i,j,k-1);
					cindices[3] = rindices[0];
					cindices[4] = getFlattenedInteriorIndex(m,i+1,j,k);
					cindices[5] = getFlattenedInteriorIndex(m,i,j+1,k);
					
					values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[1] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
					values[2] = -1.0/( (m->gcoords(2,k)-m->gcoords(2,k-1)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

					values[3] =  1.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
					values[3] += 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
					values[3] += 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

					values[4] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
					values[5] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );

					MatSetValues(A, mm, rindices, n, cindices, values, INSERT_VALUES);
				}
			}

		} // end loop
	printf("ComputeLHS: Done.\n");
}

//#undef __FUNCT__
//#define __FUNCT__ "main"

int main(int argc, char* argv[])
{
	using namespace std;

	if(argc < 3) {
		printf("Please specify a control file and a Petsc options file.\n");
		return 0;
	}

	char help[] = "Solves 3D Poisson equation by finite differences. Arguments: (1) Control file (2) Petsc options file\n\n";
	char* confile = argv[1];
	char * optfile = argv[2];
	PetscMPIInt size;
	PetscErrorCode ierr;

	ierr = PetscInitialize(&argc, &argv, optfile, help); CHKERRQ(ierr);
	MPI_Comm_size(PETSC_COMM_WORLD,&size);
	if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"Currently single processor only!");

	// Read control file
	PetscInt npdim[NDIM];
	PetscReal rmax[NDIM], rmin[NDIM];
	char temp[50];
	FILE* conf = fopen(confile, "r");
	fscanf(conf, "%s", temp);
	for(int i = 0; i < NDIM; i++)
		fscanf(conf, "%d", &npdim[i]);
	fscanf(conf, "%s", temp);
	for(int i = 0; i < NDIM; i++)
		fscanf(conf, "%lf", &rmin[i]);
	fscanf(conf, "%s", temp);
	for(int i = 0; i < NDIM; i++)
		fscanf(conf, "%lf", &rmax[i]);
	fclose(conf);

	// generate mesh
	CartMesh m(npdim);
	m.generateMesh_ChebyshevDistribution(rmin,rmax);

	// set up Petsc variables
	Vec u, uexact, b, err;
	Mat A;
	KSP ksp; PC pc;

	VecCreateSeq(PETSC_COMM_SELF, m.gninpoin(), &u);
	VecDuplicate(u, &b);
	VecDuplicate(u, &uexact);
	VecDuplicate(u, &err);
	VecSet(u, 0.0);

	MatCreateSeqAIJ(PETSC_COMM_SELF, m.gninpoin(), m.gninpoin(), NSTENCIL, NULL, &A);

	computeRHS(&m, b, uexact);
	computeLHS(&m, A);

	MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
	VecAssemblyBegin(u);
	VecAssemblyBegin(uexact);
	VecAssemblyBegin(b);
	MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
	VecAssemblyEnd(u);
	VecAssemblyEnd(uexact);
	VecAssemblyEnd(b);

	// set up solver
	ierr = KSPCreate(PETSC_COMM_SELF, &ksp);
	ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
	ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
	
	ierr = KSPSolve(ksp, b, u);

	// post-process
	int kspiters; PetscReal errnorm;
	KSPGetIterationNumber(ksp, &kspiters);
	
	VecCopy(u,err);
	VecAXPY(err, -1.0, uexact);
	VecNorm(err,NORM_2,&errnorm);
	printf("log h and log error: %f  %f\n", log10(m.gh()), log10(errnorm));

	KSPDestroy(&ksp);
	VecDestroy(&u);
	VecDestroy(&uexact);
	VecDestroy(&b);
	VecDestroy(&err);
	MatDestroy(&A);
	PetscFinalize();
	return 0;
}
