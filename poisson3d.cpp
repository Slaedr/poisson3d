/** \file poisson3d.cpp
 * \brief PETSc-based solver for Poisson Dirichlet problem on a Cartesian grid
 *
 * Note that only zero Dirichlet BCs are currently supported.
 */

#ifndef __CARTMESH_H
#include "cartmesh.hpp"
#endif

#ifdef USE_HIPERSOLVER
#ifndef __FGPILU_H
#include <fgpilu.h>
#endif
#endif

/// Gives the index of a point in the point grid collapsed to 1D
inline PetscInt getFlattenedIndex(const CartMesh *const m, const PetscInt i, const PetscInt j, const PetscInt k)
{
	return i + m->gnpoind(0)*j + m->gnpoind(0)*m->gnpoind(1)*k;
}

/// Gives the index of a point in the point grid collapsed to 1D, assuming boundary points don't exist
/** Returns -1 when passed a boundary point.
 * Make sure there's at least one interior point, or Bad Things (TM) may happen.
 */
inline PetscInt getFlattenedInteriorIndex(const CartMesh *const m, const PetscInt i, const PetscInt j, const PetscInt k)
{
	PetscInt retval = i-1 + (m->gnpoind(0)-2)*(j-1) + (m->gnpoind(0)-2)*(m->gnpoind(1)-2)*(k-1);
	if(i == 0 || i == m->gnpoind(0)-1 || j == 0 || j == m->gnpoind(1)-1 || k == 0 || k == m->gnpoind(2)-1) {
		//std::printf("getFlattenedInteriorIndex(): i, j, or k index corresponds to boundary node. Flattened index = %d, returning -1\n", retval);
		return -1;
	}
	return retval;
}

/// Set RHS = 12*pi^2*sin(2pi*x)sin(2pi*y)sin(2pi*z) for u_exact = sin(2pi*x)sin(2pi*y)sin(2pi*z)
/** Note that the values are only set for interior points.
 * \param f is the rhs vector
 * \param uexact is the exact solution
 */
void computeRHS(const CartMesh *const m, DM da, PetscMPIInt rank, Vec f, Vec uexact)
{
	if(rank == 0)
		printf("ComputeRHS: Starting\n");

	// get the starting global indices and sizes (in each direction) of the local mesh partition
	PetscInt start[NDIM], lsize[NDIM];
	DMDAGetCorners(da, &start[0], &start[1], &start[2], &lsize[0], &lsize[1], &lsize[2]);

	// get local data that can be accessed by global indices
	PetscReal *** rhs, *** uex;
	DMDAVecGetArray(da, f, (void*)&rhs);
	DMDAVecGetArray(da, uexact, (void*)&uex);

	// iterate over interior nodes
	for(PetscInt k = start[2]; k < start[2]+lsize[2]; k++)
		for(PetscInt j = start[1]; j < start[1]+lsize[1]; j++)
			for(PetscInt i = start[0]; i < start[0]+lsize[0]; i++)
			{
				rhs[k][j][i] = 12.0*PI*PI*std::sin(2*PI*m->gcoords(0,i))*std::sin(2*PI*m->gcoords(1,j))*std::sin(2*PI*m->gcoords(2,k));
				uex[k][j][i] = std::sin(2*PI*m->gcoords(0,i))*std::sin(2*PI*m->gcoords(1,j))*std::sin(2*PI*m->gcoords(2,k));
			}
	
	DMDAVecRestoreArray(da, f, (void*)&rhs);
	DMDAVecRestoreArray(da, uexact, (void*)&uex);
	if(rank == 0)
		printf("ComputeRHS: Done\n");
}

/// Set stiffness matrix corresponding to interior points
/** Inserts entries rowwise into the matrix.
 */
void computeLHS(const CartMesh *const m, DM da, PetscMPIInt rank, Mat A)
{
	if(rank == 0)	
		printf("ComputeLHS: Setting values of the LHS matrix...\n");

	// get the starting global indices and sizes (in each direction) of the local mesh partition
	PetscInt start[NDIM], lsize[NDIM];
	DMDAGetCorners(da, &start[0], &start[1], &start[2], &lsize[0], &lsize[1], &lsize[2]);

	for(PetscInt k = start[2]; k < start[2]+lsize[2]; k++)
		for(PetscInt j = start[1]; j < start[1]+lsize[1]; j++)
			for(PetscInt i = start[0]; i < start[0]+lsize[0]; i++)
			{
				PetscReal values[NSTENCIL];
				MatStencil cindices[NSTENCIL];
				MatStencil rindices[1];
				PetscInt n = NSTENCIL;
				PetscInt mm = 1;

				rindices[0] = {k,j,i,0};

				cindices[0] = {k,j,i-1,0};
				cindices[1] = {k,j-1,i,0};
				cindices[2] = {k-1,j,i,0};
				cindices[3] = {k,j,i,0};
				cindices[4] = {k,j,i+1,0};
				cindices[5] = {k,j+1,i,0};
				cindices[6] = {k+1,j,i,0};

				PetscInt I = i+1, J = j+1, K = k+1;		// 1-offset indices for mesh coords access
				
				values[0] = -1.0/( (m->gcoords(0,I)-m->gcoords(0,I-1)) * 0.5*(m->gcoords(0,I+1)-m->gcoords(0,I-1)) );
				values[1] = -1.0/( (m->gcoords(1,J)-m->gcoords(1,J-1)) * 0.5*(m->gcoords(1,J+1)-m->gcoords(1,J-1)) );
				values[2] = -1.0/( (m->gcoords(2,K)-m->gcoords(2,K-1)) * 0.5*(m->gcoords(2,K+1)-m->gcoords(2,K-1)) );

				values[3] =  2.0/(m->gcoords(0,I+1)-m->gcoords(0,I-1))*( 1.0/(m->gcoords(0,I+1)-m->gcoords(0,I))+1.0/(m->gcoords(0,I)-m->gcoords(0,I-1)) );
				values[3] += 2.0/(m->gcoords(1,J+1)-m->gcoords(1,J-1))*( 1.0/(m->gcoords(1,J+1)-m->gcoords(1,J))+1.0/(m->gcoords(1,J)-m->gcoords(1,J-1)) );
				values[3] += 2.0/(m->gcoords(2,K+1)-m->gcoords(2,K-1))*( 1.0/(m->gcoords(2,K+1)-m->gcoords(2,K))+1.0/(m->gcoords(2,K)-m->gcoords(2,K-1)) );

				values[4] = -1.0/( (m->gcoords(0,I+1)-m->gcoords(0,I)) * 0.5*(m->gcoords(0,I+1)-m->gcoords(0,I-1)) );
				values[5] = -1.0/( (m->gcoords(1,J+1)-m->gcoords(1,J)) * 0.5*(m->gcoords(1,J+1)-m->gcoords(1,J-1)) );
				values[6] = -1.0/( (m->gcoords(2,K+1)-m->gcoords(2,K)) * 0.5*(m->gcoords(2,K+1)-m->gcoords(2,K-1)) );

				MatSetValuesStencil(A, mm, rindices, n, cindices, values, INSERT_VALUES);
				//if(rank == 0)
				//	printf("\tProcessed index %d, diag value = %f\n", rindices[0], values[3]);
			}

	if(rank == 0)
		printf("ComputeLHS: Done.\n");
}

/// Computes L2 norm of a mesh function v, assuming piecewise constant values in a dual cell around each node.
/** Note that the actual norm will only be returned by process 0; the other processes return only local norms.
 */
PetscReal computeNorm(const CartMesh *const m, Vec v, DM da)
{
	// get the starting global indices and sizes (in each direction) of the local mesh partition
	PetscInt start[NDIM], lsize[NDIM];
	DMDAGetCorners(da, &start[0], &start[1], &start[2], &lsize[0], &lsize[1], &lsize[2]);
	
	// get local data that can be accessed by global indices
	PetscReal *** vv;
	DMDAVecGetArray(da, v, &vv);

	PetscReal norm = 0, global_norm = 0;

	for(PetscInt k = start[2]; k < start[2]+lsize[2]; k++)
		for(PetscInt j = start[1]; j < start[1]+lsize[1]; j++)
			for(PetscInt i = start[0]; i < start[0]+lsize[0]; i++)
			{
				PetscReal vol = 1.0/8.0*(m->gcoords(0,i+1)-m->gcoords(0,i-1))*(m->gcoords(1,j+1)-m->gcoords(1,j-1))*(m->gcoords(2,k+1)-m->gcoords(2,k-1));
				norm += vv[k][j][i]*vv[k][j][i]*vol;
			}

	DMDAVecRestoreArray(da, v, &vv);

	MPI_Barrier(PETSC_COMM_WORLD);

	// get global norm
	MPI_Reduce(&norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);

	return global_norm;
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
	PetscMPIInt size, rank;
	PetscErrorCode ierr;
	int nbsw, nasw; char fgpiluch, chtemp;
#ifdef USE_HIPERSOLVER
	H_ILU_data iluctrl;
	if(rank == 0)
		printf("Hipersolver available.\n");
#endif

	ierr = PetscInitialize(&argc, &argv, optfile, help); CHKERRQ(ierr);
	MPI_Comm_size(PETSC_COMM_WORLD,&size);
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

	// Read control file
	
	PetscInt npdim[NDIM];
	PetscReal rmax[NDIM], rmin[NDIM];
	char temp[50], gridtype[50];
	FILE* conf = fopen(confile, "r");
	fscanf(conf, "%s", temp);
	fscanf(conf, "%s", gridtype);
	fscanf(conf, "%s", temp);
	for(int i = 0; i < NDIM; i++)
		fscanf(conf, "%d", &npdim[i]);
	fscanf(conf, "%s", temp);
	for(int i = 0; i < NDIM; i++)
		fscanf(conf, "%lf", &rmin[i]);
	fscanf(conf, "%s", temp);
	for(int i = 0; i < NDIM; i++)
		fscanf(conf, "%lf", &rmax[i]);
	fscanf(conf, "%s", temp); fscanf(conf, "%c", &chtemp); fscanf(conf, "%c", &fgpiluch);
	if(fgpiluch=='y') {
		fscanf(conf, "%s", temp); fscanf(conf, "%d", &nbsw);
		fscanf(conf, "%s", temp); fscanf(conf, "%d", &nasw);
	}
	fclose(conf);

	if(rank == 0) {
		printf("Use FGPILU? %c\n", fgpiluch);
		printf("Domain boundaries in each dimension:\n");
		for(int i = 0; i < NDIM; i++)
			printf("%f %f ", rmin[i], rmax[i]);
		printf("\n");
	}
	//----------------------------------------------------------------------------------

	// set up Petsc variables
	DM da;					///< Distributed array context for the cart grid
	PetscInt ndofpernode = 1;
	PetscInt stencil_width = 1;
	DMBoundaryType bx = DM_BOUNDARY_GHOSTED;
	DMBoundaryType by = DM_BOUNDARY_GHOSTED;
	DMBoundaryType bz = DM_BOUNDARY_GHOSTED;
	DMDAStencilType stencil_type = DMDA_STENCIL_STAR;

	// generate mesh - a copy of the mesh is stored by all processes as the mesh structure is very small
	CartMesh m(npdim, ndofpernode, stencil_width, bx, by, bz, stencil_type, &da, rank);
	if(!strcmp(gridtype, "chebyshev"))
		m.generateMesh_ChebyshevDistribution(rmin,rmax, rank);
	else
		m.generateMesh_UniformDistribution(rmin,rmax, rank);

	Vec u, uexact, b, err;
	Mat A;
	KSP ksp; PC pc;

	DMCreateGlobalVector(da, &u);
	VecDuplicate(u, &b);
	VecDuplicate(u, &uexact);
	VecDuplicate(u, &err);
	VecSet(u, 0.0);

	DMCreateMatrix(da, &A);

	// compute values of LHS, RHS and exact soln
	
	computeRHS(&m, da, rank, b, uexact);
	computeLHS(&m, da, rank, A);

	// Assemble LHS

	MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

	// create a copy of A for the preconditioner
	//MatConvert(A, MATSAME, MAT_INITIAL_MATRIX, &Ap);
	//DMCreateMatrix(da, &Ap);
	//MatCopy(A, Ap, SAME_NONZERO_PATTERN);

	/*printf("Assembled RHS vector:\n");
	VecView(b, 0);
	printf("Assembled LHS matrix:\n");
	MatView(A, 0);*/

	// set up solver
	/** Note that the Richardson solver with preconditioner is nothing but the preconditioner applied iteratively in
	 * approximate factorization (which I also call error correction) form.
	 * Without preconditioner, it is \f$ \Delta x^k = r^k \f$ where r is the residual.
	 * In PETSc, it is actually a "modified" Richardson iteration: \f$ \Delta x^k = \omega r^k \f$ where omega is a relaxation parameter.
	 */
	ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);
	ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
	KSPSetType(ksp, KSPRICHARDSON);
	//KSPSetType(ksp, KSPBCGS);
	KSPRichardsonSetScale(ksp, 1.0);
	KSPSetTolerances(ksp, 1e-5, PETSC_DEFAULT, PETSC_DEFAULT, 100);
	KSPGetPC(ksp, &pc);

#ifdef USE_HIPERSOLVER
	if(fgpiluch != 'y') {
#endif
		PCSetType(pc, PCSOR);
		PCSORSetOmega(pc,1.0);
		PCSORSetIterations(pc, 1, 1);
		ierr = PCSORSetSymmetric(pc, SOR_LOCAL_SYMMETRIC_SWEEP); CHKERRQ(ierr);
#ifdef USE_HIPERSOLVER
	}
	else {
		if(rank == 0)
			printf("Using FGPILU as preconditioner.\n");
		PCSetType(pc, PCSHELL);
		iluctrl.nbuildsweeps = nbsw;
		iluctrl.napplysweeps = nasw;
		iluctrl.setup = false;
		PCShellSetContext(pc, &iluctrl);
		//PCShellSetSetUp(pc, &compute_fgpilu_local);
		PCShellSetApply(pc, &apply_fgpilu_jacobi_local);
		//PCShellSetDestroy(pc, &cleanup_fgpilu);
		PCShellSetName(pc, "FGPILU");
		compute_fgpilu_local(pc);
	}
#endif
	ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
	
	ierr = KSPSolve(ksp, b, u);
	ierr = KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

	// post-process
	if(rank == 0) {
		int kspiters; PetscReal rnorm;
		KSPGetIterationNumber(ksp, &kspiters);
		printf("Number of KSP iterations = %d\n", kspiters);
		KSPGetResidualNorm(ksp, &rnorm);
		printf("KSP residual norm = %f\n", rnorm);
	}
	
	PetscReal errnorm;
	VecCopy(u,err);
	VecAXPY(err, -1.0, uexact);
	errnorm = computeNorm(&m, err, da);
	if(rank == 0) {
		printf("h and error: %f  %f\n", m.gh(), errnorm);
		printf("log h and log error: %f  %f\n", log10(m.gh()), log10(errnorm));
	}

#ifdef USE_HIPERSOLVER
	if(fgpiluch == 'y') {
		cleanup_fgpilu(pc);
	}
#endif
	KSPDestroy(&ksp);
	VecDestroy(&u);
	VecDestroy(&uexact);
	VecDestroy(&b);
	VecDestroy(&err);
	MatDestroy(&A);
	DMDestroy(&da);
	PetscFinalize();
	return 0;
}
