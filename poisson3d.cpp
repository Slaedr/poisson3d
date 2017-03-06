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

				valuesrhs[l] = 12.0*PI*PI*std::sin(2*PI*m->gcoords(0,i))*std::sin(2*PI*m->gcoords(1,j))*std::sin(2*PI*m->gcoords(2,k));
				valuesuexact[l] = std::sin(2*PI*m->gcoords(0,i))*std::sin(2*PI*m->gcoords(1,j))*std::sin(2*PI*m->gcoords(2,k));

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

	for(PetscInt k = 1; k < m->gnpoind(2)-1; k++)
		for(PetscInt j = 1; j < m->gnpoind(1)-1; j++)
			for(PetscInt i = 1; i < m->gnpoind(0)-1; i++)
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
				cindices[6] = getFlattenedInteriorIndex(m,i,j,k+1);
				
				values[0] = -1.0/( (m->gcoords(0,i)-m->gcoords(0,i-1)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
				values[1] = -1.0/( (m->gcoords(1,j)-m->gcoords(1,j-1)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
				values[2] = -1.0/( (m->gcoords(2,k)-m->gcoords(2,k-1)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

				values[3] =  2.0/(m->gcoords(0,i+1)-m->gcoords(0,i-1))*( 1.0/(m->gcoords(0,i+1)-m->gcoords(0,i))+1.0/(m->gcoords(0,i)-m->gcoords(0,i-1)) );
				values[3] += 2.0/(m->gcoords(1,j+1)-m->gcoords(1,j-1))*( 1.0/(m->gcoords(1,j+1)-m->gcoords(1,j))+1.0/(m->gcoords(1,j)-m->gcoords(1,j-1)) );
				values[3] += 2.0/(m->gcoords(2,k+1)-m->gcoords(2,k-1))*( 1.0/(m->gcoords(2,k+1)-m->gcoords(2,k))+1.0/(m->gcoords(2,k)-m->gcoords(2,k-1)) );

				values[4] = -1.0/( (m->gcoords(0,i+1)-m->gcoords(0,i)) * 0.5*(m->gcoords(0,i+1)-m->gcoords(0,i-1)) );
				values[5] = -1.0/( (m->gcoords(1,j+1)-m->gcoords(1,j)) * 0.5*(m->gcoords(1,j+1)-m->gcoords(1,j-1)) );
				values[6] = -1.0/( (m->gcoords(2,k+1)-m->gcoords(2,k)) * 0.5*(m->gcoords(2,k+1)-m->gcoords(2,k-1)) );

				MatSetValues(A, mm, rindices, n, cindices, values, INSERT_VALUES);
				//printf("\tProcessed index %d, diag value = %f\n", rindices[0], values[3]);
			}

	printf("ComputeLHS: Done.\n");
}

/// Computes L2 norm of a mesh function v, assuming piecewise constant values in a dual cell around each node.
PetscReal computeNorm(Vec v, const CartMesh *const m)
{
	PetscInt sz;
	PetscReal * vals, norm = 0;

	VecGetArray(v, &vals);
	VecGetLocalSize(v, &sz);

	for(int k = 1; k < m->gnpoind(2)-1; k++)
		for(int j = 1; j < m->gnpoind(1)-1; j++)
			for(int i = 1; i < m->gnpoind(0)-1; i++)
			{
				PetscReal vol = 1.0/8.0*(m->gcoords(0,i+1)-m->gcoords(0,i-1))*(m->gcoords(1,j+1)-m->gcoords(1,j-1))*(m->gcoords(2,k+1)-m->gcoords(2,k-1));
				PetscInt ind = getFlattenedInteriorIndex(m,i,j,k);
				norm += vals[ind]*vals[ind]*vol;
			}

	VecRestoreArray(v, &vals);
	return norm;
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
	int nbsw, nasw; char fgpiluch, chtemp;
#ifdef USE_HIPERSOLVER
	H_ILU_data iluctrl;
	printf("Hipersolver available.\n");
#endif

	ierr = PetscInitialize(&argc, &argv, optfile, help); CHKERRQ(ierr);
	MPI_Comm_size(PETSC_COMM_WORLD,&size);

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
	printf("Use FGPILU? %c\n", fgpiluch);
	if(fgpiluch=='y') {
		fscanf(conf, "%s", temp); fscanf(conf, "%d", &nbsw);
		fscanf(conf, "%s", temp); fscanf(conf, "%d", &nasw);
	}
	fclose(conf);

	printf("Domain boundaries in each dimension:\n");
	for(int i = 0; i < NDIM; i++)
		printf("%f %f ", rmin[i], rmax[i]);
	printf("\n");

	// generate mesh
	CartMesh m(npdim, 1);
	if(!strcmp(gridtype, "chebyshev"))
		m.generateMesh_ChebyshevDistribution(rmin,rmax);
	else
		m.generateMesh_UniformDistribution(rmin,rmax);

	// set up Petsc variables
	DM * da;					///< Distributed array context for the cart grid
	PetscInt ndofpernode = 1;
	PetscInt stencil_width = 1;
	DMBoundaryType bx = DMDA_BOUNDARY_GHOST;
	DMBoundaryType by = DMDA_BOUNDARY_GHOST;
	DMBoundaryType bz = DMDA_BOUNDARY_GHOST;
	DMDAStencilType stencil_type = DMDA_STENCIL_STAR;

	Vec u, uexact, b, err;
	Mat A, Ap;
	KSP ksp; PC pc;

	DMCreateGlobalVector(da, &u);
	VecDuplicate(u, &b);
	VecDuplicate(u, &uexact);
	VecDuplicate(u, &err);
	VecSet(u, 0.0);

	MatCreateSeqAIJ(PETSC_COMM_WORLD, m.gninpoin(), m.gninpoin(), NSTENCIL, NULL, &A);

	computeRHS(&m, b, uexact);
	computeLHS(&m, A);

	MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
	VecAssemblyBegin(uexact);
	VecAssemblyBegin(b);
	MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
	VecAssemblyEnd(uexact);
	VecAssemblyEnd(b);

	// create a copy of A for the preconditioner
	MatConvert(A, MATSAME, MAT_INITIAL_MATRIX, &Ap);

	/*printf("Assembled RHS vector:\n");
	VecView(b, 0);
	printf("Assembled LHS matrix:\n");
	MatView(A, 0);*/

	// set up solver
	/** Note that the Richardson solver with preconditioner is nothing but the preconditioner applied iteratively in
	 * approximate factorization (which I also call error correction) form.
	 * Without preconditioner, it is $ \Delta x^k = r^k $ where r is the residual.
	 * In PETSc, it is actually a "modified" Richardson iteration: $ \Delta x^k = \omega r^k $ where omega is a relaxation parameter.
	 */
	ierr = KSPCreate(PETSC_COMM_SELF, &ksp);
	ierr = KSPSetOperators(ksp, A, Ap); CHKERRQ(ierr);
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

	/*PetscInt iter; PetscReal rnor; PetscViewer viewer;
	PetscViewerAndFormatCreate(viewer, PETSC_VIEWER_ASCII_COMMON, &vf);
	KSPMonitorDefault(ksp, iter, rnor, vf);*/
	
	ierr = KSPSolve(ksp, b, u);
	ierr = KSPView(ksp,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);		// change SELF to WORLD for multiprocessor

	// post-process
	int kspiters; PetscReal errnorm, rnorm;
	KSPGetIterationNumber(ksp, &kspiters);
	printf("Number of KSP iterations = %d\n", kspiters);
	KSPGetResidualNorm(ksp, &rnorm);
	printf("KSP residual norm = %f\n", rnorm);
	
	VecCopy(u,err);
	VecAXPY(err, -1.0, uexact);
	errnorm = computeNorm(err,&m);
	printf("h and error: %f  %f\n", m.gh(), errnorm);
	printf("log h and log error: %f  %f\n", log10(m.gh()), log10(errnorm));

#ifdef USE_HIPERSOLVER
	if(fgpiluch == 'y')
		cleanup_fgpilu(pc);
#endif
	KSPDestroy(&ksp);
	VecDestroy(&u);
	VecDestroy(&uexact);
	VecDestroy(&b);
	VecDestroy(&err);
	MatDestroy(&A);
	MatDestroy(&Ap);
	DMDestroy(&da);
	PetscFinalize();
	return 0;
}
