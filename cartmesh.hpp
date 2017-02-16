#ifndef __CARTMESH_H
#define __CARTMESH_H

#ifndef __DEFINITIONS_H
#include "definitions.hpp"
#endif

/// Non-uniform Cartesian grid
/** Note that memory for storage of coordinates along the axes is allocated on construction.
 */
class CartMesh
{
	PetscInt npoind[NDIM];	///< Array storing the number of points on each coordinate axis
	PetscReal ** coords;			///< Stores an array for each of the 3 axes - coords[i][j] refers to the j-th node along the i-axis
	PetscInt npointotal;			///< Total number of points in the grid
	PetscInt ninpoin;				///< Number of internal (non-boundary) points
	PetscReal h;					///< Mesh size parameter
public:
	CartMesh(const PetscInt npdim[NDIM])
	{
		std::printf("CartMesh: Number of points in each direction: ");
		for(int i = 0; i < NDIM; i++) {
			npoind[i] = npdim[i];
			std::printf("%d ", npoind[i]);
		}
		std::printf("\n");
		
		npointotal = 1;
		for(int i = 0; i < NDIM; i++)
			npointotal *= npoind[i];

		PetscInt nbpoints = npoind[0]*npoind[1]*2 + (npoind[2]-2)*npoind[0]*2 + (npoind[1]-2)*(npoind[2]-2)*2;
		ninpoin = npointotal-nbpoints;

		std::printf("CartMesh: Total points = %d, interior points = %d\n", npointotal, ninpoin);

		coords = (PetscReal**)std::malloc(NDIM*sizeof(PetscReal*));
		for(int i = 0; i < NDIM; i++)
			coords[i] = (PetscReal*)std::malloc(npoind[i]*sizeof(PetscReal));
	}

	~CartMesh()
	{
		for(int i = 0; i < NDIM; i++)
			std::free(coords[i]);
		std::free(coords);
	}

	PetscInt gnpoind(const int idim) const
	{
#if DEBUG == 1
		if(idim >= NDIM) {
			std::printf("! Cartmesh: gnpoind(): Invalid dimension %d!\n", idim);
			return 0;
		}
#endif
		return npoind[idim];
	}

	PetscReal gcoords(const int idim, const PetscInt ipoin) const
	{
#if DEBUG == 1
		if(idim >= NDIM) 
		{
			std::printf("! Cartmesh: gcoords(): Invalid dimension!\n");
			return 0;
		}
		if(ipoin >= npoind[idim]) 
		{
			std::printf("! Cartmesh: gcoords(): Point does not exist!\n");
			return 0;
		}
#endif
		return coords[idim][ipoin];
	}

	PetscInt gnpointotal() const { return npointotal; }
	PetscInt gninpoin() const { return ninpoin; }
	PetscReal gh() const { return h; }

	const PetscInt *const pointer_npoind() const
	{
		return npoind;
	}

	const PetscReal *const *const pointer_coords() const
	{
		return coords;
	}

	/// Generate a non-uniform mesh in a cuboid corresponding to Chebyshev points in each direction
	/** For interval [a,b], a Chebyshev distribution of N points including a and b is computed as
	 * x_i = (a+b)/2 + (a-b)/2 * cos(pi - i*theta)
	 * where theta = pi/(N-1)
	 */
	void generateMesh_ChebyshevDistribution(PetscReal rmin[NDIM], PetscReal rmax[NDIM])
	{
		std::printf("CartMesh: generateMesh_cheb: Points are\n");
		for(int idim = 0; idim < NDIM; idim++)
		{
			PetscReal theta = PI/(npoind[idim]-1);
			for(int i = 0; i < npoind[idim]; i++) {
				coords[idim][i] = (rmax[idim]+rmin[idim])*0.5 + (rmax[idim]-rmin[idim])*0.5*std::cos(PI-i*theta);
				printf("%f ", coords[idim][i]);
			}
			printf("\n");
		}

		// estimate h
		h = 0.0;
		PetscReal hd[NDIM];
		for(int k = 0; k < npoind[2]-1; k++)
		{
			hd[2] = coords[2][k+1]-coords[2][k];
			for(int j = 0; j < npoind[1]-1; j++)
			{
				hd[1] = coords[1][j+1]-coords[1][j];
				for(int i = 0; i < npoind[0]-1; i++)
				{
					hd[0] = coords[0][i+1]-coords[0][i];
					PetscReal diam = 0;
					for(int idim = 0; idim < NDIM; idim++)
						diam += hd[idim]*hd[idim];
					diam = std::sqrt(diam);
					if(diam > h)
						h = diam;
				}
			}
		}
		std::printf("CartMesh: generateMesh_Cheb: h = %f\n", h);
	}
	
	void generateMesh_UniformDistribution(PetscReal rmin[NDIM], PetscReal rmax[NDIM])
	{
		std::printf("CartMesh: generateMesh_Uniform: Points are\n");
		for(int idim = 0; idim < NDIM; idim++)
		{
			for(int i = 0; i < npoind[idim]; i++) {
				coords[idim][i] = rmin[i] + (rmax[i]-rmin[i])*i/(npoind[i]-1);
				printf("%f ", coords[idim][i]);
			}
			printf("\n");
		}

		// estimate h
		h = 0.0;
		PetscReal hd[NDIM];
		for(int k = 0; k < npoind[2]-1; k++)
		{
			hd[2] = coords[2][k+1]-coords[2][k];
			for(int j = 0; j < npoind[1]-1; j++)
			{
				hd[1] = coords[1][j+1]-coords[1][j];
				for(int i = 0; i < npoind[0]-1; i++)
				{
					hd[0] = coords[0][i+1]-coords[0][i];
					PetscReal diam = 0;
					for(int idim = 0; idim < NDIM; idim++)
						diam += hd[idim]*hd[idim];
					diam = std::sqrt(diam);
					if(diam > h)
						h = diam;
				}
			}
		}
		std::printf("CartMesh: generateMesh_Uniform: h = %f\n", h);
	}

};

#endif
