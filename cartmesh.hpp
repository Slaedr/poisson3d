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
	const PetscInt npoind[NDIM];	///< Array storing the number of points on each coordinate axis
	PetscReal ** coords;			///< Stores an array for each of the 3 axes - coords[i][j] refers to the j-th node along the i-axis
	PetscInt npointotal;			///< Total number of points in the grid
public:
	CartMesh(const PetscInt npdim[NDIM]) : npoind(npdim)
	{
		npointotal = 1;
		for(int i = 0; i < NDIM; i++)
			npointotal *= npoind[i];

		coords = std::malloc(NDIM*sizeof(PetscReal*));
		for(int i = 0; i < NDIM; i++)
			coords[i] = std::malloc(npoind[i]*sizeof(PetscReal));
	}

	~CartMesh()
	{
		for(int i = 0; i < NDIM; i++)
			free(coords[i]);
		free(coords);
	}

	PetscInt gnpoind(const int idim) const
	{
#if DEBUG == 1
		if(idim >= NDIM) {
			std::printf("! Cartmesh: gnpoindim(): Invalid dimension!\n");
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
			std::printf("! Cartmesh: gnpoindim(): Invalid dimension!\n");
			return 0;
		}
		if(ipoin >= npoind[idim]) 
		{
			std::printf("! Cartmesh: gnpoindim(): Point does not exist!\n");
			return 0;
		}
#endif
		return coords[idim][ipoin];
	}

	PetscInt gnpointotal() const { return npointotal; }

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
		for(int idim = 0; idim < NDIM; idim++)
		{
			PetscReal theta = PI/(npoind[idim]-1);
			for(int i = 0; i < npoind[idim]; i++)
				coords[idim][i] = (rmax[idim]+rmin[idim])*0.5 + (rmax[idim]-rmin[idim])*0.5*std::cos(PI-i*theta);
		}
	}

};
