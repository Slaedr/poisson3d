#ifndef __CARTMESH_H
#define __CARTMESH_H

#ifndef __DEFINITIONS_H
#include "definitions.hpp"
#endif

/// Non-uniform Cartesian grid
class CartMesh
{
	const PetscInt npoind[NDIM];
	PetscReal ** coords;			///< Stores an array for each of the 3 axes - coords[i][j] refers to the j-th node along the i-axis
	PetscInt npointotal;
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

	PetscReal gcoords(const PetscInt ipoin, const int idim) const
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
		return coords[ipoin][idim];
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
};
