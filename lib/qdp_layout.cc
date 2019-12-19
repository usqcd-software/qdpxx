
/*! @file
* @brief Layout support routines
*
* The layout routines provide various layouts. Most of this
* is architecture dependent, so lives in the ${arch}_specific.cc codes.
* The routines here are some auxilliary routines.
*/

#include "qdp.h"
#include "qdp_util.h"

namespace QDP {

	namespace Layout
	{
		//! Returns the logical node coordinates for the corresponding lattice coordinate
		multi1d<index_t> nodeCoord(const multi1d<index_t>& coord) 
		{
			multi1d<index_t> logical(Nd);

			for(index_t i=0; i < Nd; ++i)
				logical[i] = coord[i] / subgridLattSize()[i];

			return logical;
		}

		multi1d<index_t> localLexiCoordFromLinear(const index_t& linearr){
			multi1d<index_t> physcoord(Nd);
			index_t linear=linearr;

			for(index_t i=0; i < Nd; ++i){
				physcoord[i] = linear % subgridLattSize()[i];
				linear=(linear-physcoord[i])/subgridLattSize()[i];
			}

			return physcoord;
		}

		extern "C" { 

			/* Export this to "C" */
			void QDPXX_getSiteCoords(index_t coord[], index_t node, index_t linear) QDP_CONST {
				multi1d<index_t> wrapped_coords = siteCoords(node,linear);
				for(index_t i=0; i < Nd; i++) { 
					coord[i] = wrapped_coords[i];
				}
			}
    
			index_t QDPXX_getLinearSiteIndex(const index_t coord[]) {
				multi1d<index_t> wrapped_coords(Nd);
				for(index_t i=0; i < Nd; i++) { 
					wrapped_coords[i]=coord[i];
				}
				return linearSiteIndex(wrapped_coords);
			}

			index_t QDPXX_nodeNumber(const index_t coord[]) {
				multi1d<index_t> wrapped_coords(Nd);
				for(index_t i=0; i < Nd; i++) { 
					wrapped_coords[i]=coord[i];
				}
				return nodeNumber(wrapped_coords);
			}
		}
	}


#if QDP_USE_CB3D_LAYOUT == 1

	multi1d<index_t> crtesn(index_t ipos, const multi1d<index_t>& latt_size)
	{
		multi1d<index_t> coord(latt_size.size());
		index_t Ndim=latt_size.size() - 1; // Last elem latt size
    
		/* Calculate the Cartesian coordinates of the VALUE of IPOS where the 
		* value is defined by
		*
		*     for i = 0 to NDIM-1  {
		*        X_i  <- mod( IPOS, L(i) )
		*        IPOS <- index_t( IPOS / L(i) )
		*     }
		*
		* NOTE: here the coord(i) and IPOS have their origin at 0. 
		*/
		for(index_t i = Ndim; i < Ndim+latt_size.size(); ++i)
		{
			index_t ix=i%latt_size.size();

			coord[ix] = ipos % latt_size[ix];
			ipos = ipos / latt_size[ix];
		}

		return coord;
	}
  
	//! Calculates the lexicographic site index from the coordinate of a site
	/*! 
	* Nothing specific about the actual lattice size, can be used for 
	* any kind of latt size 
	*/
	index_t local_site(const multi1d<index_t>& coord, const multi1d<index_t>& latt_size)
	{
		index_t order = 0;

		// In the 4D Case: t+Lt(x + Lx(y + Ly*z)
		// essentially  starting from i = dim[Nd-2]
		//  order =  latt_size[i-1]*(coord[i])
		//   and need to wrap i-1 around to Nd-1 when it gets below 0
		for(index_t mmu=latt_size.size()-2; mmu >= 0; --mmu) {
			index_t wrapmu = (mmu-1) % latt_size.size();
			if ( wrapmu < 0 ) wrapmu += latt_size.size();
			order = latt_size[wrapmu]*(coord[mmu] + order);
		}

		order += coord[ latt_size.size()-1 ];

		return order;
	}


#else
	// Usual lattice decomposition -- x fastest, t slowest
	//! Decompose a lexicographic site index_to coordinates
	multi1d<index_t> crtesn(index_t ipos, const multi1d<index_t>& latt_size)
	{
		multi1d<index_t> coord(latt_size.size());

		/* Calculate the Cartesian coordinates of the VALUE of IPOS where the 
		* value is defined by
		*
		*     for i = 0 to NDIM-1  {
		*        X_i  <- mod( IPOS, L(i) )
		*        IPOS <- index_t( IPOS / L(i) )
		*     }
		*
		* NOTE: here the coord(i) and IPOS have their origin at 0. 
		*/
		for(index_t i=0; i < latt_size.size(); ++i)
		{
			coord[i] = ipos % latt_size[i];
			ipos = ipos / latt_size[i];
		}

		return coord;
	}
  
	//! Calculates the lexicographic site index from the coordinate of a site
	/*! 
	* Nothing specific about the actual lattice size, can be used for 
	* any kind of latt size 
	*/
	index_t local_site(const multi1d<index_t>& coord, const multi1d<index_t>& latt_size)
	{
		index_t order = 0;

		for(index_t mmu=latt_size.size()-1; mmu >= 1; --mmu)
			order = latt_size[mmu-1]*(coord[mmu] + order);

		order += coord[0];

		return order;
	}

#endif



} // namespace QDP;
