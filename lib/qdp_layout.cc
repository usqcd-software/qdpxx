// $Id: qdp_layout.cc,v 1.4 2007-06-10 14:32:11 edwards Exp $

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
  multi1d<int> nodeCoord(const multi1d<int>& coord) 
  {
    multi1d<int> logical(Nd);

    for(int i=0; i < Nd; ++i)
      logical[i] = coord[i] / subgridLattSize()[i];

    return logical;
  }
}


//! Decompose a lexicographic site into coordinates
multi1d<int> crtesn(int ipos, const multi1d<int>& latt_size)
{
  multi1d<int> coord(latt_size.size());

  /* Calculate the Cartesian coordinates of the VALUE of IPOS where the 
   * value is defined by
   *
   *     for i = 0 to NDIM-1  {
   *        X_i  <- mod( IPOS, L(i) )
   *        IPOS <- int( IPOS / L(i) )
   *     }
   *
   * NOTE: here the coord(i) and IPOS have their origin at 0. 
   */
  for(int i=0; i < latt_size.size(); ++i)
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
int local_site(const multi1d<int>& coord, const multi1d<int>& latt_size)
{
  int order = 0;

  for(int mmu=latt_size.size()-1; mmu >= 1; --mmu)
    order = latt_size[mmu-1]*(coord[mmu] + order);

  order += coord[0];

  return order;
}


} // namespace QDP;
