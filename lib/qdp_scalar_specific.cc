// $Id: qdp_scalar_specific.cc,v 1.3 2003-07-17 01:47:42 edwards Exp $

/*! @file
 * @brief Scalar specific routines
 * 
 * Routines for scalar implementation
 */

#include "qdp.h"
#include "qdp_util.h"

QDP_BEGIN_NAMESPACE(QDP);

//-----------------------------------------------------------------------------
//! Initializer for generic map constructor
void Map::make(const MapFunc& func)
{
//  QDP_info("Map::make");

  //--------------------------------------
  // Setup the communication index arrays
  soffsets.resize(Layout::vol());

  /* Get the offsets needed for neighbour comm.
     * soffsets(position)
     * the offsets contain the current site, i.e the neighbour for site i
     * is  soffsets(i,dir,mu) and NOT  i + soffset(..) 
     */
  const multi1d<int>& nrow = Layout::lattSize();

  // Loop over the sites on this node
  for(int linear=0; linear < Layout::vol(); ++linear)
  {
    // Get the true lattice coord of this linear site index
    multi1d<int> coord = Layout::siteCoords(0, linear);

    // Source neighbor for this destination site
    multi1d<int> fcoord = func(coord,+1);

    // Source linear site and node
    soffsets[linear] = Layout::linearSiteIndex(fcoord);
  }

#if 0
  for(int ipos=0; ipos < Layout::vol(); ++ipos)
    fprintf(stderr,"soffsets(%d,%d,%d) = %d\n",ipos,soffsets(ipos));
#endif
}



QDP_END_NAMESPACE();
