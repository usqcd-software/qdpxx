// $Id: qdp_scalarvec_specific.cc,v 1.3 2003-09-03 01:25:48 edwards Exp $

/*! @file
 * @brief Scalarvec specific routines
 * 
 * Routines for scalarvec implementation
 */

#include "qdp.h"
#include "qdp_util.h"

QDP_BEGIN_NAMESPACE(QDP);

//-----------------------------------------------------------------------------
//! Initializer for generic map constructor
void Map::make(const MapFunc& func)
{
#if QDP_DEBUG >= 3
  QDP_info("Map::make");
#endif

  //--------------------------------------
  // Setup the communication index arrays
  goffsets.resize(Layout::vol());

  /* Get the offsets needed for neighbour comm.
     * goffsets(position)
     * the offsets contain the current site, i.e the neighbour for site i
     * is  goffsets(i,dir,mu) and NOT  i + goffset(..) 
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
    goffsets[linear] = Layout::linearSiteIndex(fcoord);
  }

#if 0
  for(int ipos=0; ipos < Layout::vol(); ++ipos)
    fprintf(stderr,"goffsets(%d,%d,%d) = %d\n",ipos,goffsets(ipos));
#endif
}



QDP_END_NAMESPACE();
