// $Id: scalar_specific.cc,v 1.11 2003-04-02 21:27:43 edwards Exp $

/*! @file
 * @brief Scalar specific routines
 * 
 * Routines for scalar implementation
 */

#include "qdp.h"
#include "qdp_util.h"

QDP_BEGIN_NAMESPACE(QDP);

//-----------------------------------------------------------------------------
//! Constructor from a function object
void Set::make(const SetFunc& func)
{
  int nsubset_indices = func.numSubsets();

#if 0
  fprintf(stderr,"Set a subset: nsubset = %d\n",nsubset_indices);
#endif

  // This actually allocates the subsets
  sub.resize(nsubset_indices);

  // Create the space of the colorings of the lattice
  /*! Loop over all sites determining their color */
  lat_color.resize(Layout::vol());

  // Create the array holding the array of sitetable info
  sitetables.resize(nsubset_indices);

  for(int site=0; site < Layout::vol(); ++site)
  {
    const multi1d<int> coord = crtesn(site, Layout::lattSize());
    int linear = Layout::linearSiteIndex(coord);
    int icolor = func(coord);

    lat_color[linear] = icolor;
  }

  /*
   * Loop over the lexicographic sites.
   * Check if the linear sites are in a contiguous set.
   * This implementation only supports a single contiguous
   * block of sites.
   */
  for(int cb=0; cb < nsubset_indices; ++cb)
  {
    bool indexrep = false;
    int start = 0;
    int end = -1;
    int ntotal = 0;
    int prev = -1;
    bool found_gap = false;

    for(int linear=0; linear < Layout::vol(); ++linear)
    {
      int lexico = Layout::lexicoSiteIndex(linear);
      multi1d<int> coord = crtesn(lexico, Layout::lattSize());

      int icolor = lat_color[linear];

      if (icolor != cb) continue;

      if (ntotal > 0)
      {
	if (prev == linear-1)
	  end = linear;
	else
	{
	  found_gap = true;
	  break;
	}
      }
      else
      {
	start = end = prev = linear;
      }

      prev = linear;
      ++ntotal;
    }

    // Always construct the sitetables. This could be moved into
    // the found_gap and only initialized if the interval method 
    // was not possible

    // First loop and see how many sites are needed
    int num_sitetable = 0;
    for(int linear=0; linear < Layout::vol(); ++linear)
      if (lat_color[linear] == cb)
	++num_sitetable;

    // Now take the inverse of the lattice coloring to produce
    // the site list
    multi1d<int>& sitetable = sitetables[cb];
    sitetable.resize(num_sitetable);

    for(int linear=0, j=0; linear < Layout::vol(); ++linear)
      if (lat_color[linear] == cb)
	sitetable[j++] = linear;


    // If a gap is found, then resort to a site table lookup
    if (found_gap)
    {
      start = 0;
      end = -1;
      indexrep = true;
    }

    sub[cb].make(start, end, indexrep, &(sitetables[cb]), cb);

#if 0
    fprintf(stderr,"Subset(%d): indexrep=%d start=%d end=%d\n",cb,indexrep,start,end);
#endif
  }
}
	  

//-----------------------------------------------------------------------------
//! Initializer for generic map constructor
void Map::make(const MapFunc& func)
{
  QDP_info("Map::make");

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


//-----------------------------------------------------------------------------


QDP_END_NAMESPACE();
