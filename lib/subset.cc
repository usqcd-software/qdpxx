// $Id: subset.cc,v 1.2 2002-09-26 20:04:25 edwards Exp $
//
// QDP data parallel interface
//
// Sets and subsets

#include "qdp.h"
#include "proto.h"

QDP_BEGIN_NAMESPACE(QDP);

//! Default all set
Set set_all;

//! Default all subset
Subset all;

//! Default 2-checkerboard (red/black) set
Set rb;

//! Default 2^{Nd+1}-checkerboard set. Useful for pure gauge updating.
Set mcb;

//! Global context
Context *global_context;

//! Function used for constructing the all subset
int subset_all_func(const multi1d<int>& coordinate) {return 0;}
  
//! Function used for constructing red-black (2) checkerboard */
int subset_rb_func(const multi1d<int>& coordinate)
{
  int sum = 0;
  for(int m=0; m < coordinate.size(); ++m)
    sum += coordinate[m];

  return sum & 1;
}
  
//! Function used for constructing 32 checkerboard. */
int subset_32cb_func(const multi1d<int>& coordinate)
{
  int initial_color = 0;
  for(int m=Nd-1; m >= 0; --m)
    initial_color = (initial_color << 1) + (coordinate[m] & 1);

  int cb = 0;
  for(int m=0; m < Nd; ++m)
    cb += coordinate[m] >> 1;

  cb &= 1;
  return initial_color + (cb << Nd);
}
  

//! Initializer for sets
void InitDefaultSets()
{
  // Initialize the red/black checkerboard
  rb.Make(subset_rb_func, 2);

    // Initialize the 32-style checkerboard
  mcb.Make(subset_32cb_func, 1 << (Nd+1));

  // The all set
  set_all.Make(subset_all_func, 1);

  // The all subset
  all.Make(set_all[0]);

  // Set the global context
  global_context = new Context(all);
}


//! Simple constructor called to produce a Subset from inside a Set
void Subset::Make(int start, int end, bool rep, multi3d<int>* soff, 
		  multi1d<int>* ind, int cb)
{
  startSite = start;
  endSite = end;
  indexrep = rep;
  soffsets = soff;
  sitetable = ind;
  sub_index = cb;
}


//! Constructor from an int function
void Set::Make(int (&func)(const multi1d<int>& coordinate), int nsubset_indices)
{
#if 1
  fprintf(stderr,"Set a subset: nsubset = %d\n",nsubset_indices);
#endif

  // First initialize the offsets
  InitOffsets();

  // This actually allocates the subsets
  sub.resize(nsubset_indices);

  // Create the space of the colorings of the lattice
  /*! Loop over all sites determining their color */
  lat_color.resize(layout.Vol());

  // Create the array holding the array of sitetable info
  // This may actually hold anything 
  sitetables.resize(nsubset_indices);

  for(int site=0; site < layout.Vol(); ++site)
  {
    const multi1d<int> coord = crtesn(site, layout.LattSize());
    int linear = layout.LinearSiteIndex(coord);
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
    int prev;
    bool found_gap = false;

    for(int linear=0; linear < layout.Vol(); ++linear)
    {
      int lexico = layout.LexicoSiteIndex(linear);
      multi1d<int> coord = crtesn(lexico, layout.LattSize());

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

    // If a gap is found, then resort to a site table lookup
    if (found_gap)
    {
      start = 0;
      end = -1;
      indexrep = true;

      // First loop and see how many sites are needed
      int num_sitetable = 0;
      for(int linear=0; linear < layout.Vol(); ++linear)
	if (lat_color[linear] == cb)
	  ++num_sitetable;

      // Now take the inverse of the lattice coloring to produce
      // the site list
      multi1d<int>& sitetable = sitetables[cb];
      sitetable.resize(num_sitetable);

      for(int linear=0, j=0; linear < layout.Vol(); ++linear)
	if (lat_color[linear] == cb)
	  sitetable[j++] = linear;
    }

    sub[cb].Make(start, end, indexrep, &soffsets, &(sitetables[cb]), cb);

#if 1
    fprintf(stderr,"Subset(%d): indexrep=%d start=%d end=%d\n",cb,indexrep,start,end);
#endif
  }
}
	  

//! Initializer for sets
void Set::InitOffsets()
{
  //--------------------------------------
  // Setup the communication index arrays
  soffsets.resize(Nd, 2, layout.Vol());

  /* Get the offsets needed for neighbour comm.
     * soffsets(direction,isign,position)
     *  where  isign    = +1 : plus direction
     *                  =  0 : negative direction
     *         cb       =  0 : even lattice (includes origin)
     *                  = +1 : odd lattice (does not include origin)
     * the offsets cotain the current site, i.e the neighbour for site i
     * is  soffsets(i,dir,mu) and NOT  i + soffset(..) 
     * NOTE: the sites are order the cb=0 (even lattice - includes origin)
     * are the first vol_cb chunk and the cb=1 (odd lattice) are the second
     * chunk of position running from 0 to vol-1
     */
  const multi1d<int>& nrow = layout.LattSize();

  for(int site=0; site < layout.Vol(); ++site)
  {
    multi1d<int> coord = crtesn(site, nrow);
    int ipos = layout.LinearSiteIndex(coord);

    for(int m=0; m<Nd; ++m)
    {
      multi1d<int> tmpcoord = coord;

      /* Neighbor in backward direction */
      tmpcoord[m] = (coord[m] - 1 + nrow[m]) % nrow[m];
      soffsets(m,0,ipos) = layout.LinearSiteIndex(tmpcoord);

      /* Neighbor in forward direction */
      tmpcoord[m] = (coord[m] + 1) % nrow[m];
      soffsets(m,1,ipos) = layout.LinearSiteIndex(tmpcoord);
    }
  }

#if 0
  for(int m=0; m < Nd; ++m)
    for(int s=0; s < 2; ++s)
      for(int ipos=0; ipos < layout.Vol(); ++ipos)
	fprintf(stderr,"soffsets(%d,%d,%d) = %d\n",ipos,s,m,soffsets(m,s,ipos));
#endif
}

QDP_END_NAMESPACE();
