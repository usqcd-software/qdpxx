// $Id: scalar_specific.cc,v 1.4 2002-11-23 02:02:30 edwards Exp $
//
// QDP data parallel interface
//
// Layout
//
// This routine provides various layouts, including
//    lexicographic
//    2-checkerboard  (even/odd-checkerboarding of sites)
//    32-style checkerboard (even/odd-checkerboarding of hypercubes)

#include "qdp.h"
#include "proto.h"

#define  USE_LEXICO_LAYOUT
#undef   USE_CB2_LAYOUT
#undef   USE_CB32_LAYOUT

QDP_BEGIN_NAMESPACE(QDP);

//! Definition of shift function object
NearestNeighborMap  shift;


//-----------------------------------------------------------------------------
// Layout stuff specific to a scalar architecture
namespace Layout
{
  //-----------------------------------------------------
  //! Local data specific to a scalar architecture
  /*! 
   * NOTE: the disadvantage to using a struct to keep things together is
   * that subsequent groupings of namespaces can not just add onto the
   * current namespace. This would be useful if say in a cb=2 implementation
   * the cb_nrow stuff is needed, but does not need to be there for the 
   * lexicographic implementation
   */
  struct
  {
    //! Total lattice volume
    int vol;

    //! Lattice size
    multi1d<int> nrow;

    //! Number of checkboards
    int nsubl;

    //! Total lattice checkerboarded volume
    int vol_cb;

    //! Checkboard lattice size
    multi1d<int> cb_nrow;

    //! Subgrid lattice volume
    int subgrid_vol;
  } _layout;


  //-----------------------------------------------------
  // Functions

  //! Finalizer for a layout
  void finalize() {}

  //! Panic button
  void abort(int status) {exit(status);}

  //! Virtual grid (problem grid) lattice size
  const multi1d<int>& lattSize() {return _layout.nrow;}

  //! Total lattice volume
  int vol() {return _layout.vol;}

  //! Subgrid lattice volume
  int subgridVol() {return _layout.subgrid_vol;}

  //! Returns whether this is the primary node
  /*! Always true on a scalar platform */
  bool primaryNode() {return true;}

  //! Returns the node number of this node
  int nodeNumber() {return 0;}

  //! Returns the number of nodes
  int numNodes() {return 1;}


  //! The linearized site index for the corresponding lexicographic site
  int linearSiteIndex(int site)
  {
    multi1d<int> coord = crtesn(site, lattSize());
    
    return linearSiteIndex(coord);
  }

  //! The lexicographic site index for the corresponding coordinate
  int lexicoSiteIndex(const multi1d<int>& coord)
  {
    return local_site(coord, lattSize());
  }

};


//-----------------------------------------------------------------------------
#if defined(USE_LEXICO_LAYOUT)

#warning "Using a lexicographic layout"

namespace Layout
{
  //! The linearized site index for the corresponding coordinate
  /*! This layout is a simple lexicographic lattice ordering */
  int linearSiteIndex(const multi1d<int>& coord)
  {
    return local_site(coord, lattSize());
  }


  //! The lexicographic site index from the corresponding linearized site
  /*! This layout is a simple lexicographic lattice ordering */
  int lexicoSiteIndex(int linearsite)
  {
    return linearsite;
  }

  //! Initializer for layout
  /*! This layout is a simple lexicographic lattice ordering */
  void initialize(const multi1d<int>& nrows)
  {
    if (nrows.size() != Nd)
      QDP_error_exit("dimension of lattice size not the same as the default");

    _layout.vol=1;
    _layout.nrow = nrows;
    _layout.cb_nrow = nrows;
    for(int i=0; i < Nd; ++i) 
      _layout.vol *= nrows[i];
    _layout.subgrid_vol = _layout.vol;
  
#if defined(NO_MEM)
    if (_layout.vol > VOLUME)
      QDP_error_exit("Allocating a lattice size greater than compile time size: vol=%d",
		     _layout.vol);
#endif

    /* volume of checkerboard. Make sure global variable is set */
    _layout.nsubl = 1;
    _layout.vol_cb = _layout.vol / _layout.nsubl;

#if defined(DEBUG)
    fprintf(stderr,"vol=%d, nsubl=%d\n",_layout.vol,_layout.nsubl);
#endif

    // Default set and subsets
    InitDefaultSets();

    // Make the nearest neighbor shift function available
    shift.make();

    // Initialize RNG
    RNG::InitDefaultRNG();
  }
};


#elif defined(USE_CB2_LAYOUT)

#warning "Using a 2 checkerboard (red/black) layout"

namespace Layout
{
  //! The linearized site index for the corresponding coordinate
  /*! This layout is appropriate for a 2 checkerboard (red/black) lattice */
  int linearSiteIndex(const multi1d<int>& coord)
  {
    multi1d<int> cb_coord = coord;

    cb_coord[0] >>= 1;    // Number of checkerboards
    
    int cb = 0;
    for(int m=0; m<coord.size(); ++m)
      cb += coord[m];
    cb = cb & 1;

    return local_site(cb_coord, _layout.cb_nrow) + cb*_layout.vol_cb;
  }


  //! The lexicographic site index from the corresponding linearized site
  /*! This layout is appropriate for a 2 checkerboard (red/black) lattice */
  int lexicoSiteIndex(int linearsite)
  {
    int cb = linearsite / vol_cb;
    multi1d<int> coord = crtesn(linearsite % _layout.vol_cb, _layout.cb_nrow);

    int cbb = cb;
    for(int m=1; m<coord.size(); ++m)
      cbb += coord[m];
    cbb = cbb & 1;

    coord[0] = 2*coord[0] + cbb;

    return local_site(coord, lattSize());
  }

  //! Initializer for layout
  /*! This layout is appropriate for a 2 checkerboard (red/black) lattice */
  void initialize(const multi1d<int>& nrows)
  {
    if (nrows.size() != Nd)
      QDP_error_exit("dimension of lattice size not the same as the default");

    _layout.vol=1;
    _layout.nrow = nrows;
    _layout.cb_nrow = nrows;
    for(int i=0; i < Nd; ++i) 
      vol *= nrows[i];
    _layout.subgrid_vol = _layout.vol;
  
#if defined(NO_MEM)
    if (_layout.vol > VOLUME)
      QDP_error_exit("Allocating a lattice size greater than compile time size: vol=%d",
		     _layout.vol);
#endif

    /* volume of checkerboard. Make sure global variable is set */
    _layout.nsubl = 2;
    _layout.vol_cb = _layout.vol / _layout.nsubl;

    // Lattice checkerboard size
    _layout.cb_nrow[0] = _layout.nrow[0] / 2;

#if defined(DEBUG)
    fprintf(stderr,"vol=%d, nsubl=%d\n",_layout.vol,_layout.nsubl);
#endif

    InitDefaultSets();

    // Initialize RNG
    RNG::InitDefaultRNG();
  }
};

#elif defined(USE_CB32_LAYOUT)

#warning "Using a 32 checkerboard layout"

namespace Layout
{
  //! The linearized site index for the corresponding coordinate
  /*! This layout is appropriate for a 32-style checkerboard lattice */
  int linearSiteIndex(const multi1d<int>& coord)
  {
    int NN = coord.size();
    int subl = coord[NN-1] & 1;
    for(int m=NN-2; m >= 0; --m)
      subl = (subl << 1) + (coord[m] & 1);

    int cb = 0;
    for(int m=0; m < NN; ++m)
      cb += coord[m] >> 1;

    subl += (cb & 1) << NN;   // Final color or checkerboard

    // Construct the checkerboard lattice coord
    multi1d<int> cb_coord(NN);

    cb_coord[0] = coord[0] >> 2;
    for(int m=1; m < NN; ++m)
      cb_coord[m] = coord[m] >> 1;

    return local_site(cb_coord, _layout.cb_nrow) + subl*_layout.vol_cb;
  }


  //! The lexicographic site index from the corresponding linearized site
  /*! This layout is appropriate for a 32-style checkerboard lattice */
  int lexicoSiteIndex(int linearsite)
  {
    int subl = linearsite / _layout.vol_cb;
    multi1d<int> coord = crtesn(linearsite % vol_cb, cb_nrow);

    int cb = 0;
    for(int m=1; m<coord.size(); ++m)
      cb += coord[m];
    cb &= 1;

    coord[0] <<= 2;
    for(int m=1; m<coord.size(); ++m)
      coord[m] <<= 1;

    coord[0] ^= cb << 1;
    for(int m=0; m<coord.size(); ++m)
      coord[m] ^= (subl & (1 << m)) >> m;
    coord[0] ^= (subl & (1 << Nd)) >> 1;

    return local_site(coord, _layout.nrow);
  }


  //! Initializer for layout
  /*! This layout is appropriate for a 32-style checkerboard lattice */
  void initialize(const multi1d<int>& nrows)
  {
    if (nrows.size() != Nd)
      QDP_error_exit("dimension of lattice size not the same as the default");

    _layout.vol=1;
    _layout.nrow = nrows;
    _layout.cb_nrow = nrows;
    for(int i=0; i < Nd; ++i) 
      vol *= nrows[i];
    _layout.subgrid_vol = _layout.vol;
  
#if defined(NO_MEM)
    if (_layout.vol > VOLUME)
      QDP_error_exit("Allocating a lattice size greater than compile time size: vol=%d",
		     _layout.vol);
#endif

    /* volume of checkerboard. Make sure global variable is set */
    _layout.nsubl = 1 << (Nd+1);
    _layout.vol_cb = _layout.vol / _layout.nsubl;

    // Lattice checkerboard size
    _layout.cb_nrow[0] = _layout.nrow[0] >> 2;
    for(int i=1; i < Nd; ++i) 
      _layout.cb_nrow[i] = _layout.nrow[i] >> 1;
  
#if defined(DEBUG)
    fprintf(stderr,"vol=%d, nsubl=%d\n",vol,nsubl);
#endif

    InitDefaultSets();

    // Initialize RNG
    RNG::InitDefaultRNG();
  }
};

#else

#error "no appropriate layout defined"

#endif



//-----------------------------------------------------------------------------
// Auxilliary operations
//! coord[mu]  <- mu  : fill with lattice coord in mu direction
LatticeInteger latticeCoordinate(int mu)
{
  LatticeInteger d;

  if (mu < 0 || mu >= Nd)
    QDP_error_exit("dimension out of bounds");

  const multi1d<int> &nrow = Layout::lattSize();

  for(int i=0; i < Layout::vol(); ++i) 
  {
    int site = Layout::lexicoSiteIndex(i);
    for(int k=0; k <= mu; ++k)
    {
      d.elem(i) = Integer(site % nrow[k]).elem();
      site /= nrow[k];
    } 
  }

  return d;
}



//-----------------------------------------------------------------------------
//! Constructor from an int function
void Set::make(int (&func)(const multi1d<int>& coordinate), int nsubset_indices)
{
#if 1
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

#if 1
    fprintf(stderr,"Subset(%d): indexrep=%d start=%d end=%d\n",cb,indexrep,start,end);
#endif
  }
}
	  

//-----------------------------------------------------------------------------
//! Initializer for nearest neighbor shift
void NearestNeighborMap::make()
{
  //--------------------------------------
  // Setup the communication index arrays
  soffsets.resize(Nd, 2, Layout::vol());

  /* Get the offsets needed for neighbour comm.
     * soffsets(direction,isign,position)
     *  where  isign    = +1 : plus direction
     *                  =  0 : negative direction
     *         dir      =  0, ..., Nd-1
     * the offsets contain the current site, i.e the neighbour for site i
     * is  soffsets(i,dir,mu) and NOT  i + soffset(..) 
     */
  const multi1d<int>& nrow = Layout::lattSize();

  for(int site=0; site < Layout::vol(); ++site)
  {
    multi1d<int> coord = crtesn(site, nrow);
    int ipos = Layout::linearSiteIndex(coord);

    for(int m=0; m<Nd; ++m)
    {
      multi1d<int> tmpcoord = coord;

      /* Neighbor in backward direction */
      tmpcoord[m] = (coord[m] - 1 + nrow[m]) % nrow[m];
      soffsets(m,0,ipos) = Layout::linearSiteIndex(tmpcoord);

      /* Neighbor in forward direction */
      tmpcoord[m] = (coord[m] + 1) % nrow[m];
      soffsets(m,1,ipos) = Layout::linearSiteIndex(tmpcoord);
    }
  }

#if 0
  for(int m=0; m < Nd; ++m)
    for(int s=0; s < 2; ++s)
      for(int ipos=0; ipos < Layout::vol(); ++ipos)
	fprintf(stderr,"soffsets(%d,%d,%d) = %d\n",ipos,s,m,soffsets(m,s,ipos));
#endif
}


//-----------------------------------------------------------------------------


QDP_END_NAMESPACE();
