// $Id: scalar_layout.cc,v 1.4 2003-01-20 16:19:42 edwards Exp $

/*! @file
 * @brief Parscalar layout routines
 * 
 * Layout routines for parscalar implementation
 * QDP data parallel interface
 *
 * Layout
 *
 * This routine provides various layouts, including
 *    lexicographic
 *    2-checkerboard  (even/odd-checkerboarding of sites)
 *    32-style checkerboard (even/odd-checkerboarding of hypercubes)
 */

#include "qdp.h"
#include "proto.h"

#define  USE_LEXICO_LAYOUT
#undef   USE_CB2_LAYOUT
#undef   USE_CB32_LAYOUT

QDP_BEGIN_NAMESPACE(QDP);

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

  //! Main destruction routine
  void destroy() {}

  //! Set virtual grid (problem grid) lattice size
  void setLattSize(const multi1d<int>& nrows) {_layout.nrow = nrows;}

  //! Set SMP flag -- true if using smp/multiprocessor mode on a node
  /*! For now, this is ignored */
  void setSMPFlag(bool flag) {}

  //! Set number of processors in a multi-threaded implementation
  /*! For now, this is ignored */
  void setNumProc(int N) {}

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

  //! Initializer for all the layout defaults
  void InitDefaults()
  {
    // Default set and subsets
    InitDefaultSets();

    // Default maps
    InitDefaultMaps();

    // Initialize RNG
    RNG::InitDefaultRNG();
  }

};


//-----------------------------------------------------------------------------
#if defined(USE_LEXICO_LAYOUT)

#warning "Using a lexicographic layout"

namespace Layout
{
  //! Reconstruct the lattice coordinate from the node and site number
  /*! 
   * This is the inverse of the nodeNumber and linearSiteIndex functions.
   * The API requires this function to be here.
   */
  multi1d<int> siteCoords(int node, int linearsite)
  {
    return crtesn(linearsite, lattSize());
  }

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
  void create()
  {
    if ( ! QDP_isInitialized() )
      QDP_error_exit("QDP is not initialized");

    if (_layout.nrow.size() != Nd)
      QDP_error_exit("dimension of lattice size not the same as the default");

    _layout.vol=1;
    _layout.cb_nrow = _layout.nrow;
    for(int i=0; i < Nd; ++i) 
      _layout.vol *= _layout.nrow[i];
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

    // Initialize various defaults
    InitDefaults();
  }
};


#elif defined(USE_CB2_LAYOUT)

#warning "Using a 2 checkerboard (red/black) layout"

namespace Layout
{
  //! Reconstruct the lattice coordinate from the node and site number
  /*! 
   * This is the inverse of the nodeNumber and linearSiteIndex functions.
   * The API requires this function to be here.
   */
  multi1d<int> siteCoords(int node, int linearsite)
  {
    int cb = linearsite / vol_cb;
    multi1d<int> coord = crtesn(linearsite % _layout.vol_cb, _layout.cb_nrow);

    int cbb = cb;
    for(int m=1; m<coord.size(); ++m)
      cbb += coord[m];
    cbb = cbb & 1;

    coord[0] = 2*coord[0] + cbb;

    return coord;
  }

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
    return local_site(siteCoords(0,linearsite), lattSize());
  }

  //! Initializer for layout
  /*! This layout is appropriate for a 2 checkerboard (red/black) lattice */
  void create()
  {
    if ( ! QDP_isInitialized() )
      QDP_error_exit("QDP is not initialized");

    if (_layout::nrow.size() != Nd)
      QDP_error_exit("dimension of lattice size not the same as the default");

    _layout.vol=1;
    _layout.cb_nrow = _layout::nrow;
    for(int i=0; i < Nd; ++i) 
      vol *= _layout::nrow[i];
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

    // Initialize various defaults
    InitDefaults();
  }
};

#elif defined(USE_CB32_LAYOUT)

#warning "Using a 32 checkerboard layout"

namespace Layout
{
  //! Reconstruct the lattice coordinate from the node and site number
  /*! 
   * This is the inverse of the nodeNumber and linearSiteIndex functions.
   * The API requires this function to be here.
   */
  multi1d<int> siteCoords(int node, int linearsite) // ignore node
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

    return coord;
  }

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
    return local_site(siteCoords(0,linearsite), lattSize());
  }


  //! Initializer for layout
  /*! This layout is appropriate for a 32-style checkerboard lattice */
  void create()
  {
    if ( ! QDP_isInitialized() )
      QDP_error_exit("QDP is not initialized");

    if (_layout::nrow.size() != Nd)
      QDP_error_exit("dimension of lattice size not the same as the default");

    _layout.vol=1;
    _layout.cb_nrow = _layout::nrow;
    for(int i=0; i < Nd; ++i) 
      vol *= _layout::nrow[i];
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

    // Initialize various defaults
    InitDefaults();
  }
};

#else

#error "no appropriate layout defined"

#endif

//-----------------------------------------------------------------------------


QDP_END_NAMESPACE();
