// $Id: qdp_parscalar_layout.cc,v 1.3 2003-07-18 20:03:44 edwards Exp $

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
#include "qdp_util.h"

#include "qmp.h"

#define  USE_LEXICO_LAYOUT
#undef   USE_CB2_LAYOUT
#undef   USE_CB32_LAYOUT

QDP_BEGIN_NAMESPACE(QDP);

namespace Layout
{
  //-----------------------------------------------------
  //! Local data specific to all architectures
  /*! 
   * NOTE: the disadvantage to using a struct to keep things together is
   * that subsequent groupings of namespaces can not just add onto the
   * current namespace.
   */
  struct
  {
    //! Total lattice volume
    int vol;

    //! Lattice size
    multi1d<int> nrow;

    //! Subgrid lattice volume
    int subgrid_vol;

    //! Subgrid lattice size
    multi1d<int> subgrid_nrow;

    //! Logical node coordinates
    multi1d<int> logical_coord;

    //! Logical system size
    multi1d<int> logical_size;

    //! Node rank
    int node_rank;

    //! Total number of nodes
    int num_nodes;
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
  int sitesOnNode() {return _layout.subgrid_vol;}

  //! Returns whether this is the primary node
  bool primaryNode() {return (_layout.node_rank == 0) ? true : false;}

  //! Subgrid (grid on each node) lattice size
  const multi1d<int>& subgridLattSize() {return _layout.subgrid_nrow;}

  //! Returns the node number of this node
  int nodeNumber() {return _layout.node_rank;}

  //! Returns the number of nodes
  int numNodes() {return _layout.num_nodes;}

  //! Returns the logical node coordinates for this node
  const multi1d<int>& nodeCoord() {return _layout.logical_coord;}

  //! Returns the logical size of this machine
  const multi1d<int>& logicalSize() {return _layout.logical_size;}


  //! Initializer for layout
  void init()
  {
    _layout.num_nodes = QMP_get_number_of_nodes();
    _layout.node_rank = QMP_get_node_number();
  }



  //! The linearized site index for the corresponding lexicographic site
  int linearSiteIndex(int site)
  { 
    multi1d<int> coord = crtesn(site, lattSize());
    
    return linearSiteIndex(coord);
  }


  //! Initializer for all the layout defaults
  void InitDefaults()
  {
#if defined(DEBUG)
    QDP_info("Create default subsets");
#endif
    // Default set and subsets
    InitDefaultSets();

    // Default maps
    InitDefaultMaps();

    // Initialize RNG
    RNG::InitDefaultRNG();
  }


  //! Main lattice creation routine
  void create()
  {
    if ( ! QDP_isInitialized() )
      QDP_error_exit("QDP is not initialized");

    if (_layout.nrow.size() != Nd)
      QDP_error_exit("dimension of lattice size not the same as the default");

    _layout.vol=1;
    for(int i=0; i < Nd; ++i) 
      _layout.vol *= _layout.nrow[i];
  
#if defined(DEBUG)
    QDP_info("vol=%d",_layout.vol);
#endif

#if defined(DEBUG)
    QDP_info("Initialize layout");
#endif

    // Simple check - we insist here that the total volume is divisible
    // by the number of processors. Will also insist that the problem
    // size is regular on each node
    if (_layout.vol % numNodes() != 0)
    {
      if (Layout::primaryNode())
	QDP_error_exit("Layout::create - problem size not divisible by number of processors");
    }
    
    // Crap to make the compiler happy with the C prototype
    unsigned int unsigned_nrow[Nd];
    for(int i=0; i < Nd; ++i)
      unsigned_nrow[i] = _layout.nrow[i];

    QMP_layout_grid(unsigned_nrow, Nd);


    // Pull out useful stuff
    const unsigned int* phys_size = QMP_get_logical_dimensions();
    const unsigned int* phys_coord = QMP_get_logical_coordinates();
    const unsigned int* subgrid_size = QMP_get_subgrid_dimensions();

    _layout.subgrid_vol = QMP_get_number_of_subgrid_sites();

    _layout.subgrid_nrow.resize(Nd);
    _layout.logical_coord.resize(Nd);
    _layout.logical_size.resize(Nd);

    for(int i=0; i < Nd; ++i)
    {
      _layout.subgrid_nrow[i] = subgrid_size[i];
      _layout.logical_coord[i] = phys_coord[i];
      _layout.logical_size[i] = phys_size[i];
    }

    // Diagnostics
    if (Layout::primaryNode())
    {
      cerr << "Lattice initialized:\n";
      cerr << "  problem size =";
      for(int i=0; i < Nd; ++i)
	cerr << " " << _layout.nrow[i];
      cerr << endl;

      cerr << "  logical machine size =";
      for(int i=0; i < Nd; ++i)
	cerr << " " << _layout.logical_size[i];
      cerr << endl;

      cerr << "  logical node coord =";
      for(int i=0; i < Nd; ++i)
	cerr << " " << _layout.logical_coord[i];
      cerr << endl;

      cerr << "  subgrid size =";
      for(int i=0; i < Nd; ++i)
	cerr << " " << _layout.subgrid_nrow[i];
      cerr << endl;

      cerr << "  total volume = " << _layout.vol << endl;
      cerr << "  subgrid volume = " << _layout.subgrid_vol << endl;
    }

    // Sanity check - check the layout functions make sense
    for(int i=0; i < _layout.vol; ++i) 
    {
      int j = Layout::linearSiteIndex(Layout::siteCoords(Layout::nodeNumber(),i));
      if (i != j)
	QDP_error_exit("Layout::create - Layout problems, the layout functions do not work correctly with this lattice size");
    }

    // Initialize various defaults
    InitDefaults();

    if (Layout::primaryNode())
      cerr << "Finished lattice layout\n";
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
    multi1d<int> tmp_coord(Nd);

    for(int i=0; i < coord.size(); ++i)
      tmp_coord[i] = coord[i] % Layout::subgridLattSize()[i];
    
    return local_site(tmp_coord, Layout::subgridLattSize());
  }


  //! The node number for the corresponding lattice coordinate
  /*! This layout is a simple lexicographic lattice ordering */
  int nodeNumber(const multi1d<int>& coord)
  {
    multi1d<int> tmp_coord(Nd);

    for(int i=0; i < coord.size(); ++i)
      tmp_coord[i] = coord[i] / Layout::subgridLattSize()[i];
    
    return local_site(tmp_coord, Layout::logicalSize());
  }


  //! Returns the lattice site for some input node and linear index
  /*! This layout is a simple lexicographic lattice ordering */
  multi1d<int> siteCoords(int node, int linear)
  {
    multi1d<int> coord = crtesn(node, Layout::logicalSize());

    // Get the base (origins) of the absolute lattice coord
    coord *= Layout::subgridLattSize();
    
    // Find the coordinate within a node and accumulate
    // This is a lexicographic ordering
    coord += crtesn(linear, Layout::subgridLattSize());

    return coord;
  }
};

//-----------------------------------------------------------------------------

#elif defined(USE_CB2_LAYOUT)

#warning "Using a 2 checkerboard (red/black) layout"

namespace Layout
{
  //! The linearized site index for the corresponding coordinate
  /*! This layout is appropriate for a 2 checkerboard (red/black) lattice */
  int linearSiteIndex(const multi1d<int>& coord)
  {
    int subgrid_vol_cb = Layout::sitesOnNode() >> 1;
    multi1d<int> subgrid_cb_nrow = Layout::subgridLattSize();
    subgrid_cb_nrow[0] >>= 1;

    int cb = 0;
    for(int m=0; m < Nd; ++m)
      cb += coord[m];
    cb &= 1;

    multi1d<int> subgrid_cb_coord(Nd);
    for(int i=0; i < Nd; ++i)
      subgrid_cb_coord[i] = coord[i] % subgrid_cb_nrow[i];
    
    return local_site(subgrid_cb_coord, subgrid_cb_nrow) + cb*subgrid_vol_cb;
  }


  //! The node number for the corresponding lattice coordinate
  /*! This layout is a simple lexicographic lattice ordering */
  int nodeNumber(const multi1d<int>& coord)
  {
    int subgrid_vol_cb = Layout::sitesOnNode() >> 1;
    multi1d<int> subgrid_cb_nrow = Layout::subgridLattSize();
    subgrid_cb_nrow[0] >>= 1;

    int cb = 0;
    for(int m=0; m < Nd; ++m)
      cb += coord[m];
    cb &= 1;

    multi1d<int> logical_cb_coord(Nd);
    for(int i=0; i < Nd; ++i)
      logical_cb_coord[i] = coord[i] / subgrid_cb_nrow[i];
    
    return local_site(logical_cb_coord, Layout::logicalSize());
  }


  //! Reconstruct the lattice coordinate from the node and site number
  /*! 
   * This is the inverse of the nodeNumber and linearSiteIndex functions.
   * The API requires this function to be here.
   */
  multi1d<int> siteCoords(int node, int linearsite) // ignore node
  {
    int subgrid_vol_cb = Layout::sitesOnNode() >> 1;
    multi1d<int> subgrid_cb_nrow = Layout::subgridLattSize();
    subgrid_cb_nrow[0] >>= 1;

    multi1d<int> coord = crtesn(node, Layout::logicalSize());

    // Get the base (origins) of the absolute lattice coord
    coord *= Layout::subgridLattSize();
    
    int cb = linearsite / subgrid_vol_cb;
    coord += crtesn(linearsite % subgrid_vol_cb, subgrid_cb_nrow);

    int cbb = cb;
    for(int m=1; m < Nd; ++m)
      cbb += coord[m];
    cbb &= 1;

    coord[0] = 2*coord[0] + cbb;

    return coord;
  }
};

//-----------------------------------------------------------------------------

#elif defined(USE_CB32_LAYOUT)

#error "Using a 32 checkerboard layout"


#else

#error "no appropriate layout defined"

#endif

//-----------------------------------------------------------------------------


QDP_END_NAMESPACE();
