// -*- C++ -*-
// $Id: layout.h,v 1.12 2003-04-09 19:32:27 edwards Exp $

/*! @file
 * @brief Lattice layout
 *
 * Lattice layout namespace and operations
 */

QDP_BEGIN_NAMESPACE(QDP);

/*! @defgroup layout  Layout 
 *
 * Namespace holding info on problem size and machine info
 *
 * @{
 */

//! Layout namespace holding info on problem size and machine info
/*! 
 * This is a namespace instead of a class since it is completely static -
 * no such object should be created 
 *
 * The functions here should be common to all architectures
 */
namespace Layout
{
  //! Main creation routine
  void create();

  //! Main destruction routine
  void destroy();

  //! Set lattice size -- problem size
  void setLattSize(const multi1d<int>& nrows);

  //! Set SMP flag -- true if using smp/multiprocessor mode on a node
  void setSMPFlag(bool);

  //! Set number of processors in a multi-threaded implementation
  void setNumProc(int N);

  //! Returns the logical node number for the corresponding lattice coordinate
  /*! The API requires this function to be here */
  int nodeNumber(const multi1d<int>& coord);

  //! The linearized site index within a node for the corresponding lattice coordinate
  /*! The API requires this function to be here */
  int linearSiteIndex(const multi1d<int>& coord);

  //! Reconstruct the lattice coordinate from the node and site number
  /*! 
   * This is the inverse of the nodeNumber and linearSiteIndex functions.
   * The API requires this function to be here.
   */
  multi1d<int> siteCoords(int node, int index);

  //! Returns the node number of this node
  int nodeNumber();

  //! Returns the number of nodes
  int numNodes();

  //! Virtual grid (problem grid) lattice size
  const multi1d<int>& lattSize();

  //! Total lattice volume
  int vol();

  //! Number of sites on node
  int sitesOnNode();

  //! Returns whether this is the primary node
  bool primaryNode();

  //! The lexicographic site index within a node for the corresponding lattice coordinate
  int lexicoSiteIndex(const multi1d<int>& coord);

  //! The linearized site index for the corresponding lexicographic site
  int linearSiteIndex(int lexicosite);

  //! The lexicographic site index from the corresponding linearized site
  int lexicoSiteIndex(int linearsite);
}

//! Declaration of shift function object
extern ArrayBiDirectionalMap  shift;


/*! @} */   // end of group layout

QDP_END_NAMESPACE();
