// -*- C++ -*-
// $Id: layout.h,v 1.4 2002-10-28 03:08:44 edwards Exp $

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
  //! Probably standard initializer for a layout
  /*! This should read from some internal state and start initializing */
  inline void initialize() {SZ_ERROR("generic initialize not implemented");}

  //! Rudimentary initializer for a layout
  void initialize(const multi1d<int>& nrows);

  //! Finalizer for a layout
  void finalize();

  //! Panic button
  void abort(int status);

  //! Virtual grid (problem grid) lattice size
  const multi1d<int>& lattSize();

  //! Total lattice volume
  int vol();

  //! Subgrid lattice volume
  int subgridVol();

  //! Returns whether this is the primary node
  bool primaryNode();


  //! The linearized site index within a node for the corresponding lattice coordinate
  int linearSiteIndex(const multi1d<int>& coord);

  //! The lexicographic site index within a node for the corresponding lattice coordinate
  int lexicoSiteIndex(const multi1d<int>& coord);



  //! The linearized site index for the corresponding lexicographic site
  int linearSiteIndex(int lexicosite);

  //! The lexicographic site index from the corresponding linearized site
  int lexicoSiteIndex(int linearsite);

};

/*! @} */   // end of group layout

QDP_END_NAMESPACE();
