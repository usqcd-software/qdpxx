// -*- C++ -*-
// $Id: layout.h,v 1.2 2002-10-12 00:58:32 edwards Exp $
//
// QDP data parallel interface
//
// Layout

QDP_BEGIN_NAMESPACE(QDP);

/*! @defgroup layout  Layout 
 *
 * Class holding info on problem size and machine info
 *
 * @{
 */

//! Layout class holding info on problem size and machine info
class Layout
{
public:
  //! Initializer for a geometry
  void Init(const multi1d<int>& nrows);

  //! Virtual grid (problem grid) lattice size
  const multi1d<int>& LattSize() const {return nrow;}

  //! Total lattice volume
  int Vol() {return vol;}

  //! The linearized site index for the corresponding coordinate
  int LinearSiteIndex(const multi1d<int>& coord);

  //! The linearized site index for the corresponding lexicographic site
  int LinearSiteIndex(int lexicosite);

  //! The lexicographic site index from the corresponding linearized site
  int LexicoSiteIndex(int linearsite);

  //! The lexicographic site index for the corresponding coordinate
  int LexicoSiteIndex(const multi1d<int>& coord);

protected:

private:
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
};

/*! Main layout object */
extern Layout layout;

/*! @} */   // end of group layout

QDP_END_NAMESPACE();
