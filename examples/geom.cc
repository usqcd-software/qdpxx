// $Id: geom.cc,v 1.2 2002-10-28 03:08:44 edwards Exp $
//
// QDP data parallel interface
//
// Geometry

#include "tests.h"

/*! Main geometry object */
Geometry geom;

//! Initializer for geometry
void Geometry::init(const multi1d<int>& nrows)
{
  Layout::initialize(nrows);

  aniso = false;   // No anisotropy by default
  _xi_0 = 1.0;       // Anisotropy factor
  // By default, what is called the time direction is Nd-1
  // NOTE: nothing really depends on this except when aniso is turn on
  // The user can use any nrow direction for time
  t_dir = Nd - 1;
}


//! Initializer for geometry
void Geometry::initAniso(const multi1d<int>& nrows, int aniso_dir, float xx)
{
  Geometry::init(nrows);

  aniso = true;   // No anisotropy by default
  _xi_0 = xx;       // Anisotropy factor
  t_dir = aniso_dir;

  if (t_dir < 0 || _xi_0 <= 0.0)
    SZ_ERROR("anisotropy values not set");
}
