// -*- C++ -*-
// $Id: geom.h,v 1.2 2002-10-28 03:08:44 edwards Exp $
//
// QDP data parallel interface
//
// Geometry

//! Geometry class holding info on problem size and machine info
class Geometry
{
public:
  //! An empty geometry constructor is allowed
  Geometry() {}

  //! Initializer for a geometry
  void init(const multi1d<int>& nrows);

  //! Initializer for a anisotropic geometry
  void initAniso(const multi1d<int>& nrows, int aniso_dir, float xi_0);

  //! Virtual grid (problem grid) lattice size
  const multi1d<int>& LattSize() const {return Layout::lattSize();}

  //! Total lattice volume
  int vol() {return Layout::vol();}

  //! Is anisotropy enabled?
  bool anisoP() {return aniso;}

  //! Time direction
  int tDir() {return t_dir;}

  //! Anisotropy factor
  float xi_0() {return _xi_0;}

protected:
  //! No public copy constructor
  Geometry(const Geometry&) {}

private:
  //! Anisotropy flag
  bool aniso;

  //! Anisotropy factor (in time)
  float _xi_0;

  //! The time direction (used only in conjunction with aniso)
  int t_dir;
};

/*! Main geometry object */
extern Geometry geom;

