// -*- C++ -*-
// $Id: geom.h,v 1.1 2002-09-12 18:22:17 edwards Exp $
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
  void Init(const multi1d<int>& nrows);

  //! Initializer for a anisotropic geometry
  void InitAniso(const multi1d<int>& nrows, int aniso_dir, float xi_0);

  //! Virtual grid (problem grid) lattice size
  const multi1d<int>& LattSize() const {return layout.LattSize();}

  //! Total lattice volume
  int Vol() {return layout.Vol();}

  //! Is anisotropy enabled?
  bool AnisoP() {return aniso;}

  //! Time direction
  int Tdir() {return t_dir;}

  //! Anisotropy factor
  float Xi_0() {return xi_0;}

protected:
  //! No public copy constructor
  Geometry(const Geometry&) {}

private:
  //! Anisotropy flag
  bool aniso;

  //! Anisotropy factor (in time)
  float xi_0;

  //! The time direction (used only in conjunction with aniso)
  int t_dir;
};

/*! Main geometry object */
extern Geometry geom;

