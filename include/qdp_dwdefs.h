// -*- C++ -*-
// $Id: qdp_dwdefs.h,v 1.12 2004-05-23 20:43:05 edwards Exp $

/*! \file
 * \brief Type definitions
 */

QDP_BEGIN_NAMESPACE(QDP);

/*! \addtogroup defs Type definitions
 *
 * User constructed types made from QDP type compositional nesting.
 * The layout is suitable for a scalar-like implementation. Namely,
 * sites are the slowest varying index.
 *
 * @{
 */

#include <qdp_config.h>
#include "qdp_precision.h"

//----------------------------------------------------------------------
//! Gamma matrices are conveniently defined for this Ns
typedef GammaType<Ns> Gamma;


// Aliases for a scalar architecture

// Fixed fermion type
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL> >, Nc>, 4> > LatticeDiracFermion;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL> >, Nc>, 1> > LatticeStaggeredFermion;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL> >, Nc>, 4> > LatticeDiracPropagator;

// Floating aliases
typedef OLattice< PScalar< PColorVector< RComplex< PScalar<REAL> >, Nc> > > LatticeColorVector;
typedef OLattice< PSpinVector< PScalar< RComplex< PScalar<REAL> > >, Ns> > LatticeSpinVector;
typedef OLattice< PScalar< PColorMatrix< RComplex< PScalar<REAL> >, Nc> > > LatticeColorMatrix;
typedef OLattice< PSpinMatrix< PScalar< RComplex< PScalar<REAL> > >, Ns> > LatticeSpinMatrix;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL> >, Nc>, Ns> > LatticeFermion;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL> >, Nc>, Ns>>1 > > LatticeHalfFermion;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL> >, Nc>, Ns> > LatticePropagator;
typedef OLattice< PScalar< PScalar< RComplex< PScalar<REAL> > > > > LatticeComplex;

typedef OLattice< PScalar< PSeed < RScalar< PScalar<INTEGER32> > > > > LatticeSeed;
typedef OLattice< PScalar< PScalar< RScalar< PScalar<INTEGER32> > > > > LatticeInteger;
typedef OLattice< PScalar< PScalar< RScalar< PScalar<REAL> > > > > LatticeReal;
typedef OLattice< PScalar< PScalar< RScalar< PScalar<DOUBLE> > > > > LatticeDouble;
typedef OLattice< PScalar< PScalar< RScalar< PScalar<LOGICAL> > > > > LatticeBoolean;

typedef OScalar< PScalar< PColorVector< RComplex< PScalar<REAL> >, Nc> > > ColorVector;
typedef OScalar< PScalar< PColorMatrix< RComplex< PScalar<REAL> >, Nc> > > ColorMatrix;
typedef OScalar< PSpinVector< PScalar< RComplex< PScalar<REAL> > >, Ns> > SpinVector;
typedef OScalar< PSpinMatrix< PScalar< RComplex< PScalar<REAL> > >, Ns> > SpinMatrix;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL> >, Nc>, 1> > StaggeredFermion;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL> >, Nc>, Ns> > Fermion;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL> >, Nc>, Ns>>1 > > HalfFermion;
typedef OScalar< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL> >, Nc>, Ns> > Propagator;
typedef OScalar< PScalar< PScalar< RComplex< PScalar<REAL> > > > > Complex;

typedef OScalar< PScalar< PSeed< RScalar< PScalar<INTEGER32> > > > > Seed;
typedef OScalar< PScalar< PScalar< RScalar< PScalar<INTEGER32> > > > > Integer;
typedef OScalar< PScalar< PScalar< RScalar< PScalar<REAL> > > > > Real;
typedef OScalar< PScalar< PScalar< RScalar< PScalar<DOUBLE> > > > > Double;
typedef OScalar< PScalar< PScalar< RScalar< PScalar<LOGICAL> > > > > Boolean;

typedef OScalar< PScalar< PColorVector< RComplex< PScalar<DOUBLE> >, Nc> > > DColorVector;
typedef OScalar< PScalar< PColorMatrix< RComplex< PScalar<DOUBLE> >, Nc> > > DColorMatrix;
typedef OScalar< PSpinVector< PScalar< RComplex< PScalar<DOUBLE> > >, Ns> > DSpinVector;
typedef OScalar< PSpinMatrix< PScalar< RComplex< PScalar<DOUBLE> > >, Ns> > DSpinMatrix;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<DOUBLE> >, Nc>, 1> > DStaggeredFermion;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<DOUBLE> >, Nc>, Ns> > DFermion;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<DOUBLE> >, Nc>, Ns>>1 > > DHalfFermion;
typedef OScalar< PSpinMatrix< PColorMatrix< RComplex< PScalar<DOUBLE> >, Nc>, Ns> > DPropagator;
typedef OScalar< PScalar< PScalar< RComplex< PScalar<DOUBLE> > > > > DComplex;

// Other useful names
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL> >, Nc>, Ns> > ColorVectorSpinVector;
typedef OScalar< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL> >, Nc>, Ns> > ColorMatrixSpinMatrix;

// Level below outer for internal convenience
typedef PScalar< PScalar< RScalar< PScalar<REAL> > > > IntReal;
typedef PScalar< PScalar< RScalar< PScalar<INTEGER32> > > > IntInteger;
typedef PScalar< PScalar< RScalar< PScalar<DOUBLE> > > > IntDouble;
typedef PScalar< PScalar< RScalar< PScalar<LOGICAL> > > > IntBoolean;

// Odd-ball to support random numbers
typedef Real ILatticeReal;
typedef Seed ILatticeSeed;

// Floating precision, but specific to a fixed color or spin
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL> >, 3>, 4> > LatticeDiracFermion3;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL> >, 2>, 4> > LatticeDiracFermion2;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL> >, 1>, 4> > LatticeDiracFermion1;

typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL> >, 3>, 1> > LatticeStaggeredFermion3;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL> >, 2>, 1> > LatticeStaggeredFermion2;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL> >, 1>, 1> > LatticeStaggeredFermion1;

typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL> >, 3>, 4> > LatticeDiracPropagator3;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL> >, 2>, 4> > LatticeDiracPropagator2;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL> >, 1>, 4> > LatticeDiracPropagator1;

typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL> >, 3>, Ns> > LatticeFermion3;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL> >, 2>, Ns> > LatticeFermion2;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL> >, 1>, Ns> > LatticeFermion1;

typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL> >, 3>, Ns>>1 > > LatticeHalfFermion3;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL> >, 2>, Ns>>1 > > LatticeHalfFermion2;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL> >, 1>, Ns>>1 > > LatticeHalfFermion1;

typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL> >, 3>, Ns> > LatticePropagator3;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL> >, 2>, Ns> > LatticePropagator2;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL> >, 1>, Ns> > LatticePropagator1;

typedef OLattice< PScalar< PColorMatrix< RComplex< PScalar<REAL> >, 3> > > LatticeColorMatrix3;
typedef OLattice< PScalar< PColorMatrix< RComplex< PScalar<REAL> >, 2> > > LatticeColorMatrix2;
typedef OLattice< PScalar< PColorMatrix< RComplex< PScalar<REAL> >, 1> > > LatticeColorMatrix1;

typedef OLattice< PScalar< PColorVector< RComplex< PScalar<REAL> >, 3> > > LatticeColorVector3;
typedef OLattice< PScalar< PColorVector< RComplex< PScalar<REAL> >, 2> > > LatticeColorVector2;
typedef OLattice< PScalar< PColorVector< RComplex< PScalar<REAL> >, 1> > > LatticeColorVector1;

typedef OScalar< PScalar< PColorMatrix< RComplex< PScalar<REAL> >, 3> > > ColorMatrix3;
typedef OScalar< PScalar< PColorMatrix< RComplex< PScalar<REAL> >, 2> > > ColorMatrix2;
typedef OScalar< PScalar< PColorMatrix< RComplex< PScalar<REAL> >, 1> > > ColorMatrix1;

typedef OScalar< PScalar< PColorVector< RComplex< PScalar<REAL> >, 3> > > ColorVector3;
typedef OScalar< PScalar< PColorVector< RComplex< PScalar<REAL> >, 2> > > ColorVector2;
typedef OScalar< PScalar< PColorVector< RComplex< PScalar<REAL> >, 1> > > ColorVector1;

//
// Fixed precision
//
// REAL32 types
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, Nc>, 4> > LatticeDiracFermionF;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, 3>, 4> > LatticeDiracFermionF3;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, 2>, 4> > LatticeDiracFermionF2;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, 1>, 4> > LatticeDiracFermionF1;

typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, Nc>, 1> > LatticeStaggeredFermionF;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, 3>, 1> > LatticeStaggeredFermionF3;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, 2>, 1> > LatticeStaggeredFermionF2;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, 1>, 1> > LatticeStaggeredFermionF1;

typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL32> >, Nc>, 4> > LatticeDiracPropagatorF;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL32> >, 3>, 4> > LatticeDiracPropagatorF3;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL32> >, 2>, 4> > LatticeDiracPropagatorF2;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL32> >, 1>, 4> > LatticeDiracPropagatorF1;

typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, Nc>, Ns> > LatticeFermionF;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, 3>, Ns> > LatticeFermionF3;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, 2>, Ns> > LatticeFermionF2;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, 1>, Ns> > LatticeFermionF1;

typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, Nc>, Ns>>1 > > LatticeHalfFermionF;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, 3>, Ns>>1 > > LatticeHalfFermionF3;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, 2>, Ns>>1 > > LatticeHalfFermionF2;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, 1>, Ns>>1 > > LatticeHalfFermionF1;

typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL32> >, Nc>, Ns> > LatticePropagatorF;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL32> >, 3>, Ns> > LatticePropagatorF3;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL32> >, 2>, Ns> > LatticePropagatorF2;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL32> >, 1>, Ns> > LatticePropagatorF1;

typedef OLattice< PSpinMatrix< PScalar< RComplex< PScalar<REAL32> > >, Ns> > LatticeSpinMatrixF;
typedef OLattice< PSpinVector< PScalar< RComplex< PScalar<REAL32> > >, Ns> > LatticeSpinVectorF;

typedef OLattice< PScalar< PColorMatrix< RComplex< PScalar<REAL32> >, Nc> > > LatticeColorMatrixF;
typedef OLattice< PScalar< PColorMatrix< RComplex< PScalar<REAL32> >, 3> > > LatticeColorMatrixF3;
typedef OLattice< PScalar< PColorMatrix< RComplex< PScalar<REAL32> >, 2> > > LatticeColorMatrixF2;
typedef OLattice< PScalar< PColorMatrix< RComplex< PScalar<REAL32> >, 1> > > LatticeColorMatrixF1;

typedef OLattice< PScalar< PColorVector< RComplex< PScalar<REAL32> >, Nc> > > LatticeColorVectorF;
typedef OLattice< PScalar< PColorVector< RComplex< PScalar<REAL32> >, 3> > > LatticeColorVectorF3;
typedef OLattice< PScalar< PColorVector< RComplex< PScalar<REAL32> >, 2> > > LatticeColorVectorF2;
typedef OLattice< PScalar< PColorVector< RComplex< PScalar<REAL32> >, 1> > > LatticeColorVectorF1;

typedef OLattice< PScalar< PScalar< RComplex< PScalar<REAL32> > > > > LatticeComplexF;
typedef OLattice< PScalar< PScalar< RScalar< PScalar<REAL32> > > > > LatticeRealF;

typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, Nc>, 4> > DiracFermionF;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, 3>, 4> > DiracFermionF3;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, 2>, 4> > DiracFermionF2;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, 1>, 4> > DiracFermionF1;

typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, Nc>, 1> > StaggeredFermionF;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, 3>, 1> > StaggeredFermionF3;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, 2>, 1> > StaggeredFermionF2;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, 1>, 1> > StaggeredFermionF1;

typedef OScalar< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL32> >, Nc>, 4> > DiracPropagatorF;
typedef OScalar< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL32> >, 3>, 4> > DiracPropagatorF3;
typedef OScalar< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL32> >, 2>, 4> > DiracPropagatorF2;
typedef OScalar< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL32> >, 1>, 4> > DiracPropagatorF1;

typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, Nc>, Ns> > FermionF;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, 3>, Ns> > FermionF3;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, 2>, Ns> > FermionF2;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, 1>, Ns> > FermionF1;

typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, Nc>, Ns>>1 > > HalfFermionF;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, 3>, Ns>>1 > > HalfFermionF3;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, 2>, Ns>>1 > > HalfFermionF2;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, 1>, Ns>>1 > > HalfFermionF1;

typedef OScalar< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL32> >, Nc>, Ns> > PropagatorF;
typedef OScalar< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL32> >, 3>, Ns> > PropagatorF3;
typedef OScalar< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL32> >, 2>, Ns> > PropagatorF2;
typedef OScalar< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL32> >, 1>, Ns> > PropagatorF1;

typedef OScalar< PSpinVector< PScalar< RComplex< PScalar<REAL32> > >, Ns> > SpinVectorF;
typedef OScalar< PSpinMatrix< PScalar< RComplex< PScalar<REAL32> > >, Ns> > SpinMatrixF;

typedef OScalar< PScalar< PColorMatrix< RComplex< PScalar<REAL32> >, Nc> > > ColorMatrixF;
typedef OScalar< PScalar< PColorMatrix< RComplex< PScalar<REAL32> >, 3> > > ColorMatrixF3;
typedef OScalar< PScalar< PColorMatrix< RComplex< PScalar<REAL32> >, 2> > > ColorMatrixF2;
typedef OScalar< PScalar< PColorMatrix< RComplex< PScalar<REAL32> >, 1> > > ColorMatrixF1;

typedef OScalar< PScalar< PColorVector< RComplex< PScalar<REAL32> >, Nc> > > ColorVectorF;
typedef OScalar< PScalar< PColorVector< RComplex< PScalar<REAL32> >, 3> > > ColorVectorF3;
typedef OScalar< PScalar< PColorVector< RComplex< PScalar<REAL32> >, 2> > > ColorVectorF2;
typedef OScalar< PScalar< PColorVector< RComplex< PScalar<REAL32> >, 1> > > ColorVectorF1;

typedef OScalar< PScalar< PScalar< RComplex< PScalar<REAL32> > > > > ComplexF;
typedef OScalar< PScalar< PScalar< RScalar< PScalar<REAL32> > > > > RealF;

// REAL64 types
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, Nc>, 4> > LatticeDiracFermionD;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, 3>, 4> > LatticeDiracFermionD3;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, 2>, 4> > LatticeDiracFermionD2;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, 1>, 4> > LatticeDiracFermionD1;

typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, Nc>, 1> > LatticeStaggeredFermionD;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, 3>, 1> > LatticeStaggeredFermionD3;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, 2>, 1> > LatticeStaggeredFermionD2;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, 1>, 1> > LatticeStaggeredFermionD1;

typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL64> >, Nc>, 4> > LatticeDiracPropagatorD;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL64> >, 3>, 4> > LatticeDiracPropagatorD3;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL64> >, 2>, 4> > LatticeDiracPropagatorD2;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL64> >, 1>, 4> > LatticeDiracPropagatorD1;

typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, Nc>, Ns> > LatticeFermionD;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, 3>, Ns> > LatticeFermionD3;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, 2>, Ns> > LatticeFermionD2;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, 1>, Ns> > LatticeFermionD1;

typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, Nc>, Ns>>1 > > LatticeHalfFermionD;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, 3>, Ns>>1 > > LatticeHalfFermionD3;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, 2>, Ns>>1 > > LatticeHalfFermionD2;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, 1>, Ns>>1 > > LatticeHalfFermionD1;

typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL64> >, Nc>, Ns> > LatticePropagatorD;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL64> >, 3>, Ns> > LatticePropagatorD3;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL64> >, 2>, Ns> > LatticePropagatorD2;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL64> >, 1>, Ns> > LatticePropagatorD1;

typedef OLattice< PSpinMatrix< PScalar< RComplex< PScalar<REAL64> > >, Ns> > LatticeSpinMatrixD;
typedef OLattice< PSpinVector< PScalar< RComplex< PScalar<REAL64> > >, Ns> > LatticeSpinVectorD;

typedef OLattice< PScalar< PColorMatrix< RComplex< PScalar<REAL64> >, Nc> > > LatticeColorMatrixD;
typedef OLattice< PScalar< PColorMatrix< RComplex< PScalar<REAL64> >, 3> > > LatticeColorMatrixD3;
typedef OLattice< PScalar< PColorMatrix< RComplex< PScalar<REAL64> >, 2> > > LatticeColorMatrixD2;
typedef OLattice< PScalar< PColorMatrix< RComplex< PScalar<REAL64> >, 1> > > LatticeColorMatrixD1;

typedef OLattice< PScalar< PColorVector< RComplex< PScalar<REAL64> >, Nc> > > LatticeColorVectorD;
typedef OLattice< PScalar< PColorVector< RComplex< PScalar<REAL64> >, 3> > > LatticeColorVectorD3;
typedef OLattice< PScalar< PColorVector< RComplex< PScalar<REAL64> >, 2> > > LatticeColorVectorD2;
typedef OLattice< PScalar< PColorVector< RComplex< PScalar<REAL64> >, 1> > > LatticeColorVectorD1;

typedef OLattice< PScalar< PScalar< RComplex< PScalar<REAL64> > > > > LatticeComplexD;
typedef OLattice< PScalar< PScalar< RScalar< PScalar<REAL64> > > > > LatticeRealD;

typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, Nc>, 4> > DiracFermionD;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, 3>, 4> > DiracFermionD3;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, 2>, 4> > DiracFermionD2;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, 1>, 4> > DiracFermionD1;

typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, Nc>, 1> > StaggeredFermionD;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, 3>, 1> > StaggeredFermionD3;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, 2>, 1> > StaggeredFermionD2;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, 1>, 1> > StaggeredFermionD1;

typedef OScalar< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL64> >, Nc>, 4> > DiracPropagatorD;
typedef OScalar< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL64> >, 3>, 4> > DiracPropagatorD3;
typedef OScalar< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL64> >, 2>, 4> > DiracPropagatorD2;
typedef OScalar< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL64> >, 1>, 4> > DiracPropagatorD1;

typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, Nc>, Ns> > FermionD;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, 3>, Ns> > FermionD3;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, 2>, Ns> > FermionD2;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, 1>, Ns> > FermionD1;

typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, Nc>, Ns>>1 > > HalfFermionD;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, 3>, Ns>>1 > > HalfFermionD3;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, 2>, Ns>>1 > > HalfFermionD2;
typedef OScalar< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, 1>, Ns>>1 > > HalfFermionD1;

typedef OScalar< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL64> >, Nc>, Ns> > PropagatorD;
typedef OScalar< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL64> >, 3>, Ns> > PropagatorD3;
typedef OScalar< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL64> >, 2>, Ns> > PropagatorD2;
typedef OScalar< PSpinMatrix< PColorMatrix< RComplex< PScalar<REAL64> >, 1>, Ns> > PropagatorD1;

typedef OScalar< PScalar< PColorMatrix< RComplex< PScalar<REAL64> >, Nc> > > ColorMatrixD;
typedef OScalar< PScalar< PColorMatrix< RComplex< PScalar<REAL64> >, 3> > > ColorMatrixD3;
typedef OScalar< PScalar< PColorMatrix< RComplex< PScalar<REAL64> >, 2> > > ColorMatrixD2;
typedef OScalar< PScalar< PColorMatrix< RComplex< PScalar<REAL64> >, 1> > > ColorMatrixD1;

typedef OScalar< PScalar< PColorVector< RComplex< PScalar<REAL64> >, Nc> > > ColorVectorD;
typedef OScalar< PScalar< PColorVector< RComplex< PScalar<REAL64> >, 3> > > ColorVectorD3;
typedef OScalar< PScalar< PColorVector< RComplex< PScalar<REAL64> >, 2> > > ColorVectorD2;
typedef OScalar< PScalar< PColorVector< RComplex< PScalar<REAL64> >, 1> > > ColorVectorD1;

typedef OScalar< PScalar< PScalar< RComplex< PScalar<REAL64> > > > > ComplexD;
typedef OScalar< PScalar< PScalar< RScalar< PScalar<REAL64> > > > > RealD;

// Equivalent names
typedef Integer  Int;

typedef RealF  Real32;
typedef RealD  Real64;
typedef ComplexF  Complex32;
typedef ComplexD  Complex64;

typedef LatticeInteger  LatticeInt;


/*! @} */   // end of group defs

QDP_END_NAMESPACE();

