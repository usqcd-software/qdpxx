// -*- C++ -*-
// $Id: qdp_dwdefs.h,v 1.10 2004-04-06 15:35:12 bjoo Exp $

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
typedef OScalar< PSpinMatrix< PColorMatrix< RComplex< PScalar<DOUBLE> >, Nc>, Ns> > PropagatorD;
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
typedef OLattice< PScalar< PColorMatrix< RComplex< PScalar<REAL> >, 3> > > LatticeColorMatrix3;
typedef OScalar< PScalar< PColorMatrix< RComplex< PScalar<REAL> >, 3> > > ColorMatrix3;
typedef OLattice< PScalar< PColorMatrix< RComplex< PScalar<REAL> >, 2> > > LatticeColorMatrix2;
typedef OScalar< PScalar< PColorMatrix< RComplex< PScalar<REAL> >, 2> > > ColorMatrix2;
typedef OLattice< PScalar< PColorMatrix< RComplex< PScalar<REAL> >, 1> > > LatticeColorMatrix1;
typedef OScalar< PScalar< PColorMatrix< RComplex< PScalar<REAL> >, 1> > > ColorMatrix1;

// Fixed precision
typedef OLattice< PScalar< PColorMatrix< RComplex< PScalar<REAL32> >, Nc> > > LatticeColorMatrixF;
typedef OLattice< PScalar< PColorMatrix< RComplex< PScalar<REAL32> >, 3> > > LatticeColorMatrixF3;
typedef OLattice< PScalar< PColorMatrix< RComplex< PScalar<REAL32> >, 2> > > LatticeColorMatrixF2;
typedef OLattice< PScalar< PColorMatrix< RComplex< PScalar<REAL32> >, 1> > > LatticeColorMatrixF1;
typedef OScalar< PScalar< PColorMatrix< RComplex< PScalar<REAL32> >, Nc> > > ColorMatrixF;
typedef OScalar< PScalar< PColorMatrix< RComplex< PScalar<REAL32> >, 3> > > ColorMatrixF3;
typedef OScalar< PScalar< PColorMatrix< RComplex< PScalar<REAL32> >, 2> > > ColorMatrixF2;
typedef OScalar< PScalar< PColorMatrix< RComplex< PScalar<REAL32> >, 1> > > ColorMatrixF1;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL32> >, Nc>, Ns> > LatticeFermionF;

typedef OLattice< PScalar< PColorMatrix< RComplex< PScalar<REAL64> >, Nc> > > LatticeColorMatrixD;
typedef OLattice< PScalar< PColorMatrix< RComplex< PScalar<REAL64> >, 3> > > LatticeColorMatrixD3;
typedef OLattice< PScalar< PColorMatrix< RComplex< PScalar<REAL64> >, 2> > > LatticeColorMatrixD2;
typedef OLattice< PScalar< PColorMatrix< RComplex< PScalar<REAL64> >, 1> > > LatticeColorMatrixD1;
typedef OScalar< PScalar< PColorMatrix< RComplex< PScalar<REAL64> >, Nc> > > ColorMatrixD;
typedef OScalar< PScalar< PColorMatrix< RComplex< PScalar<REAL64> >, 3> > > ColorMatrixD3;
typedef OScalar< PScalar< PColorMatrix< RComplex< PScalar<REAL64> >, 2> > > ColorMatrixD2;
typedef OScalar< PScalar< PColorMatrix< RComplex< PScalar<REAL64> >, 1> > > ColorMatrixD1;
typedef OLattice< PSpinVector< PColorVector< RComplex< PScalar<REAL64> >, Nc>, Ns> > LatticeFermionD;


typedef OScalar< PScalar< PScalar< RScalar< PScalar<REAL32> > > > > Real32;
typedef OScalar< PScalar< PScalar< RScalar< PScalar<REAL64> > > > > Real64;
typedef OScalar< PScalar< PScalar< RComplex< PScalar<REAL32> > > > > Complex32;
typedef OScalar< PScalar< PScalar< RComplex< PScalar<REAL64> > > > > Complex64;

// Equivalent names
typedef Integer  Int;

typedef Real32  RealF;
typedef Real64  RealD;

typedef LatticeInteger  LatticeInt;


/*! @} */   // end of group defs

QDP_END_NAMESPACE();

