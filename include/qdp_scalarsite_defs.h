// -*- C++ -*-
// $Id: qdp_scalarsite_defs.h,v 1.5 2003-09-23 16:54:12 edwards Exp $

/*! \file
 * \brief Type definitions for scalar-site like architectures
 */

QDP_BEGIN_NAMESPACE(QDP);

#include <qdp_config.h>

/*! \addtogroup defs Type definitions
 *
 * User constructed types made from QDP type compositional nesting.
 * The layout is suitable for a scalar-like implementation. Namely,
 * sites are the slowest varying index.
 *
 * @{
 */

// Fix default precision
#if ! defined(BASE_PRECISION)
#define BASE_PRECISION 32
#endif

// These are fixed precision versions
#define INTEGER32 int
#define REAL32    float
#define REAL64    double
#define LOGICAL   bool

// Set the base floating precision
#if BASE_PRECISION == 32
// Use single precision for base precision
#define REAL      REAL32
#define DOUBLE    REAL64

#elif BASE_PRECISION == 64
// Use double precision for base precision
#define REAL      REAL64
#define DOUBLE    REAL64

#else
#error "Unknown BASE_PRECISION"
#endif


//----------------------------------------------------------------------
//! Gamma matrices are conveniently defined for this Ns
typedef GammaType<Ns> Gamma;


// Aliases for a scalar architecture

// Fixed fermion type
typedef OLattice< PSpinVector< PColorVector< RComplex< IScalar<REAL> >, 3>, 4> > LatticeDiracFermion;
typedef OLattice< PSpinVector< PColorVector< RComplex< IScalar<REAL> >, 3>, 1> > LatticeStaggeredFermion;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< IScalar<REAL> >, 3>, 4> > LatticeDiracPropagator;

// Floating aliases
typedef OLattice< PScalar< PColorVector< RComplex< IScalar<REAL> >, Nc> > > LatticeColorVector;
typedef OLattice< PSpinVector< PScalar< RComplex< IScalar<REAL> > >, Ns> > LatticeSpinVector;
typedef OLattice< PScalar< PColorMatrix< RComplex< IScalar<REAL> >, Nc> > > LatticeColorMatrix;
typedef OLattice< PSpinMatrix< PScalar< RComplex< IScalar<REAL> > >, Ns> > LatticeSpinMatrix;
typedef OLattice< PSpinVector< PColorVector< RComplex< IScalar<REAL> >, Nc>, 1> > LatticeStaggeredFermion;
typedef OLattice< PSpinVector< PColorVector< RComplex< IScalar<REAL> >, Nc>, Ns> > LatticeFermion;
typedef OLattice< PSpinVector< PColorVector< RComplex< IScalar<REAL> >, Nc>, Ns>>1 > > LatticeHalfFermion;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< IScalar<REAL> >, Nc>, Ns> > LatticePropagator;
typedef OLattice< PScalar< PScalar< RComplex< IScalar<REAL> > > > > LatticeComplex;

typedef OLattice< PScalar< PSeed < RScalar< IScalar<INTEGER32> > > > > LatticeSeed;
typedef OLattice< PScalar< PScalar< RScalar< IScalar<INTEGER32> > > > > LatticeInteger;
typedef OLattice< PScalar< PScalar< RScalar< IScalar<REAL> > > > > LatticeReal;
typedef OLattice< PScalar< PScalar< RScalar< IScalar<DOUBLE> > > > > LatticeDouble;
typedef OLattice< PScalar< PScalar< RScalar< IScalar<LOGICAL> > > > > LatticeBoolean;

typedef OScalar< PScalar< PColorVector< RComplex< IScalar<REAL> >, Nc> > > ColorVector;
typedef OScalar< PScalar< PColorMatrix< RComplex< IScalar<REAL> >, Nc> > > ColorMatrix;
typedef OScalar< PSpinVector< PScalar< RComplex< IScalar<REAL> > >, Ns> > SpinVector;
typedef OScalar< PSpinMatrix< PScalar< RComplex< IScalar<REAL> > >, Ns> > SpinMatrix;
typedef OScalar< PSpinVector< PColorVector< RComplex< IScalar<REAL> >, Nc>, 1> > StaggeredFermion;
typedef OScalar< PSpinVector< PColorVector< RComplex< IScalar<REAL> >, Nc>, Ns> > Fermion;
typedef OScalar< PSpinVector< PColorVector< RComplex< IScalar<REAL> >, Nc>, Ns>>1 > > HalfFermion;
typedef OScalar< PSpinMatrix< PColorMatrix< RComplex< IScalar<REAL> >, Nc>, Ns> > Propagator;
typedef OScalar< PScalar< PScalar< RComplex< IScalar<REAL> > > > > Complex;

typedef OScalar< PScalar< PSeed< RScalar< IScalar<INTEGER32> > > > > Seed;
typedef OScalar< PScalar< PScalar< RScalar< IScalar<INTEGER32> > > > > Integer;
typedef OScalar< PScalar< PScalar< RScalar< IScalar<REAL> > > > > Real;
typedef OScalar< PScalar< PScalar< RScalar< IScalar<DOUBLE> > > > > Double;
typedef OScalar< PScalar< PScalar< RScalar< IScalar<LOGICAL> > > > > Boolean;

typedef OScalar< PScalar< PScalar< RComplex< IScalar<DOUBLE> > > > > DComplex;


// Other useful names
typedef OScalar< PSpinVector< PColorVector< RComplex< IScalar<REAL> >, Nc>, Ns> > ColorVectorSpinVector;
typedef OScalar< PSpinMatrix< PColorMatrix< RComplex< IScalar<REAL> >, Nc>, Ns> > ColorMatrixSpinMatrix;

// Level below outer for internal convenience
typedef PScalar< PScalar< RScalar< IScalar<REAL> > > > IntReal;
typedef PScalar< PScalar< RScalar< IScalar<INTEGER32> > > > IntInteger;
typedef PScalar< PScalar< RScalar< IScalar<DOUBLE> > > > IntDouble;
typedef PScalar< PScalar< RScalar< IScalar<LOGICAL> > > > IntBoolean;

// Odd-ball to support random numbers
typedef Real ILatticeReal;
typedef Seed ILatticeSeed;

// Fixed precision
typedef OLattice< PScalar< PColorMatrix< RComplex< IScalar<REAL32> >, Nc> > > LatticeColorMatrixF;
typedef OScalar< PScalar< PColorMatrix< RComplex< IScalar<REAL32> >, Nc> > > ColorMatrixF;


typedef OScalar< PScalar< PScalar< RScalar< IScalar<REAL32> > > > > Real32;
typedef OScalar< PScalar< PScalar< RScalar< IScalar<REAL64> > > > > Real64;
typedef OScalar< PScalar< PScalar< RComplex< IScalar<REAL32> > > > > Complex32;
typedef OScalar< PScalar< PScalar< RComplex< IScalar<REAL64> > > > > Complex64;

// Equivalent names
typedef Integer  Int;

typedef Real32  RealF;
typedef Real64  RealD;

typedef LatticeInteger  LatticeInt;


/*! @} */   // end of group defs

QDP_END_NAMESPACE();

