// -*- C++ -*-
// $Id: qdp_scalarvecsite_defs.h,v 1.1 2003-08-20 21:06:51 edwards Exp $

/*! \file
 * \brief Type definitions for scalar/vector extensions-like architectures
 */

QDP_BEGIN_NAMESPACE(QDP);

/*! \addtogroup defs Type definitions
 *
 * User constructed types made from QDP type compositional nesting.
 * The layout is suitable for a scalar/vector-like implementation. 
 * Namely, there are stripped-mined sites as the slowest varying index
 * with an additional inner-grid loop
 *
 * @{
 */

#if defined(SINGLE_PRECISION)
#undef SINGLE_PRECISION
#endif

#if defined(DOUBLE_PRECISION)
#undef DOUBLE_PRECISION
#endif

// Fix default precision
#if ! defined(BASE_PRECISION)
#define BASE_PRECISION SINGLE_PRECISION
#endif

// These are fixed precision versions
#define INTEGER32 int
#define REAL32    float
#define REAL64    double
#define LOGICAL   bool

// Set the base floating precision
#if BASE_PRECISION == SINGLE_PRECISION
// Use single precision for base precision
#define REAL      REAL32
#define DOUBLE    REAL64
#define INNER_LEN 4

#elif BASE_PRECISION == DOUBLE_PRECISION
// Use double precision for base precision
#define REAL      REAL64
#define DOUBLE    REAL64
#define INNER_LEN 2

#else
#error "Unknown BASE_PRECISION"
#endif


//----------------------------------------------------------------------
//! Gamma matrices are conveniently defined for this Ns
typedef GammaType<Ns> Gamma;

// For now, fix this inner-grid length to 4 or 2 depending on base precision. 
// This causes problems for doubles which I eventually must work out.
#define INNER INNER_LEN

// Aliases for a scalarvec-like architecture

// Fixed fermion type
typedef OLattice< PSpinVector< PColorVector< RComplex< ILattice<REAL,INNER> >, 3>, 4> > LatticeDiracFermion;
typedef OLattice< PSpinVector< PColorVector< RComplex< ILattice<REAL,INNER> >, 3>, 1> > LatticeStaggeredFermion;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< ILattice<REAL,INNER> >, 3>, 4> > LatticeDiracPropagator;

// Floating aliases
typedef OLattice< PScalar< PColorVector< RComplex< ILattice<REAL,INNER> >, Nc> > > LatticeColorVector;
typedef OLattice< PSpinVector< PScalar< RComplex< ILattice<REAL,INNER> > >, Ns> > LatticeSpinVector;
typedef OLattice< PScalar< PColorMatrix< RComplex< ILattice<REAL,INNER> >, Nc> > > LatticeColorMatrix;
typedef OLattice< PSpinMatrix< PScalar< RComplex< ILattice<REAL,INNER> > >, Ns> > LatticeSpinMatrix;
typedef OLattice< PSpinVector< PColorVector< RComplex< ILattice<REAL,INNER> >, Nc>, 1> > LatticeStaggeredFermion;
typedef OLattice< PSpinVector< PColorVector< RComplex< ILattice<REAL,INNER> >, Nc>, Ns> > LatticeFermion;
typedef OLattice< PSpinVector< PColorVector< RComplex< ILattice<REAL,INNER> >, Nc>, Ns>>1 > > LatticeHalfFermion;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< ILattice<REAL,INNER> >, Nc>, Ns> > LatticePropagator;
typedef OLattice< PScalar< PScalar< RComplex< ILattice<REAL,INNER> > > > > LatticeComplex;

typedef OLattice< PScalar< PSeed < RScalar< ILattice<INTEGER32,INNER> > > > > LatticeSeed;
typedef OLattice< PScalar< PScalar< RScalar< ILattice<INTEGER32,INNER> > > > > LatticeInteger;
typedef OLattice< PScalar< PScalar< RScalar< ILattice<REAL,INNER> > > > > LatticeReal;
typedef OLattice< PScalar< PScalar< RScalar< ILattice<DOUBLE,INNER> > > > > LatticeDouble;
typedef OLattice< PScalar< PScalar< RScalar< ILattice<LOGICAL,INNER> > > > > LatticeBoolean;

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

// Fixed precision
typedef OLattice< PScalar< PColorMatrix< RComplex< ILattice<REAL32,INNER> >, Nc> > > LatticeColorMatrixF;
typedef OScalar< PScalar< PColorMatrix< RComplex< IScalar<REAL32> >, Nc> > > ColorMatrixF;


typedef OScalar< PScalar< PScalar< RScalar< IScalar<REAL32> > > > > Real32;
typedef OScalar< PScalar< PScalar< RScalar< IScalar<REAL64> > > > > Real64;

// Equivalent names
typedef Integer  Int;

typedef Real32  RealF;
typedef Real64  RealD;

typedef LatticeInteger  LatticeInt;


/*! @} */   // end of group defs

QDP_END_NAMESPACE();

