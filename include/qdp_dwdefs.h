// -*- C++ -*-
// $Id: qdp_dwdefs.h,v 1.1 2003-10-17 20:22:53 edwards Exp $

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

const int Ls = 8;   // HACK - FOR now fixed DW index here


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

// New guy beyond qdp_defs.h
typedef OLattice< PDWVector< PSpinVector< PColorVector< RComplex<REAL>, 3>, 4>, Ls> > LatticeDWFermion;
typedef OScalar< PDWVector< PSpinVector< PColorVector< RComplex<REAL>, 3>, 4>, Ls> > DWFermion;

// Fixed fermion type
typedef OLattice< PScalar< PSpinVector< PColorVector< RComplex<REAL>, 3>, 4> > > LatticeDiracFermion;
typedef OLattice< PScalar< PSpinVector< PColorVector< RComplex<REAL>, 3>, 1> > > LatticeStaggeredFermion;
typedef OLattice< PScalar< PSpinMatrix< PColorMatrix< RComplex<REAL>, 3>, 4> > > LatticeDiracPropagator;

// Floating aliases
typedef OLattice< PScalar< PScalar< PColorVector< RComplex<REAL>, Nc> > > > LatticeColorVector;
typedef OLattice< PScalar< PSpinVector< PScalar< RComplex<REAL> >, Ns> > > LatticeSpinVector;
typedef OLattice< PScalar< PScalar< PColorMatrix< RComplex<REAL>, Nc> > > > LatticeColorMatrix;
typedef OLattice< PScalar< PSpinMatrix< PScalar< RComplex<REAL> >, Ns> > > LatticeSpinMatrix;
typedef OLattice< PScalar< PSpinVector< PColorVector< RComplex<REAL>, Nc>, 1> > > LatticeStaggeredFermion;
typedef OLattice< PScalar< PSpinVector< PColorVector< RComplex<REAL>, Nc>, Ns> > > LatticeFermion;
typedef OLattice< PScalar< PSpinVector< PColorVector< RComplex<REAL>, Nc>, Ns>>1 > > > LatticeHalfFermion;
typedef OLattice< PScalar< PSpinMatrix< PColorMatrix< RComplex<REAL>, Nc>, Ns> > > LatticePropagator;
typedef OLattice< PScalar< PScalar< PScalar< RComplex<REAL> > > > > LatticeComplex;

typedef OLattice< PScalar< PScalar< PSeed < RScalar<INTEGER32> > > > > LatticeSeed;
typedef OLattice< PScalar< PScalar< PScalar< RScalar<INTEGER32> > > > > LatticeInteger;
typedef OLattice< PScalar< PScalar< PScalar< RScalar<REAL> > > > > LatticeReal;
typedef OLattice< PScalar< PScalar< PScalar< RScalar<DOUBLE> > > > > LatticeDouble;
typedef OLattice< PScalar< PScalar< PScalar< RScalar<LOGICAL> > > > > LatticeBoolean;

typedef OScalar< PScalar< PScalar< PColorVector< RComplex<REAL>, Nc> > > > ColorVector;
typedef OScalar< PScalar< PScalar< PColorMatrix< RComplex<REAL>, Nc> > > > ColorMatrix;
typedef OScalar< PScalar< PSpinVector< PScalar< RComplex<REAL> >, Ns> > > SpinVector;
typedef OScalar< PScalar< PSpinMatrix< PScalar< RComplex<REAL> >, Ns> > > SpinMatrix;
typedef OScalar< PScalar< PSpinVector< PColorVector< RComplex<REAL>, Nc>, 1> > > StaggeredFermion;
typedef OScalar< PScalar< PSpinVector< PColorVector< RComplex<REAL>, Nc>, Ns> > > Fermion;
typedef OScalar< PScalar< PSpinVector< PColorVector< RComplex<REAL>, Nc>, Ns>>1 > > > HalfFermion;
typedef OScalar< PScalar< PSpinMatrix< PColorMatrix< RComplex<REAL>, Nc>, Ns> > > Propagator;
typedef OScalar< PScalar< PScalar< PScalar< RComplex<REAL> > > > > Complex;

typedef OScalar< PScalar< PScalar< PSeed< RScalar<INTEGER32> > > > > Seed;
typedef OScalar< PScalar< PScalar< PScalar< RScalar<INTEGER32> > > > > Integer;
typedef OScalar< PScalar< PScalar< PScalar< RScalar<REAL> > > > > Real;
typedef OScalar< PScalar< PScalar< PScalar< RScalar<DOUBLE> > > > > Double;
typedef OScalar< PScalar< PScalar< PScalar< RScalar<LOGICAL> > > > > Boolean;

typedef OScalar< PScalar< PScalar< PScalar< RComplex<DOUBLE> > > > > DComplex;


// Other useful names
typedef OScalar< PScalar< PSpinVector< PColorVector< RComplex<REAL>, Nc>, Ns> > > ColorVectorSpinVector;
typedef OScalar< PScalar< PSpinMatrix< PColorMatrix< RComplex<REAL>, Nc>, Ns> > > ColorMatrixSpinMatrix;

// Level below outer for internal convenience
typedef PScalar< PScalar< PScalar< RScalar<REAL> > > > IntReal;
typedef PScalar< PScalar< PScalar< RScalar<INTEGER32> > > > IntInteger;
typedef PScalar< PScalar< PScalar< RScalar<DOUBLE> > > > IntDouble;
typedef PScalar< PScalar< PScalar< RScalar<LOGICAL> > > > IntBoolean;

// Odd-ball to support random numbers
typedef Real ILatticeReal;
typedef Seed ILatticeSeed;

// Fixed precision
typedef OLattice< PScalar< PScalar< PColorMatrix< RComplex<REAL32>, Nc> > > > LatticeColorMatrixF;
typedef OScalar< PScalar< PScalar< PColorMatrix< RComplex<REAL32>, Nc> > > > ColorMatrixF;


typedef OScalar< PScalar< PScalar< PScalar< RScalar<REAL32> > > > > Real32;
typedef OScalar< PScalar< PScalar< PScalar< RScalar<REAL64> > > > > Real64;
typedef OScalar< PScalar< PScalar< PScalar< RComplex<REAL32> > > > > Complex32;
typedef OScalar< PScalar< PScalar< PScalar< RComplex<REAL64> > > > > Complex64;

// Equivalent names
typedef Integer  Int;

typedef Real32  RealF;
typedef Real64  RealD;

typedef LatticeInteger  LatticeInt;


/*! @} */   // end of group defs

QDP_END_NAMESPACE();

