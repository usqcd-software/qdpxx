// -*- C++ -*-
// $Id: qdp_scalarvecsite_defs.h,v 1.10 2004-04-06 15:35:12 bjoo Exp $

/*! \file
 * \brief Type definitions for scalar/vector extensions-like architectures
 */

QDP_BEGIN_NAMESPACE(QDP);

#include <qdp_config.h>
#include "qdp_precision.h"

/*! \addtogroup defs Type definitions
 *
 * User constructed types made from QDP type compositional nesting.
 * The layout is suitable for a scalar/vector-like implementation. 
 * Namely, there are stripped-mined sites as the slowest varying index
 * with an additional inner-grid loop
 *
 * @{
 */


//----------------------------------------------------------------------
//! Gamma matrices are conveniently defined for this Ns
typedef GammaType<Ns> Gamma;

// For now, fix this inner-grid length to 4 or 2 depending on base precision. 
// This causes problems for doubles which I eventually must work out.
// Here, INNER_LOG is the log_2(INNER)
#define INNER_LEN (1 << INNER_LOG)

// Aliases for a scalarvec-like architecture

// Fixed fermion type
typedef OLattice< PSpinVector< PColorVector< RComplex< ILattice<REAL,INNER_LEN> >, Nc>, 4> > LatticeDiracFermion;
typedef OLattice< PSpinVector< PColorVector< RComplex< ILattice<REAL,INNER_LEN> >, Nc>, 1> > LatticeStaggeredFermion;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< ILattice<REAL,INNER_LEN> >, Nc>, 4> > LatticeDiracPropagator;

// Floating aliases
typedef OLattice< PScalar< PColorVector< RComplex< ILattice<REAL,INNER_LEN> >, Nc> > > LatticeColorVector;
typedef OLattice< PSpinVector< PScalar< RComplex< ILattice<REAL,INNER_LEN> > >, Ns> > LatticeSpinVector;
typedef OLattice< PScalar< PColorMatrix< RComplex< ILattice<REAL,INNER_LEN> >, Nc> > > LatticeColorMatrix;
typedef OLattice< PSpinMatrix< PScalar< RComplex< ILattice<REAL,INNER_LEN> > >, Ns> > LatticeSpinMatrix;
typedef OLattice< PSpinVector< PColorVector< RComplex< ILattice<REAL,INNER_LEN> >, Nc>, Ns> > LatticeFermion;
typedef OLattice< PSpinVector< PColorVector< RComplex< ILattice<REAL,INNER_LEN> >, Nc>, Ns>>1 > > LatticeHalfFermion;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex< ILattice<REAL,INNER_LEN> >, Nc>, Ns> > LatticePropagator;
typedef OLattice< PScalar< PScalar< RComplex< ILattice<REAL,INNER_LEN> > > > > LatticeComplex;

typedef OLattice< PScalar< PSeed < RScalar< ILattice<INTEGER32,INNER_LEN> > > > > LatticeSeed;
typedef OLattice< PScalar< PScalar< RScalar< ILattice<INTEGER32,INNER_LEN> > > > > LatticeInteger;
typedef OLattice< PScalar< PScalar< RScalar< ILattice<REAL,INNER_LEN> > > > > LatticeReal;
typedef OLattice< PScalar< PScalar< RScalar< ILattice<DOUBLE,INNER_LEN> > > > > LatticeDouble;
typedef OLattice< PScalar< PScalar< RScalar< ILattice<LOGICAL,INNER_LEN> > > > > LatticeBoolean;

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
typedef OScalar< PScalar< PScalar < RScalar< ILattice<REAL,INNER_LEN> > > > > ILatticeReal;
typedef OScalar< PScalar< PSeed < RScalar< ILattice<INTEGER32,INNER_LEN> > > > > ILatticeSeed;

// Fixed precision
typedef OLattice< PScalar< PColorMatrix< RComplex< ILattice<REAL32,INNER_LEN> >, Nc> > > LatticeColorMatrixF;
typedef OScalar< PScalar< PColorMatrix< RComplex< IScalar<REAL32> >, Nc> > > ColorMatrixF;
typedef OLattice< PSpinVector< PColorVector< RComplex< ILattice<REAL32,INNER_LEN> >, Nc>, Ns> > LatticeFermionF;
typedef OLattice< PSpinVector< PColorVector< RComplex< ILattice<REAL64,INNER_LEN> >, Nc>, Ns> > LatticeFermionD;

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

