// -*- C++ -*-
// $Id: defs.h,v 1.3 2002-10-12 04:10:15 edwards Exp $

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

#define INTEGER32 int
#define REAL32    float
#define REAL64    double
#define LOGICAL   bool

//! Gamma matrices are conveniently defined for this Ns
typedef GammaType<Ns> Gamma;

// Aliases for a scalar architecture
typedef OLattice< PScalar< PColorMatrix< RComplex<REAL32>, Nc> > > LatticeGauge;
typedef OLattice< PSpinVector< PColorVector< RComplex<REAL32>, Nc>, Ns> > LatticeFermion;
typedef OLattice< PSpinVector< PColorVector< RComplex<REAL32>, Nc>, Ns>>1 > > LatticeHalfFermion;
typedef OLattice< PSpinMatrix< PColorMatrix< RComplex<REAL32>, Nc>, Ns> > LatticePropagator;
typedef OLattice< PScalar< PColorMatrix< RComplex<REAL32>, Nc> > > LatticeColorMatrix;
typedef OLattice< PSpinMatrix< PScalar< RComplex<REAL32> >, Ns> > LatticeSpinMatrix;
typedef OLattice< PScalar< PScalar< RComplex<REAL32> > > > LatticeComplex;

typedef OLattice< PScalar< PSeed < RScalar<INTEGER32> > > > LatticeSeed;
typedef OLattice< PScalar< PScalar< RScalar<INTEGER32> > > > LatticeInteger;
typedef OLattice< PScalar< PScalar< RScalar<REAL32> > > > LatticeReal;
typedef OLattice< PScalar< PScalar< RScalar<REAL64> > > > LatticeDouble;
typedef OLattice< PScalar< PScalar< RScalar<LOGICAL> > > > LatticeBoolean;

typedef OScalar< PScalar< PColorMatrix< RComplex<REAL32>, Nc> > > Gauge;
typedef OScalar< PSpinVector< PColorVector< RComplex<REAL32>, Nc>, Ns> > Fermion;
typedef OScalar< PSpinMatrix< PColorMatrix< RComplex<REAL32>, Nc>, Ns> > Propagator;
typedef OScalar< PScalar< PScalar< RComplex<REAL32> > > > Complex;

typedef OScalar< PScalar< PSeed< RScalar<INTEGER32> > > > Seed;
typedef OScalar< PScalar< PScalar< RScalar<INTEGER32> > > > Integer;
typedef OScalar< PScalar< PScalar< RScalar<REAL32> > > > Real;
typedef OScalar< PScalar< PScalar< RScalar<REAL64> > > > Double;
typedef OScalar< PScalar< PScalar< RScalar<LOGICAL> > > > Boolean;

typedef OScalar< PScalar< PScalar< RComplex<REAL64> > > > DComplex;


// Duplicate names
typedef OScalar< PScalar< PColorMatrix< RComplex<REAL32>, Nc> > > ColorMatrix;
typedef OScalar< PScalar< PColorVector< RComplex<REAL32>, Nc> > > ColorVector;
typedef OScalar< PSpinMatrix< PScalar< RComplex<REAL32> >, Ns> > SpinMatrix;
typedef OScalar< PSpinVector< PScalar< RComplex<REAL32> >, Ns> > SpinVector;
typedef OScalar< PSpinVector< PColorVector< RComplex<REAL32>, Nc>, Ns> > ColorVectorSpinVector;
typedef OScalar< PSpinMatrix< PColorMatrix< RComplex<REAL32>, Nc>, Ns> > ColorMatrixSpinMatrix;

// Level below outer for internal convenience
typedef PScalar< PScalar< RScalar<REAL32> > > IntReal;
typedef PScalar< PScalar< RScalar<INTEGER32> > > IntInteger;
typedef PScalar< PScalar< RScalar<REAL64> > > IntDouble;
typedef PScalar< PScalar< RScalar<LOGICAL> > > IntBoolean;

/*! @} */   // end of group defs

QDP_END_NAMESPACE();

