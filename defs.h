// -*- C++ -*-
// $Id: defs.h,v 1.1 2002-09-12 18:22:16 edwards Exp $
//
// QDP data parallel interface
//
QDP_BEGIN_NAMESPACE(QDP);

#define INTEGER32 int
#define REAL32    float
#define REAL64    double
#define LOGICAL   bool

//! Gamma matrices are conveniently defined for this Ns
typedef GammaType<Ns> Gamma;

// Aliases for a scalar architecture
typedef OLattice< PScalar< PColorMatrix< RComplex<REAL32>, Nc> > > LatticeGauge;
//typedef OLattice< PScalar< PMatrix< RComplex<REAL32>, Nc> > > LatticeGauge;
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

  // Duplicate names
typedef OScalar< PScalar< PColorMatrix< RComplex<REAL32>, Nc> > > ColorMatrix;
typedef OScalar< PScalar< PColorVector< RComplex<REAL32>, Nc> > > ColorVector;
typedef OScalar< PSpinMatrix< PScalar< RComplex<REAL32> >, Ns> > SpinMatrix;
typedef OScalar< PSpinVector< PScalar< RComplex<REAL32> >, Ns> > SpinVector;
typedef OScalar< PSpinVector< PColorVector< RComplex<REAL32>, Nc>, Ns> > ColorVectorSpinVector;
typedef OScalar< PSpinMatrix< PColorMatrix< RComplex<REAL32>, Nc>, Ns> > ColorMatrixSpinMatrix;

typedef OScalar< PScalar< PSeed< RScalar<INTEGER32> > > > Seed;
typedef OScalar< PScalar< PScalar< RScalar<INTEGER32> > > > Integer;
typedef OScalar< PScalar< PScalar< RScalar<REAL32> > > > Real;
typedef OScalar< PScalar< PScalar< RScalar<REAL64> > > > Double;
typedef OScalar< PScalar< PScalar< RScalar<LOGICAL> > > > Boolean;

typedef OScalar< PScalar< PScalar< RComplex<REAL64> > > > DComplex;


typedef PScalar< PScalar< RScalar<REAL32> > > IntReal;
typedef PScalar< PScalar< RScalar<INTEGER32> > > IntInteger;
typedef PScalar< PScalar< RScalar<REAL64> > > IntDouble;
typedef PScalar< PScalar< RScalar<LOGICAL> > > IntBoolean;


// Construct simple float word
template<>
struct SimpleScalar<float>
{
  typedef Real   Type_t;
};

// Construct simple float word
template<>
struct SimpleScalar<int>
{
  typedef Integer   Type_t;
};

// Construct simple float word
template<>
struct SimpleScalar<double>
{
  typedef Double   Type_t;
};

// Construct simple boolean word
template<>
struct SimpleScalar<bool>
{
  typedef Boolean   Type_t;
};


// Construct simple float word
template<>
struct InternalScalar<float>
{
  typedef float  Type_t;
};

// Construct simple int word
template<>
struct InternalScalar<int>
{
  typedef int   Type_t;
};

// Construct simple double word
template<>
struct InternalScalar<double>
{
  typedef double   Type_t;
};

// Construct simple boolean word
template<>
struct InternalScalar<bool>
{
  typedef bool  Type_t;
};



struct CreateLeaf<int>
{
  typedef int Inp_t;
  typedef Integer  Leaf_t;
  inline static
  Leaf_t make(const Inp_t &a) { return Leaf_t(a); }
};

struct CreateLeaf<float>
{
  typedef float Inp_t;
  typedef Real  Leaf_t;
  inline static
  Leaf_t make(const Inp_t &a) { return Leaf_t(a); }
};

struct CreateLeaf<double>
{
  typedef double Inp_t;
  typedef Double  Leaf_t;
  inline static
  Leaf_t make(const Inp_t &a) { return Leaf_t(a); }
};

struct CreateLeaf<bool>
{
  typedef bool Inp_t;
  typedef Boolean  Leaf_t;
  inline static
  Leaf_t make(const Inp_t &a) { return Leaf_t(a); }
};


QDP_END_NAMESPACE();

