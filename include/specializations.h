// -*- C++ -*-
// $Id: specializations.h,v 1.7 2003-04-20 04:03:34 edwards Exp $
//
// QDP data parallel interface
//

QDP_BEGIN_NAMESPACE(QDP);

//
// Conversion routines. These cannot be implicit conversion functions
// since they foul up the PETE defs in QDPOperators.h using primitive
// types
//

//! Make an int from an Integer
inline int 
toInt(const Integer& s) 
{
  return toInt(s.elem());
}

//! Make a float from a Real
inline float
toFloat(const Real& s) 
{
  return toFloat(s.elem());
}

//! Make a double from a Double
inline double
toDouble(const Double& s) 
{
  return toDouble(s.elem());
}

//! Make a bool from a Boolean
inline bool
toBool(const Boolean& s) 
{
  return toBool(s.elem());
}


//! Function overload read of  Integer
NmlReader& read(NmlReader& nml, const string& s, Integer& d);

//! Function overload read of  Real
NmlReader& read(NmlReader& nml, const string& s, Real& d);

//! Function overload read of  Double
NmlReader& read(NmlReader& nml, const string& s, Double& d);

//! Function overload read of  Complex
NmlReader& read(NmlReader& nml, const string& s, Complex& d);


//
// Return an equivalent QDP type given some simple machine type
//
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

// Construct simple double word
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


//
// Type constructors for QDP types within the type system. Namely,
// at some level like a primitive, sometimes scalar temporaries are needed
// These are the bottom most constructors given a machine type
//
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


// Internally used real scalars
template<>
struct RealScalar<int> {
  typedef REAL32  Type_t;
};

template<>
struct RealScalar<float> {
  typedef REAL32  Type_t;
};

template<>
struct RealScalar<double> {
  typedef REAL64  Type_t;
};



//
// Leaf constructors for simple machine types. These are a specialization over the
// default constructors. The point is to avoid wrapping the simple types in
// references which do not help much (primitive objects are word size!), 
// but get in the way of type computations.
//
template<>
struct CreateLeaf<int>
{
  typedef int Inp_t;
  typedef Integer  Leaf_t;
  inline static
  Leaf_t make(const Inp_t &a) { return Leaf_t(a); }
};

template<>
struct CreateLeaf<float>
{
  typedef float Inp_t;
  typedef Real  Leaf_t;
  inline static
  Leaf_t make(const Inp_t &a) { return Leaf_t(a); }
};

template<>
struct CreateLeaf<double>
{
  typedef double Inp_t;
  typedef Double  Leaf_t;
  inline static
  Leaf_t make(const Inp_t &a) { return Leaf_t(a); }
};

template<>
struct CreateLeaf<bool>
{
  typedef bool Inp_t;
  typedef Boolean  Leaf_t;
  inline static
  Leaf_t make(const Inp_t &a) { return Leaf_t(a); }
};


template<>
struct CreateLeaf<OScalar<IntReal> >
{
  typedef OScalar<IntReal> Leaf_t;
  inline static
  Leaf_t make(const OScalar<IntReal> &a) { return Leaf_t(a); }
};


template<>
struct CreateLeaf<OScalar<IntDouble> >
{
  typedef OScalar<IntDouble> Leaf_t;
  inline static
  Leaf_t make(const OScalar<IntDouble> &a) { return Leaf_t(a); }
};


template<>
struct CreateLeaf<OScalar<IntInteger> >
{
  typedef OScalar<IntInteger> Leaf_t;
  inline static
  Leaf_t make(const OScalar<IntInteger> &a) { return Leaf_t(a); }
};


template<>
struct CreateLeaf<OScalar<IntBoolean> >
{
  typedef OScalar<IntBoolean> Leaf_t;
  inline static
  Leaf_t make(const OScalar<IntBoolean> &a) { return Leaf_t(a); }
};




QDP_END_NAMESPACE();

