// -*- C++ -*-
// $Id: simpleword.h,v 1.2 2002-09-14 19:48:26 edwards Exp $
//
// QDP data parallel interface
//

#include <cmath>

QDP_BEGIN_NAMESPACE(QDP);

// All these are explicit to avoid any general template clashes

//! dest = 0
inline
void zero(int& dest) 
{
  dest = 0;
}

//! dest = 0
inline
void zero(float& dest) 
{
  dest = 0;
}

//! dest = 0
inline
void zero(double& dest) 
{
  dest = 0;
}

//! No bool(dest) = 0


//! d = (mask) ? s1 : d;
inline
void copymask(int& d, const bool& mask, const int& s1) 
{
  if (mask)
    d = s1;
}

//! d = (mask) ? s1 : d;
inline
void copymask(float& d, const bool& mask, const float& s1) 
{
  if (mask)
    d = s1;
}

//! d = (mask) ? s1 : d;
inline
void copymask(double& d, const bool& mask, const double& s1) 
{
  if (mask)
    d = s1;
}

//! dest  = random  
template<class T1, class T2>
inline void
fill_random(float& d, T1& seed, T2& skewed_seed, const T1& seed_mult)
{
  d = float(RNG::sranf(seed, skewed_seed, seed_mult));
}

//! dest  = random  
template<class T1, class T2>
inline void
fill_random(double& d, T1& seed, T2& skewed_seed, const T1& seed_mult)
{
  d = double(RNG::sranf(seed, skewed_seed, seed_mult));
}



//! dest [float type] = source [int type]
inline
void cast_rep(float& d, const int& s1)
{
  d = float(s1);
}

//! dest [float type] = source [float type]
inline
void cast_rep(float& d, const float& s1)
{
  d = float(s1);
}

//! dest [float type] = source [double type]
inline
void cast_rep(float& d, const double& s1)
{
  d = float(s1);
}


//! dest [float type] = source [int type]
inline
void cast_rep(double& d, const int& s1)
{
  d = double(s1);
}

//! dest [double type] = source [float type]
inline
void cast_rep(double& d, const float& s1)
{
  d = double(s1);
}

//! dest [double type] = source [double type]
inline
void cast_rep(double& d, const double& s1)
{
  d = double(s1);
}




//-------------------------------------------------------
// Functions

// Conjugate
inline 
float conj(const float& l)
{
  return l;
}

// Conjugate
inline 
double conj(const double& l)
{
  return l;
}

// Conjugate
inline 
int conj(const int& l)
{
  return l;
}



// TRACE
// trace = Trace(source1)
inline 
float trace(const float& s1)
{
  return s1;
}


// trace = Trace(source1)
inline 
double trace(const double& s1)
{
  return s1;
}


// trace = Trace(source1)
inline 
int trace(const int& s1)
{
  return s1;
}


// Global sum over site indices only
inline
int sum(const int& s1)
{
  return s1;
}

inline
int localNorm2(const int& s1)
{
  return s1*s1;
}

inline
int localInnerproduct(const int& s1, const int& s2)
{
  return s1*s2;
}

inline
double sum(const float& s1)
{
  return double(s1);
}

inline
double localNorm2(const float& s1)
{
  return double(s1*s1);
}

inline
double localInnerproduct(const float& s1, const float& s2)
{
  return double(s1*s2);
}

inline
double sum(const double& s1)
{
  return s1;
}

inline
double localNorm2(const double& s1)
{
  return s1*s1;
}

inline
double localInnerproduct(const double& s1, const double& s2)
{
  return s1*s2;
}




//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

struct UnaryReturn<int, FnSum > {
  typedef int  Type_t;
};

struct UnaryReturn<int, FnNorm2 > {
  typedef int  Type_t;
};

struct UnaryReturn<int, FnInnerproduct > {
  typedef int  Type_t;
};

struct UnaryReturn<int, FnInnerproductReal > {
  typedef int  Type_t;
};

struct UnaryReturn<int, FnLocalNorm2 > {
  typedef int  Type_t;
};

struct UnaryReturn<int, FnLocalInnerproduct > {
  typedef int  Type_t;
};

struct UnaryReturn<int, FnLocalInnerproductReal > {
  typedef int  Type_t;
};



struct UnaryReturn<float, FnSum > {
  typedef double  Type_t;
};

struct UnaryReturn<float, FnNorm2 > {
  typedef double  Type_t;
};

struct UnaryReturn<float, FnInnerproduct > {
  typedef double  Type_t;
};

struct UnaryReturn<float, FnInnerproductReal > {
  typedef double  Type_t;
};

struct UnaryReturn<float, FnLocalNorm2 > {
  typedef float  Type_t;
};

struct UnaryReturn<float, FnLocalInnerproduct > {
  typedef float  Type_t;
};

struct UnaryReturn<float, FnLocalInnerproductReal > {
  typedef float  Type_t;
};


struct UnaryReturn<double, FnSum > {
  typedef double  Type_t;
};

struct UnaryReturn<double, FnNorm2 > {
  typedef double  Type_t;
};

struct UnaryReturn<double, FnInnerproduct > {
  typedef double  Type_t;
};

struct UnaryReturn<double, FnInnerproductReal > {
  typedef double  Type_t;
};

struct UnaryReturn<double, FnLocalNorm2 > {
  typedef double  Type_t;
};

struct UnaryReturn<double, FnLocalInnerproduct > {
  typedef double  Type_t;
};

struct UnaryReturn<double, FnLocalInnerproductReal > {
  typedef double  Type_t;
};



struct UnaryReturn<int, FnSliceSum > {
  typedef int  Type_t;
};

struct UnaryReturn<float, FnSliceSum > {
  typedef double  Type_t;
};

struct UnaryReturn<double, FnSliceSum > {
  typedef double  Type_t;
};



struct BinaryReturn<int, int, OpAssign > {
  typedef int  Type_t;
};
 
struct BinaryReturn<int, int, OpAddAssign > {
  typedef int  Type_t;
};
 
struct BinaryReturn<int, int, OpSubtractAssign > {
  typedef int  Type_t;
};
 
struct BinaryReturn<int, int, OpMultiplyAssign > {
  typedef int  Type_t;
};
 
struct BinaryReturn<int, int, OpDivideAssign > {
  typedef int  Type_t;
};
 
struct BinaryReturn<int, int, OpModAssign > {
  typedef int  Type_t;
};
 
struct BinaryReturn<int, int, OpBitwiseOrAssign > {
  typedef int  Type_t;
};
 
struct BinaryReturn<int, int, OpBitwiseAndAssign > {
  typedef int  Type_t;
};
 
struct BinaryReturn<int, int, OpBitwiseXorAssign > {
  typedef int  Type_t;
};
 
struct BinaryReturn<int, int, OpLeftShiftAssign > {
  typedef int  Type_t;
};
 
struct BinaryReturn<int, int, OpRightShiftAssign > {
  typedef int  Type_t;
};
 

struct BinaryReturn<float, float, OpAssign > {
  typedef float  Type_t;
};
 
struct BinaryReturn<float, float, OpAddAssign > {
  typedef float  Type_t;
};
 
struct BinaryReturn<float, float, OpSubtractAssign > {
  typedef float  Type_t;
};
 
struct BinaryReturn<float, float, OpMultiplyAssign > {
  typedef float  Type_t;
};
 
struct BinaryReturn<float, float, OpDivideAssign > {
  typedef float  Type_t;
};
 
struct BinaryReturn<float, float, OpModAssign > {
  typedef float  Type_t;
};
 
struct BinaryReturn<float, float, OpBitwiseOrAssign > {
  typedef float  Type_t;
};
 
struct BinaryReturn<float, float, OpBitwiseAndAssign > {
  typedef float  Type_t;
};
 
struct BinaryReturn<float, float, OpBitwiseXorAssign > {
  typedef float  Type_t;
};
 
struct BinaryReturn<float, float, OpLeftShiftAssign > {
  typedef float  Type_t;
};
 
struct BinaryReturn<float, float, OpRightShiftAssign > {
  typedef float  Type_t;
};
 

struct BinaryReturn<double, double, OpAssign > {
  typedef double  Type_t;
};
 
struct BinaryReturn<double, double, OpAddAssign > {
  typedef double  Type_t;
};
 
struct BinaryReturn<double, double, OpSubtractAssign > {
  typedef double  Type_t;
};
 
struct BinaryReturn<double, double, OpMultiplyAssign > {
  typedef double  Type_t;
};
 
struct BinaryReturn<double, double, OpDivideAssign > {
  typedef double  Type_t;
};
 
struct BinaryReturn<double, double, OpModAssign > {
  typedef double  Type_t;
};
 
struct BinaryReturn<double, double, OpBitwiseOrAssign > {
  typedef double  Type_t;
};
 
struct BinaryReturn<double, double, OpBitwiseAndAssign > {
  typedef double  Type_t;
};
 
struct BinaryReturn<double, double, OpBitwiseXorAssign > {
  typedef double  Type_t;
};
 
struct BinaryReturn<double, double, OpLeftShiftAssign > {
  typedef double  Type_t;
};
 
struct BinaryReturn<double, double, OpRightShiftAssign > {
  typedef double  Type_t;
};
 

struct BinaryReturn<bool, bool, OpAssign > {
  typedef bool  Type_t;
};
 

QDP_END_NAMESPACE();
