// -*- C++ -*-
// $Id: forward.h,v 1.1 2002-09-12 18:22:16 edwards Exp $
//
// Forward declarations for QDP
//

QDP_BEGIN_NAMESPACE(QDP);

// Forward declarations
namespace RNG 
{
//  float sranf(Seed&, Seed&, const Seed&);
};

  
// Word
class Logical;
template<class T> class Word;
class Integer32;
class Real64;
class Real32;

//typedef int Integer32;
//typedef float Real32;
//typedef double Real64;

// Inner
template<class T> class IScalar;
template<class T> class ILattice;

// Reality
template<class T> class RScalar;
template<class T> class RComplex;

// Primitives
template<class T> class PScalar;
template <class T, int N, template<class,int> class C> class PMatrix;
template <class T, int N, template<class,int> class C> class PVector;
template <class T, int N> class PColorVector;
template <class T, int N> class PSpinVector;
template <class T, int N> class PColorMatrix;
template <class T, int N> class PSpinMatrix;
template <class T> class PSeed;

template<int N> class GammaType;
template<int N, int m> class GammaConst;


// Outer
template<class T> class OScalar;
template<class T> class OLattice;

// Main type
template<class T, class C> class QDPType;

// Simple scalar trait class
template<class T> struct SimpleScalar;
template<class T> struct InternalScalar;
template<class T> struct WordType;

// Empty leaf functor tag
struct ElemLeaf;


QDP_END_NAMESPACE();

  
