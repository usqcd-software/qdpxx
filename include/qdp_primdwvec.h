// -*- C++ -*-
// $Id: qdp_primdwvec.h,v 1.2 2003-10-20 20:14:57 edwards Exp $

/*! \file
 * \brief Domain-wall Vector (lives in fictitious flavor space)
 */


QDP_BEGIN_NAMESPACE(QDP);

//-------------------------------------------------------------------------------------
/*! \addtogroup primdwvector Domain-wall vector primitive
 * \ingroup primvector
 *
 * Primitive type that transforms like a Domain-Wall flavor vector
 *
 * @{
 */

//! Primitive domain-wall vector class
/*! 
 * Supports domain-wall manipulations
 */
template <class T, int N> class PDWVector : public PVector<T, N, PDWVector>
{
public:
  //! PVector = PVector
  /*! Set equal to another PVector */
  inline
  PDWVector& operator=(const PDWVector& rhs) 
    {
      assign(rhs);
      return *this;
    }

};

/*! @} */  // end of group primdwvector

//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

// Underlying word type
template<class T1, int N>
struct WordType<PDWVector<T1,N> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};

// Internally used scalars
template<class T, int N>
struct InternalScalar<PDWVector<T,N> > {
  typedef PScalar<typename InternalScalar<T>::Type_t>  Type_t;
};

// Makes a primitive scalar leaving other indices along
template<class T, int N>
struct PrimitiveScalar<PDWVector<T,N> > {
  typedef PScalar<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

// Makes a lattice scalar leaving primitive indices alone
template<class T, int N>
struct LatticeScalar<PDWVector<T,N> > {
  typedef PDWVector<typename LatticeScalar<T>::Type_t, N>  Type_t;
};

//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

// Default unary(PDWVector) -> PDWVector
template<class T1, int N, class Op>
struct UnaryReturn<PDWVector<T1,N>, Op> {
  typedef PDWVector<typename UnaryReturn<T1, Op>::Type_t, N>  Type_t;
};
// Default binary(PScalar,PDWVector) -> PDWVector
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PScalar<T1>, PDWVector<T2,N>, Op> {
  typedef PDWVector<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PDWVector,PScalar) -> PDWVector
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PDWVector<T1,N>, PScalar<T2>, Op> {
  typedef PDWVector<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PDWVector,PDWVector) -> PDWVector
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PDWVector<T1,N>, PDWVector<T2,N>, Op> {
  typedef PDWVector<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};


#if 0
template<class T1, class T2>
struct UnaryReturn<PScalar<T2>, OpCast<T1> > {
  typedef PScalar<typename UnaryReturn<T, OpCast>::Type_t>  Type_t;
//  typedef T1 Type_t;
};
#endif


// Assignment is different
template<class T1, class T2, int N>
struct BinaryReturn<PDWVector<T1,N>, PDWVector<T2,N>, OpAssign > {
  typedef PDWVector<T1,N> &Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PDWVector<T1,N>, PDWVector<T2,N>, OpAddAssign > {
  typedef PDWVector<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PDWVector<T1,N>, PDWVector<T2,N>, OpSubtractAssign > {
  typedef PDWVector<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PDWVector<T1,N>, PScalar<T2>, OpMultiplyAssign > {
  typedef PDWVector<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PDWVector<T1,N>, PScalar<T2>, OpDivideAssign > {
  typedef PDWVector<T1,N> &Type_t;
};
 

// DWVector
template<class T, int N>
struct UnaryReturn<PDWVector<T,N>, FnNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PDWVector<T,N>, FnLocalNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PDWVector<T1,N>, PDWVector<T2,N>, FnInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PDWVector<T1,N>, PDWVector<T2,N>, FnLocalInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PDWVector<T1,N>, PDWVector<T2,N>, FnInnerProductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PDWVector<T1,N>, PDWVector<T2,N>, FnLocalInnerProductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};




//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------

// Peeking and poking
//! Extract DW vector components 
template<class T, int N>
struct UnaryReturn<PDWVector<T,N>, FnPeekDWVector > {
  typedef PScalar<typename UnaryReturn<T, FnPeekDWVector>::Type_t>  Type_t;
};

template<class T, int N>
inline typename UnaryReturn<PDWVector<T,N>, FnPeekDWVector>::Type_t
peekDW(const PDWVector<T,N>& l, int row)
{
  typename UnaryReturn<PDWVector<T,N>, FnPeekDWVector>::Type_t  d;

  // Note, do not need to propagate down since the function is eaten at this level
  d.elem() = l.elem(row);
  return d;
}

//! Insert color vector components
template<class T1, class T2, int N>
inline PDWVector<T1,N>&
pokeDW(PDWVector<T1,N>& l, const PScalar<T2>& r, int row)
{
  // Note, do not need to propagate down since the function is eaten at this level
  l.elem(row) = r.elem();
  return l;
}


//-----------------------------------------------------------------------------
//! PDWVector = Gamma<M,m> * PDWVector
template<class T2, int N, int M, int m>
inline typename BinaryReturn<GammaConst<M,m>, PDWVector<T2,N>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<M,m>& l, const PDWVector<T2,N>& r)
{
  typename BinaryReturn<GammaConst<M,m>, PDWVector<T2,N>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = l * r.elem(i);

  return d;
}

//! PDWVector = PDWVector * Gamma<M,m>
template<class T2, int N, int M, int m>
inline typename BinaryReturn<PDWVector<T2,N>, GammaConst<M,m>, OpGammaConstMultiply>::Type_t
operator*(const PDWVector<T2,N>& l, const GammaConst<M,m>& r)
{
  typename BinaryReturn<GammaConst<M,m>, PDWVector<T2,N>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem() * r;

  return d;
}

//-----------------------------------------------------------------------------
//! PDWVector = SpinProject(PDWVector)
template<class T, int N>
inline typename UnaryReturn<PDWVector<T,N>, FnSpinProjectDir0Minus>::Type_t
spinProjectDir0Minus(const PDWVector<T,N>& s1)
{
  typename UnaryReturn<PDWVector<T,N>, FnSpinProjectDir0Minus>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = spinProjectDir0Minus(s1.elem(i));

  return d;
}

//! PDWVector = SpinReconstruct(PDWVector)
template<class T, int N>
inline typename UnaryReturn<PDWVector<T,N>, FnSpinReconstructDir0Minus>::Type_t
spinReconstructDir0Minus(const PDWVector<T,N>& s1)
{
  typename UnaryReturn<PDWVector<T,N>, FnSpinReconstructDir0Minus>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = spinReconstructDir0Minus(s1.elem(i));

  return d;
}


//! PDWVector = SpinProject(PDWVector)
template<class T, int N>
inline typename UnaryReturn<PDWVector<T,N>, FnSpinProjectDir1Minus>::Type_t
spinProjectDir1Minus(const PDWVector<T,N>& s1)
{
  typename UnaryReturn<PDWVector<T,N>, FnSpinProjectDir1Minus>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = spinProjectDir1Minus(s1.elem(i));

  return d;
}

//! PDWVector = SpinReconstruct(PDWVector)
template<class T, int N>
inline typename UnaryReturn<PDWVector<T,N>, FnSpinReconstructDir1Minus>::Type_t
spinReconstructDir1Minus(const PDWVector<T,N>& s1)
{
  typename UnaryReturn<PDWVector<T,N>, FnSpinReconstructDir1Minus>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = spinReconstructDir1Minus(s1.elem(i));

  return d;
}


//! PDWVector = SpinProject(PDWVector)
template<class T, int N>
inline typename UnaryReturn<PDWVector<T,N>, FnSpinProjectDir2Minus>::Type_t
spinProjectDir2Minus(const PDWVector<T,N>& s1)
{
  typename UnaryReturn<PDWVector<T,N>, FnSpinProjectDir2Minus>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = spinProjectDir2Minus(s1.elem(i));

  return d;
}

//! PDWVector = SpinReconstruct(PDWVector)
template<class T, int N>
inline typename UnaryReturn<PDWVector<T,N>, FnSpinReconstructDir2Minus>::Type_t
spinReconstructDir2Minus(const PDWVector<T,N>& s1)
{
  typename UnaryReturn<PDWVector<T,N>, FnSpinReconstructDir2Minus>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = spinReconstructDir2Minus(s1.elem(i));

  return d;
}


//! PDWVector = SpinProject(PDWVector)
template<class T, int N>
inline typename UnaryReturn<PDWVector<T,N>, FnSpinProjectDir3Minus>::Type_t
spinProjectDir3Minus(const PDWVector<T,N>& s1)
{
  typename UnaryReturn<PDWVector<T,N>, FnSpinProjectDir3Minus>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = spinProjectDir3Minus(s1.elem(i));

  return d;
}

//! PDWVector = SpinReconstruct(PDWVector)
template<class T, int N>
inline typename UnaryReturn<PDWVector<T,N>, FnSpinReconstructDir3Minus>::Type_t
spinReconstructDir3Minus(const PDWVector<T,N>& s1)
{
  typename UnaryReturn<PDWVector<T,N>, FnSpinReconstructDir3Minus>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = spinReconstructDir3Minus(s1.elem(i));

  return d;
}


//! PDWVector = SpinProject(PDWVector)
template<class T, int N>
inline typename UnaryReturn<PDWVector<T,N>, FnSpinProjectDir0Plus>::Type_t
spinProjectDir0Plus(const PDWVector<T,N>& s1)
{
  typename UnaryReturn<PDWVector<T,N>, FnSpinProjectDir0Plus>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = spinProjectDir0Plus(s1.elem(i));

  return d;
}

//! PDWVector = SpinReconstruct(PDWVector)
template<class T, int N>
inline typename UnaryReturn<PDWVector<T,N>, FnSpinReconstructDir0Plus>::Type_t
spinReconstructDir0Plus(const PDWVector<T,N>& s1)
{
  typename UnaryReturn<PDWVector<T,N>, FnSpinReconstructDir0Plus>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = spinReconstructDir0Plus(s1.elem(i));

  return d;
}


//! PDWVector = SpinProject(PDWVector)
template<class T, int N>
inline typename UnaryReturn<PDWVector<T,N>, FnSpinProjectDir1Plus>::Type_t
spinProjectDir1Plus(const PDWVector<T,N>& s1)
{
  typename UnaryReturn<PDWVector<T,N>, FnSpinProjectDir1Plus>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = spinProjectDir1Plus(s1.elem(i));

  return d;
}

//! PDWVector = SpinReconstruct(PDWVector)
template<class T, int N>
inline typename UnaryReturn<PDWVector<T,N>, FnSpinReconstructDir1Plus>::Type_t
spinReconstructDir1Plus(const PDWVector<T,N>& s1)
{
  typename UnaryReturn<PDWVector<T,N>, FnSpinReconstructDir1Plus>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = spinReconstructDir1Plus(s1.elem(i));

  return d;
}


//! PDWVector = SpinProject(PDWVector)
template<class T, int N>
inline typename UnaryReturn<PDWVector<T,N>, FnSpinProjectDir2Plus>::Type_t
spinProjectDir2Plus(const PDWVector<T,N>& s1)
{
  typename UnaryReturn<PDWVector<T,N>, FnSpinProjectDir2Plus>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = spinProjectDir2Plus(s1.elem(i));

  return d;
}

//! PDWVector = SpinReconstruct(PDWVector)
template<class T, int N>
inline typename UnaryReturn<PDWVector<T,N>, FnSpinReconstructDir2Plus>::Type_t
spinReconstructDir2Plus(const PDWVector<T,N>& s1)
{
  typename UnaryReturn<PDWVector<T,N>, FnSpinReconstructDir2Plus>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = spinReconstructDir2Plus(s1.elem(i));

  return d;
}


//! PDWVector = SpinProject(PDWVector)
template<class T, int N>
inline typename UnaryReturn<PDWVector<T,N>, FnSpinProjectDir3Plus>::Type_t
spinProjectDir3Plus(const PDWVector<T,N>& s1)
{
  typename UnaryReturn<PDWVector<T,N>, FnSpinProjectDir3Plus>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = spinProjectDir3Plus(s1.elem(i));

  return d;
}

//! PDWVector = SpinReconstruct(PDWVector)
template<class T, int N>
inline typename UnaryReturn<PDWVector<T,N>, FnSpinReconstructDir3Plus>::Type_t
spinReconstructDir3Plus(const PDWVector<T,N>& s1)
{
  typename UnaryReturn<PDWVector<T,N>, FnSpinReconstructDir3Plus>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = spinReconstructDir3Plus(s1.elem(i));

  return d;
}


QDP_END_NAMESPACE();

