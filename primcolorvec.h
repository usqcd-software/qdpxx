// -*- C++ -*-
//
// $Id: primcolorvec.h,v 1.2 2002-09-14 19:48:26 edwards Exp $
//
// QDP data parallel interface
//

QDP_BEGIN_NAMESPACE(QDP);

//-------------------------------------------------------------------------------------
//! Primitive color Vector class
template <class T, int N> class PColorVector : public PVector<T, N, PColorVector>
{
public:
  //! PColorVector = PColorVector
  /*! Set equal to another PColorVector */
  inline
  PColorVector& operator=(const PColorVector& rhs) 
    {
      assign(rhs);
      return *this;
    }

};


//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

// Underlying word type
template<class T1, int N>
struct WordType<PColorVector<T1,N> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};

// Internally used scalars
template<class T, int N>
struct InternalScalar<PColorVector<T,N> > {
  typedef PScalar<typename InternalScalar<T>::Type_t>  Type_t;
};

//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

// Default unary(PColorVector) -> PColorVector
template<class T1, int N, class Op>
struct UnaryReturn<PColorVector<T1,N>, Op> {
  typedef PColorVector<typename UnaryReturn<T1, Op>::Type_t, N>  Type_t;
};
// Default binary(PScalar,PColorVector) -> PColorVector
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PScalar<T1>, PColorVector<T2,N>, Op> {
  typedef PColorVector<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PColorMatrix,PColorVector) -> PColorVector
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PColorMatrix<T1,N>, PColorVector<T2,N>, Op> {
  typedef PColorVector<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PColorVector,PScalar) -> PColorVector
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PColorVector<T1,N>, PScalar<T2>, Op> {
  typedef PColorVector<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PColorVector,PColorVector) -> PColorVector
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PColorVector<T1,N>, PColorVector<T2,N>, Op> {
  typedef PColorVector<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
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
struct BinaryReturn<PColorVector<T1,N>, PColorVector<T2,N>, OpAssign > {
  typedef PColorVector<T1,N> &Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorVector<T1,N>, PColorVector<T2,N>, OpAddAssign > {
  typedef PColorVector<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PColorVector<T1,N>, PColorVector<T2,N>, OpSubtractAssign > {
  typedef PColorVector<T1,N> &Type_t;
};
 

// ColorVector
template<class T, int N>
struct UnaryReturn<PColorVector<T,N>, FnNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PColorVector<T,N>, FnLocalNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorVector<T1,N>, PColorVector<T2,N>, FnInnerproduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerproduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorVector<T1,N>, PColorVector<T2,N>, FnLocalInnerproduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerproduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorVector<T1,N>, PColorVector<T2,N>, FnInnerproductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerproductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorVector<T1,N>, PColorVector<T2,N>, FnLocalInnerproductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerproductReal>::Type_t>  Type_t;
};




//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------


QDP_END_NAMESPACE();

