// -*- C++ -*-
//
// $Id: primspinmat.h,v 1.2 2002-09-14 19:48:26 edwards Exp $
//
// QDP data parallel interface
//

QDP_BEGIN_NAMESPACE(QDP);

//-------------------------------------------------------------------------------------
//! Primitive spin Matrix class
/*! 
   * Spin matrix class support gamma matrix algebra 
   *
   * NOTE: the class is mostly empty - it is the specialized versions below
   * that know for a fixed size how gamma matrices (constants) should act
   * on the spin vectors.
   */
template <class T, int N> class PSpinMatrix : public PMatrix<T, N, PSpinMatrix>
{
public:
  //! PSpinMatrix = PScalar
  /*! Fill with primitive scalar */
  template<class T1>
  inline
  PSpinMatrix& operator=(const PScalar<T1>& rhs)
    {
      assign(rhs);
      return *this;
    }

  //! PSpinMatrix = PSpinMatrix
  /*! Set equal to another PSpinMatrix */
  inline
  PSpinMatrix& operator=(const PSpinMatrix& rhs) 
    {
      assign(rhs);
      return *this;
    }

};


#if 0
//! Specialization of primitive spin Matrix class for 4 spin components
/*! 
 * Spin matrix class support gamma matrix algebra for 4 spin components
 */
template <class T> class PSpinMatrix<T,4> : public PMatrix<T,4, PSpinMatrix>
{
public:
};
#endif


//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

// Underlying word type
template<class T1, int N>
struct WordType<PSpinMatrix<T1,N> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};

// Internally used scalars
template<class T, int N>
struct InternalScalar<PSpinMatrix<T,N> > {
  typedef PScalar<typename InternalScalar<T>::Type_t>  Type_t;
};


//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

// Default unary(PSpinMatrix) -> PSpinMatrix
template<class T1, int N, class Op>
struct UnaryReturn<PSpinMatrix<T1,N>, Op> {
  typedef PSpinMatrix<typename UnaryReturn<T1, Op>::Type_t, N>  Type_t;
};

// Default binary(PScalar,PSpinMatrix) -> PSpinMatrix
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PScalar<T1>, PSpinMatrix<T2,N>, Op> {
  typedef PSpinMatrix<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PSpinMatrix,PSpinMatrix) -> PSpinMatrix
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PSpinMatrix<T1,N>, PSpinMatrix<T2,N>, Op> {
  typedef PSpinMatrix<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PSpinMatrix,PScalar) -> PSpinMatrix
template<class T1, int N, class T2, class Op>
struct BinaryReturn<PSpinMatrix<T1,N>, PScalar<T2>, Op> {
  typedef PSpinMatrix<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};


#if 0
template<class T1, class T2>
struct UnaryReturn<PSpinMatrix<T2,N>, OpCast<T1> > {
  typedef PScalar<typename UnaryReturn<T, OpCast>::Type_t, N>  Type_t;
//  typedef T1 Type_t;
};
#endif


// Assignment is different
template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrix<T1,N>, PSpinMatrix<T2,N>, OpAssign > {
  typedef PSpinMatrix<T1,N> &Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrix<T1,N>, PSpinMatrix<T2,N>, OpAddAssign > {
  typedef PSpinMatrix<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrix<T1,N>, PSpinMatrix<T2,N>, OpSubtractAssign > {
  typedef PSpinMatrix<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrix<T1,N>, PSpinMatrix<T2,N>, OpMultiplyAssign > {
  typedef PSpinMatrix<T1,N> &Type_t;
};
 

template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrix<T1,N>, PScalar<T2>, OpAssign > {
  typedef PSpinMatrix<T1,N> &Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrix<T1,N>, PScalar<T2>, OpAddAssign > {
  typedef PSpinMatrix<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrix<T1,N>, PScalar<T2>, OpSubtractAssign > {
  typedef PSpinMatrix<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrix<T1,N>, PScalar<T2>, OpMultiplyAssign > {
  typedef PSpinMatrix<T1,N> &Type_t;
};
 


// SpinMatrix
template<class T, int N>
struct UnaryReturn<PSpinMatrix<T,N>, FnTrace > {
  typedef PScalar<typename UnaryReturn<T, FnTrace>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinMatrix<T,N>, FnTraceReal > {
  typedef PScalar<typename UnaryReturn<T, FnTraceReal>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinMatrix<T,N>, FnTraceImag > {
  typedef PScalar<typename UnaryReturn<T, FnTraceImag>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinMatrix<T,N>, FnNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinMatrix<T,N>, FnLocalNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrix<T1,N>, PSpinMatrix<T2,N>, FnInnerproduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerproduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrix<T1,N>, PSpinMatrix<T2,N>, FnLocalInnerproduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerproduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrix<T1,N>, PSpinMatrix<T2,N>, FnInnerproductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerproductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinMatrix<T1,N>, PSpinMatrix<T2,N>, FnLocalInnerproductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerproductReal>::Type_t>  Type_t;
};



// Gamma algebra
template<int m, class T2, int N, class OpGammaConstMultiply>
struct BinaryReturn<GammaConst<N,m>, PSpinMatrix<T2,N>, OpGammaConstMultiply> {
  typedef PSpinMatrix<typename UnaryReturn<T2, OpUnaryPlus>::Type_t, N>  Type_t;
};

template<class T2, int N, int m, class OpMultiplyGammaConst>
struct BinaryReturn<PSpinMatrix<T2,N>, GammaConst<N,m>, OpMultiplyGammaConst> {
  typedef PSpinMatrix<typename UnaryReturn<T2, OpUnaryPlus>::Type_t, N>  Type_t;
};

template<class T2, int N, class OpGammaTypeMultiply>
struct BinaryReturn<GammaType<N>, PSpinMatrix<T2,N>, OpGammaTypeMultiply> {
  typedef PSpinMatrix<typename UnaryReturn<T2, OpUnaryPlus>::Type_t, N>  Type_t;
};

template<class T2, int N, class OpMultiplyGammaType>
struct BinaryReturn<PSpinMatrix<T2,N>, GammaType<N>, OpMultiplyGammaType> {
  typedef PSpinMatrix<typename UnaryReturn<T2, OpUnaryPlus>::Type_t, N>  Type_t;
};




//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------

// SpinMatrix class primitive operations

// trace = spinTrace(source1)
/*! This only acts on spin indices and is diagonal in all other indices */
template<class T, int N>
struct UnaryReturn<PSpinMatrix<T,N>, FnSpinTrace > {
  typedef PScalar<typename UnaryReturn<T, FnSpinTrace>::Type_t>  Type_t;
};

template<class T, int N>
inline typename UnaryReturn<PSpinMatrix<T,N>, FnSpinTrace>::Type_t
spinTrace(const PSpinMatrix<T,N>& s1)
{
  typename UnaryReturn<PSpinMatrix<T,N>, FnSpinTrace>::Type_t  d;

  // Since the spin index is eaten, do not need to pass on function by
  // calling trace(...) again
  d.elem() = s1.elem(0,0);
  for(int i=1; i < N; ++i)
    d.elem() += s1.elem(i,i);

  return d;
}




#if 0
// SpinMatrix<N> = Gamma<N,m> * SpinMatrix<N>
// Default case 
template<class T2, int N, int m>
inline typename BinaryReturn<GammaConst<N,m>, PSpinMatrix<T2,N>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<N,m>&, const PSpinMatrix<T2,N>& r)
{
  // Not implemented
}
#endif



// SpinMatrix<4> = Gamma<4,m> * SpinMatrix<4>
// There are 16 cases here for Nd=4
template<class T2>
inline typename BinaryReturn<GammaConst<4,0>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,0>&, const PSpinMatrix<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,0>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t  d;
  
  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = r.elem(0,i);
    d.elem(1,i) = r.elem(1,i);
    d.elem(2,i) = r.elem(2,i);
    d.elem(3,i) = r.elem(3,i);
  }

  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,1>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,1>&, const PSpinMatrix<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,1>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t  d;
  
  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = multiplyI(r.elem(3,i));
    d.elem(1,i) = multiplyI(r.elem(2,i));
    d.elem(2,i) = multiplyMinusI(r.elem(1,i));
    d.elem(3,i) = multiplyMinusI(r.elem(0,i));
  }

  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,2>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,2>&, const PSpinMatrix<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,2>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = -r.elem(3,i);
    d.elem(1,i) = r.elem(2,i);
    d.elem(2,i) = r.elem(1,i);
    d.elem(3,i) = -r.elem(0,i);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,3>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,3>&, const PSpinMatrix<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,3>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = multiplyMinusI(r.elem(0,i));
    d.elem(1,i) = multiplyI(r.elem(1,i));
    d.elem(2,i) = multiplyMinusI(r.elem(2,i));
    d.elem(3,i) = multiplyI(r.elem(3,i));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,4>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,4>&, const PSpinMatrix<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,4>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = multiplyI(r.elem(2,i));
    d.elem(1,i) = multiplyMinusI(r.elem(3,i));
    d.elem(2,i) = multiplyMinusI(r.elem(0,i));
    d.elem(3,i) = multiplyI(r.elem(1,i));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,5>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,5>&, const PSpinMatrix<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,5>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = -r.elem(1,i);
    d.elem(1,i) = r.elem(0,i);
    d.elem(2,i) = -r.elem(3,i);
    d.elem(3,i) = r.elem(2,i);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,6>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,6>&, const PSpinMatrix<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,6>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = multiplyMinusI(r.elem(1,i));
    d.elem(1,i) = multiplyMinusI(r.elem(0,i));
    d.elem(2,i) = multiplyMinusI(r.elem(3,i));
    d.elem(3,i) = multiplyMinusI(r.elem(2,i));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,7>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,7>&, const PSpinMatrix<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,7>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = r.elem(2,i);
    d.elem(1,i) = r.elem(3,i);
    d.elem(2,i) = -r.elem(0,i);
    d.elem(3,i) = -r.elem(1,i);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,8>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,8>&, const PSpinMatrix<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,8>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = r.elem(2,i);
    d.elem(1,i) = r.elem(3,i);
    d.elem(2,i) = r.elem(0,i);
    d.elem(3,i) = r.elem(1,i);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,9>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,9>&, const PSpinMatrix<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,9>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = multiplyI(r.elem(1,i));
    d.elem(1,i) = multiplyI(r.elem(0,i));
    d.elem(2,i) = multiplyMinusI(r.elem(3,i));
    d.elem(3,i) = multiplyMinusI(r.elem(2,i));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,10>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,10>&, const PSpinMatrix<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,10>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = -r.elem(1,i);
    d.elem(1,i) = r.elem(0,i);
    d.elem(2,i) = r.elem(3,i);
    d.elem(3,i) = -r.elem(2,i);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,11>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,11>&, const PSpinMatrix<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,11>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = multiplyMinusI(r.elem(2,i));
    d.elem(1,i) = multiplyI(r.elem(3,i));
    d.elem(2,i) = multiplyMinusI(r.elem(0,i));
    d.elem(3,i) = multiplyI(r.elem(1,i));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,12>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,12>&, const PSpinMatrix<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,12>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = multiplyI(r.elem(0,i));
    d.elem(1,i) = multiplyMinusI(r.elem(1,i));
    d.elem(2,i) = multiplyMinusI(r.elem(2,i));
    d.elem(3,i) = multiplyI(r.elem(3,i));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,13>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,13>&, const PSpinMatrix<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,13>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = -r.elem(3,i);
    d.elem(1,i) = r.elem(2,i);
    d.elem(2,i) = -r.elem(1,i);
    d.elem(3,i) = r.elem(0,i);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,14>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,14>&, const PSpinMatrix<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,14>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = multiplyMinusI(r.elem(3,i));
    d.elem(1,i) = multiplyMinusI(r.elem(2,i));
    d.elem(2,i) = multiplyMinusI(r.elem(1,i));
    d.elem(3,i) = multiplyMinusI(r.elem(0,i));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,15>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,15>&, const PSpinMatrix<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,15>, PSpinMatrix<T2,4>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(0,i) = r.elem(0,i);
    d.elem(1,i) = r.elem(1,i);
    d.elem(2,i) = -r.elem(2,i);
    d.elem(3,i) = -r.elem(3,i);
  }
  
  return d;
}


// SpinMatrix<4> = SpinMatrix<4> * Gamma<4,m>
// There are 16 cases here for Nd=4
template<class T2>
inline typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,0>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrix<T2,4>& l, const GammaConst<4,0>&)
{
  typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,0>, OpGammaConstMultiply>::Type_t  d; 

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) =  l.elem(i,0);
    d.elem(i,1) =  l.elem(i,1);
    d.elem(i,2) =  l.elem(i,2);
    d.elem(i,3) =  l.elem(i,3);
  }
 
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,1>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrix<T2,4>& l, const GammaConst<4,1>&)
{
  typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,1>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = multiplyMinusI(l.elem(i,3));
    d.elem(i,1) = multiplyMinusI(l.elem(i,2));
    d.elem(i,2) = multiplyI(l.elem(i,1));
    d.elem(i,3) = multiplyI(l.elem(i,0));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,2>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrix<T2,4>& l, const GammaConst<4,2>&)
{
  typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,2>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = -l.elem(i,3);
    d.elem(i,1) =  l.elem(i,2);
    d.elem(i,2) =  l.elem(i,1);
    d.elem(i,3) = -l.elem(i,0);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,3>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrix<T2,4>& l, const GammaConst<4,3>&)
{
  typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,3>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = multiplyMinusI(l.elem(i,0));
    d.elem(i,1) = multiplyI(l.elem(i,1));
    d.elem(i,2) = multiplyMinusI(l.elem(i,2));
    d.elem(i,3) = multiplyI(l.elem(i,3));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,4>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrix<T2,4>& l, const GammaConst<4,4>&)
{
  typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,4>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = multiplyMinusI(l.elem(i,2));
    d.elem(i,1) = multiplyI(l.elem(i,3));
    d.elem(i,2) = multiplyI(l.elem(i,0));
    d.elem(i,3) = multiplyMinusI(l.elem(i,1));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,5>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrix<T2,4>& l, const GammaConst<4,5>&)
{
  typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,5>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) =  l.elem(i,1);
    d.elem(i,1) = -l.elem(i,0);
    d.elem(i,2) =  l.elem(i,3);
    d.elem(i,3) = -l.elem(i,2);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,6>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrix<T2,4>& l, const GammaConst<4,6>&)
{
  typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,6>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = multiplyMinusI(l.elem(i,1));
    d.elem(i,1) = multiplyMinusI(l.elem(i,0));
    d.elem(i,2) = multiplyMinusI(l.elem(i,3));
    d.elem(i,3) = multiplyMinusI(l.elem(i,2));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,7>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrix<T2,4>& l, const GammaConst<4,7>&)
{
  typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,7>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = -l.elem(i,2);
    d.elem(i,1) = -l.elem(i,3);
    d.elem(i,2) =  l.elem(i,0);
    d.elem(i,3) =  l.elem(i,1);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,8>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrix<T2,4>& l, const GammaConst<4,8>&)
{
  typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,8>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) =  l.elem(i,2);
    d.elem(i,1) =  l.elem(i,3);
    d.elem(i,2) =  l.elem(i,0);
    d.elem(i,3) =  l.elem(i,1);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,9>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrix<T2,4>& l, const GammaConst<4,9>&)
{
  typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,9>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = multiplyI(l.elem(i,1));
    d.elem(i,1) = multiplyI(l.elem(i,0));
    d.elem(i,2) = multiplyMinusI(l.elem(i,3));
    d.elem(i,3) = multiplyMinusI(l.elem(i,2));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,10>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrix<T2,4>& l, const GammaConst<4,10>&)
{
  typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,10>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) =  l.elem(i,1);
    d.elem(i,1) = -l.elem(i,0);
    d.elem(i,2) = -l.elem(i,3);
    d.elem(i,3) =  l.elem(i,2);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,11>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrix<T2,4>& l, const GammaConst<4,11>&)
{
  typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,11>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = multiplyMinusI(l.elem(i,2));
    d.elem(i,1) = multiplyI(l.elem(i,3));
    d.elem(i,2) = multiplyMinusI(l.elem(i,0));
    d.elem(i,3) = multiplyI(l.elem(i,1));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,12>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrix<T2,4>& l, const GammaConst<4,12>&)
{
  typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,12>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = multiplyI(l.elem(i,0));
    d.elem(i,1) = multiplyMinusI(l.elem(i,1));
    d.elem(i,2) = multiplyMinusI(l.elem(i,2));
    d.elem(i,3) = multiplyI(l.elem(i,3));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,13>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrix<T2,4>& l, const GammaConst<4,13>&)
{
  typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,13>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) =  l.elem(i,3);
    d.elem(i,1) = -l.elem(i,2);
    d.elem(i,2) =  l.elem(i,1);
    d.elem(i,3) = -l.elem(i,0);
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,14>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrix<T2,4>& l, const GammaConst<4,14>&)
{
  typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,14>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) = multiplyMinusI(l.elem(i,3));
    d.elem(i,1) = multiplyMinusI(l.elem(i,2));
    d.elem(i,2) = multiplyMinusI(l.elem(i,1));
    d.elem(i,3) = multiplyMinusI(l.elem(i,0));
  }
  
  return d;
}

template<class T2>
inline typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,15>, OpGammaConstMultiply>::Type_t
operator*(const PSpinMatrix<T2,4>& l, const GammaConst<4,15>&)
{
  typename BinaryReturn<PSpinMatrix<T2,4>, GammaConst<4,15>, OpGammaConstMultiply>::Type_t  d;

  for(int i=0; i < 4; ++i)
  {
    d.elem(i,0) =  l.elem(i,0);
    d.elem(i,1) =  l.elem(i,1);
    d.elem(i,2) = -l.elem(i,2);
    d.elem(i,3) = -l.elem(i,3);
  }
  
  return d;
}


//------------------------------------------
// quark propagator contraction
template<class T1, class T2>
inline typename BinaryReturn<PSpinMatrix<T1,4>, PSpinMatrix<T2,4>, FnQuarkContract13>::Type_t
quarkContract13(const PSpinMatrix<T1,4>& s1, const PSpinMatrix<T2,4>& s2)
{
  typename BinaryReturn<PSpinMatrix<T1,4>, PSpinMatrix<T2,4>, FnQuarkContract13>::Type_t  d;

  for(int j=0; j < 4; ++j)
    for(int i=0; i < 4; ++i)
    {
      d.elem(i,j) = quarkcontractxx(s1.elem(0,i), s2.elem(0,j));
      for(int k=1; k < 4; ++k)
	d.elem(i,j) += quarkcontractxx(s1.elem(k,i), s2.elem(k,j));
    }

  return d;
}

template<class T1, class T2>
inline typename BinaryReturn<PSpinMatrix<T1,4>, PSpinMatrix<T2,4>, FnQuarkContract14>::Type_t
quarkContract14(const PSpinMatrix<T1,4>& s1, const PSpinMatrix<T2,4>& s2)
{
  typename BinaryReturn<PSpinMatrix<T1,4>, PSpinMatrix<T2,4>, FnQuarkContract14>::Type_t  d;

  for(int j=0; j < 4; ++j)
    for(int i=0; i < 4; ++i)
    {
      d.elem(i,j) = quarkcontractxx(s1.elem(0,i), s2.elem(j,0));
      for(int k=1; k < 4; ++k)
	d.elem(i,j) += quarkcontractxx(s1.elem(k,i), s2.elem(j,k));
    }

  return d;
}

template<class T1, class T2>
inline typename BinaryReturn<PSpinMatrix<T1,4>, PSpinMatrix<T2,4>, FnQuarkContract23>::Type_t
quarkContract23(const PSpinMatrix<T1,4>& s1, const PSpinMatrix<T2,4>& s2)
{
  typename BinaryReturn<PSpinMatrix<T1,4>, PSpinMatrix<T2,4>, FnQuarkContract23>::Type_t  d;

  for(int j=0; j < 4; ++j)
    for(int i=0; i < 4; ++i)
    {
      d.elem(i,j) = quarkcontractxx(s1.elem(i,0), s2.elem(0,j));
      for(int k=1; k < 4; ++k)
	d.elem(i,j) += quarkcontractxx(s1.elem(i,k), s2.elem(k,j));
    }

  return d;
}

template<class T1, class T2>
inline typename BinaryReturn<PSpinMatrix<T1,4>, PSpinMatrix<T2,4>, FnQuarkContract24>::Type_t
quarkContract24(const PSpinMatrix<T1,4>& s1, const PSpinMatrix<T2,4>& s2)
{
  typename BinaryReturn<PSpinMatrix<T1,4>, PSpinMatrix<T2,4>, FnQuarkContract24>::Type_t  d;

  for(int j=0; j < 4; ++j)
    for(int i=0; i < 4; ++i)
    {
      d.elem(i,j) = quarkcontractxx(s1.elem(i,0), s2.elem(j,0));
      for(int k=1; k < 4; ++k)
	d.elem(i,j) += quarkcontractxx(s1.elem(i,k), s2.elem(j,k));
    }

  return d;
}

template<class T1, class T2>
inline typename BinaryReturn<PSpinMatrix<T1,4>, PSpinMatrix<T2,4>, FnQuarkContract12>::Type_t
quarkContract12(const PSpinMatrix<T1,4>& s1, const PSpinMatrix<T2,4>& s2)
{
  typename BinaryReturn<PSpinMatrix<T1,4>, PSpinMatrix<T2,4>, FnQuarkContract12>::Type_t  d;

  for(int j=0; j < 4; ++j)
    for(int i=0; i < 4; ++i)
    {
      d.elem(i,j) = quarkcontractxx(s1.elem(0,0), s2.elem(i,j));
      for(int k=1; k < 4; ++k)
	d.elem(i,j) += quarkcontractxx(s1.elem(k,k), s2.elem(i,j));
    }

  return d;
}

template<class T1, class T2>
inline typename BinaryReturn<PSpinMatrix<T1,4>, PSpinMatrix<T2,4>, FnQuarkContract34>::Type_t
quarkContract34(const PSpinMatrix<T1,4>& s1, const PSpinMatrix<T2,4>& s2)
{
  typename BinaryReturn<PSpinMatrix<T1,4>, PSpinMatrix<T2,4>, FnQuarkContract34>::Type_t  d;

  for(int j=0; j < 4; ++j)
    for(int i=0; i < 4; ++i)
    {
      d.elem(i,j) = quarkcontractxx(s1.elem(i,j), s2.elem(0,0));
      for(int k=1; k < 4; ++k)
	d.elem(i,j) += quarkcontractxx(s1.elem(i,j), s2.elem(k,k));
    }

  return d;
}



QDP_END_NAMESPACE();

