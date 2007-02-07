// -*- C++ -*-
// $Id: qdp_primspinvec.h,v 1.7 2007-02-07 20:45:45 bjoo Exp $

/*! \file
 * \brief Primitive Spin Vector
 */


QDP_BEGIN_NAMESPACE(QDP);

//-------------------------------------------------------------------------------------
/*! \addtogroup primspinvector Spin vector primitive
 * \ingroup primvector
 *
 * Primitive type that transforms like a Spin vector
 *
 * @{
 */

//! Primitive spin Vector class
/*! 
 * Spin vector class supports gamma matrix algebra 
 *
 * NOTE: the class is mostly empty - it is the specialized versions below
 * that know for a fixed size how gamma matrices (constants) should act
 * on the spin vectors.
 */
template <class T, int N> class PSpinVector : public PVector<T, N, PSpinVector>
{
public:
  //! PVector = PVector
  /*! Set equal to another PVector */
  template<class T1>
  inline
  PSpinVector& operator=(const PSpinVector<T1,N>& rhs) 
    {
      assign(rhs);
      return *this;
    }

};


//! Specialization of primitive spin Vector class for 4 spin components
/*! 
 * Spin vector class supports gamma matrix algebra for 4 spin components
 */
template<class T> class PSpinVector<T,4> : public PVector<T, 4, PSpinVector>
{
};


//! Specialization of primitive spin Vector class for 2 spin components
/*! 
 * Spin vector class supports gamma matrix algebra for 2 spin components
 * NOTE: this can be used for spin projection tricks of a 4 component spinor
 * to 2 spin components, or a 2 spin component Dirac fermion in 2 dimensions
 */
template<class T> class PSpinVector<T,2> : public PVector<T, 2, PSpinVector>
{
public:
};


/*! @} */   // end of group primspinvec

//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

// Underlying word type
template<class T1, int N>
struct WordType<PSpinVector<T1,N> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};

// Fixed Precision
template<class T1, int N>
struct SinglePrecType< PSpinVector<T1, N> > 
{
  typedef PSpinVector< typename SinglePrecType<T1>::Type_t, N> Type_t;
};

template<class T1, int N>
struct DoublePrecType< PSpinVector<T1, N> > 
{
  typedef PSpinVector< typename DoublePrecType<T1>::Type_t, N> Type_t;
};

// Internally used scalars
template<class T, int N>
struct InternalScalar<PSpinVector<T,N> > {
  typedef PScalar<typename InternalScalar<T>::Type_t>  Type_t;
};

// Makes a primitive into a scalar leaving grid alone
template<class T, int N>
struct PrimitiveScalar<PSpinVector<T,N> > {
  typedef PScalar<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

// Makes a lattice scalar leaving primitive indices alone
template<class T, int N>
struct LatticeScalar<PSpinVector<T,N> > {
  typedef PSpinVector<typename LatticeScalar<T>::Type_t, N>  Type_t;
};

//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

// Default unary(PSpinVector) -> PSpinVector
template<class T1, int N, class Op>
struct UnaryReturn<PSpinVector<T1,N>, Op> {
  typedef PSpinVector<typename UnaryReturn<T1, Op>::Type_t, N>  Type_t;
};
// Default binary(PScalar,PSpinVector) -> PSpinVector
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PScalar<T1>, PSpinVector<T2,N>, Op> {
  typedef PSpinVector<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PSpinMatrix,PSpinVector) -> PSpinVector
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PSpinMatrix<T1,N>, PSpinVector<T2,N>, Op> {
  typedef PSpinVector<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PSpinVector,PScalar) -> PSpinVector
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PSpinVector<T1,N>, PScalar<T2>, Op> {
  typedef PSpinVector<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PSpinVector,PSpinVector) -> PSpinVector
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PSpinVector<T1,N>, PSpinVector<T2,N>, Op> {
  typedef PSpinVector<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
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
struct BinaryReturn<PSpinVector<T1,N>, PSpinVector<T2,N>, OpAssign > {
  typedef PSpinVector<T1,N> &Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinVector<T1,N>, PSpinVector<T2,N>, OpAddAssign > {
  typedef PSpinVector<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PSpinVector<T1,N>, PSpinVector<T2,N>, OpSubtractAssign > {
  typedef PSpinVector<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PSpinVector<T1,N>, PScalar<T2>, OpMultiplyAssign > {
  typedef PSpinVector<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PSpinVector<T1,N>, PScalar<T2>, OpDivideAssign > {
  typedef PSpinVector<T1,N> &Type_t;
};
 


// SpinVector
template<class T, int N>
struct UnaryReturn<PSpinVector<T,N>, FnNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVector<T,N>, FnLocalNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinVector<T1,N>, PSpinVector<T2,N>, FnInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinVector<T1,N>, PSpinVector<T2,N>, FnLocalInnerProduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinVector<T1,N>, PSpinVector<T2,N>, FnInnerProductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PSpinVector<T1,N>, PSpinVector<T2,N>, FnLocalInnerProductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};



// Gamma algebra
template<int m, class T2, int N>
struct BinaryReturn<GammaConst<N,m>, PSpinVector<T2,N>, OpGammaConstMultiply> {
  typedef PSpinVector<typename UnaryReturn<T2, OpUnaryPlus>::Type_t, N>  Type_t;
};

template<class T2, int N>
struct BinaryReturn<GammaType<N>, PSpinVector<T2,N>, OpGammaTypeMultiply> {
  typedef PSpinVector<typename UnaryReturn<T2, OpUnaryPlus>::Type_t, N>  Type_t;
};

// Generic Spin projection
template<class T, int N>
struct UnaryReturn<PSpinVector<T,N>, FnSpinProject > {
  typedef PSpinVector<typename UnaryReturn<T, FnSpinProject>::Type_t, (N>>1) >  Type_t;
};

// spin projection for each direction
template<class T, int N>
struct UnaryReturn<PSpinVector<T,N>, FnSpinProjectDir0Plus > {
  typedef PSpinVector<typename UnaryReturn<T, FnSpinProjectDir0Plus>::Type_t, (N>>1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVector<T,N>, FnSpinProjectDir1Plus > {
  typedef PSpinVector<typename UnaryReturn<T, FnSpinProjectDir1Plus>::Type_t, (N>>1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVector<T,N>, FnSpinProjectDir2Plus > {
  typedef PSpinVector<typename UnaryReturn<T, FnSpinProjectDir2Plus>::Type_t, (N>>1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVector<T,N>, FnSpinProjectDir3Plus > {
  typedef PSpinVector<typename UnaryReturn<T, FnSpinProjectDir3Plus>::Type_t, (N>>1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVector<T,N>, FnSpinProjectDir0Minus > {
  typedef PSpinVector<typename UnaryReturn<T, FnSpinProjectDir0Minus>::Type_t, (N>>1) > Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVector<T,N>, FnSpinProjectDir1Minus > {
  typedef PSpinVector<typename UnaryReturn<T, FnSpinProjectDir1Minus>::Type_t, (N>>1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVector<T,N>, FnSpinProjectDir2Minus > {
  typedef PSpinVector<typename UnaryReturn<T, FnSpinProjectDir2Minus>::Type_t, (N>>1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVector<T,N>, FnSpinProjectDir3Minus > {
  typedef PSpinVector<typename UnaryReturn<T, FnSpinProjectDir3Minus>::Type_t, (N>>1) >  Type_t;
};


// Generic Spin reconstruction
template<class T, int N>
struct UnaryReturn<PSpinVector<T,N>, FnSpinReconstruct > {
  typedef PSpinVector<typename UnaryReturn<T, FnSpinReconstruct>::Type_t, (N<<1) >  Type_t;
};

// spin reconstruction for each direction
template<class T, int N>
struct UnaryReturn<PSpinVector<T,N>, FnSpinReconstructDir0Plus > {
  typedef PSpinVector<typename UnaryReturn<T, FnSpinReconstructDir0Plus>::Type_t, (N<<1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVector<T,N>, FnSpinReconstructDir1Plus > {
  typedef PSpinVector<typename UnaryReturn<T, FnSpinReconstructDir1Plus>::Type_t, (N<<1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVector<T,N>, FnSpinReconstructDir2Plus > {
  typedef PSpinVector<typename UnaryReturn<T, FnSpinReconstructDir2Plus>::Type_t, (N<<1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVector<T,N>, FnSpinReconstructDir3Plus > {
  typedef PSpinVector<typename UnaryReturn<T, FnSpinReconstructDir3Plus>::Type_t, (N<<1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVector<T,N>, FnSpinReconstructDir0Minus > {
  typedef PSpinVector<typename UnaryReturn<T, FnSpinReconstructDir0Minus>::Type_t, (N<<1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVector<T,N>, FnSpinReconstructDir1Minus > {
  typedef PSpinVector<typename UnaryReturn<T, FnSpinReconstructDir1Minus>::Type_t, (N<<1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVector<T,N>, FnSpinReconstructDir2Minus > {
  typedef PSpinVector<typename UnaryReturn<T, FnSpinReconstructDir2Minus>::Type_t, (N<<1) >  Type_t;
};

template<class T, int N>
struct UnaryReturn<PSpinVector<T,N>, FnSpinReconstructDir3Minus > {
  typedef PSpinVector<typename UnaryReturn<T, FnSpinReconstructDir3Minus>::Type_t, (N<<1) >  Type_t;
};




//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------

/*! \addtogroup primspinvector */
/*! @{ */

// Peeking and poking
//! Extract spin vector components 
template<class T, int N>
struct UnaryReturn<PSpinVector<T,N>, FnPeekSpinVector > {
  typedef PScalar<typename UnaryReturn<T, FnPeekSpinVector>::Type_t>  Type_t;
};

template<class T, int N>
inline typename UnaryReturn<PSpinVector<T,N>, FnPeekSpinVector>::Type_t
peekSpin(const PSpinVector<T,N>& l, int row)
{
  typename UnaryReturn<PSpinVector<T,N>, FnPeekSpinVector>::Type_t  d;

  // Note, do not need to propagate down since the function is eaten at this level
  d.elem() = l.elem(row);
  return d;
}

//! Insert spin vector components
template<class T1, class T2, int N>
inline PSpinVector<T1,N>&
pokeSpin(PSpinVector<T1,N>& l, const PScalar<T2>& r, int row)
{
  // Note, do not need to propagate down since the function is eaten at this level
  l.elem(row) = r.elem();
  return l;
}


//-----------------------------------------------
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



// SpinVector<4> = Gamma<4,m> * SpinVector<4>
// There are 16 cases here for Nd=4
template<class T2>
inline typename BinaryReturn<GammaConst<4,0>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,0>&, const PSpinVector<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,0>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t  d;
  
  d.elem(0) =  r.elem(0);
  d.elem(1) =  r.elem(1);
  d.elem(2) =  r.elem(2);
  d.elem(3) =  r.elem(3);

  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,1>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,1>&, const PSpinVector<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,1>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t  d;
  
  d.elem(0) = timesI(r.elem(3));
  d.elem(1) = timesI(r.elem(2));
  d.elem(2) = timesMinusI(r.elem(1));
  d.elem(3) = timesMinusI(r.elem(0));

  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,2>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,2>&, const PSpinVector<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,2>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) = -r.elem(3);
  d.elem(1) =  r.elem(2);
  d.elem(2) =  r.elem(1);
  d.elem(3) = -r.elem(0);
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,3>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,3>&, const PSpinVector<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,3>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) = timesMinusI(r.elem(0));
  d.elem(1) = timesI(r.elem(1));
  d.elem(2) = timesMinusI(r.elem(2));
  d.elem(3) = timesI(r.elem(3));
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,4>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,4>&, const PSpinVector<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,4>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) = timesI(r.elem(2));
  d.elem(1) = timesMinusI(r.elem(3));
  d.elem(2) = timesMinusI(r.elem(0));
  d.elem(3) = timesI(r.elem(1));
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,5>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,5>&, const PSpinVector<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,5>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) = -r.elem(1);
  d.elem(1) =  r.elem(0);
  d.elem(2) = -r.elem(3);
  d.elem(3) =  r.elem(2);
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,6>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,6>&, const PSpinVector<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,6>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) = timesMinusI(r.elem(1));
  d.elem(1) = timesMinusI(r.elem(0));
  d.elem(2) = timesMinusI(r.elem(3));
  d.elem(3) = timesMinusI(r.elem(2));
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,7>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,7>&, const PSpinVector<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,7>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) =  r.elem(2);
  d.elem(1) =  r.elem(3);
  d.elem(2) = -r.elem(0);
  d.elem(3) = -r.elem(1);
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,8>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,8>&, const PSpinVector<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,8>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) =  r.elem(2);
  d.elem(1) =  r.elem(3);
  d.elem(2) =  r.elem(0);
  d.elem(3) =  r.elem(1);
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,9>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,9>&, const PSpinVector<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,9>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) = timesI(r.elem(1));
  d.elem(1) = timesI(r.elem(0));
  d.elem(2) = timesMinusI(r.elem(3));
  d.elem(3) = timesMinusI(r.elem(2));
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,10>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,10>&, const PSpinVector<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,10>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) = -r.elem(1);
  d.elem(1) =  r.elem(0);
  d.elem(2) =  r.elem(3);
  d.elem(3) = -r.elem(2);
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,11>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,11>&, const PSpinVector<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,11>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) = timesMinusI(r.elem(2));
  d.elem(1) = timesI(r.elem(3));
  d.elem(2) = timesMinusI(r.elem(0));
  d.elem(3) = timesI(r.elem(1));
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,12>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,12>&, const PSpinVector<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,12>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) = timesI(r.elem(0));
  d.elem(1) = timesMinusI(r.elem(1));
  d.elem(2) = timesMinusI(r.elem(2));
  d.elem(3) = timesI(r.elem(3));
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,13>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,13>&, const PSpinVector<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,13>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) = -r.elem(3);
  d.elem(1) =  r.elem(2);
  d.elem(2) = -r.elem(1);
  d.elem(3) =  r.elem(0);
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,14>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,14>&, const PSpinVector<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,14>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) = timesMinusI(r.elem(3));
  d.elem(1) = timesMinusI(r.elem(2));
  d.elem(2) = timesMinusI(r.elem(1));
  d.elem(3) = timesMinusI(r.elem(0));
  
  return d;
}

template<class T2>
inline typename BinaryReturn<GammaConst<4,15>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t
operator*(const GammaConst<4,15>&, const PSpinVector<T2,4>& r)
{
  typename BinaryReturn<GammaConst<4,15>, PSpinVector<T2,4>, OpGammaConstMultiply>::Type_t  d;

  d.elem(0) =  r.elem(0);
  d.elem(1) =  r.elem(1);
  d.elem(2) = -r.elem(2);
  d.elem(3) = -r.elem(3);
  
  return d;
}


// SpinVector<2> = SpinProject(SpinVector<4>)
// There are 4 cases here for Nd=4 for each forward/backward direction
template<class T>
inline typename UnaryReturn<PSpinVector<T,4>, FnSpinProjectDir0Minus>::Type_t
spinProjectDir0Minus(const PSpinVector<T,4>& s1)
{
  typename UnaryReturn<PSpinVector<T,4>, FnSpinProjectDir0Minus>::Type_t  d;

  /*                              ( 1  0  0 -i)  ( a0 )    ( a0 - i a3 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1 -i  0)  ( a1 )  = ( a1 - i a2 )
   *                    0         ( 0  i  1  0)  ( a2 )    ( a2 + i a1 )
   *                              ( i  0  0  1)  ( a3 )    ( a3 + i a0 )

   * Therefore the top components are

   *      ( b0r + i b0i )  =  ( {a0r + a3i} + i{a0i - a3r} )
   *      ( b1r + i b1i )     ( {a1r + a2i} + i{a1i - a2r} )

   * The bottom components of be may be reconstructed using the formula

   *      ( b2r + i b2i )  =  ( {a2r - a1i} + i{a2i + a1r} )  =  ( - b1i + i b1r )
   *      ( b3r + i b3i )     ( {a3r - a0i} + i{a3i + a0r} )     ( - b0i + i b0r ) 
   */
  d.elem(0) = s1.elem(0) - timesI(s1.elem(3));
  d.elem(1) = s1.elem(1) - timesI(s1.elem(2));

  return d;
}

template<class T>
inline typename UnaryReturn<PSpinVector<T,4>, FnSpinProjectDir1Minus>::Type_t
spinProjectDir1Minus(const PSpinVector<T,4>& s1)
{
  typename UnaryReturn<PSpinVector<T,4>, FnSpinProjectDir1Minus>::Type_t  d;

  /*                              ( 1  0  0  1)  ( a0 )    ( a0 + a3 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1 -1  0)  ( a1 )  = ( a1 - a2 )
   *                    1         ( 0 -1  1  0)  ( a2 )    ( a2 - a1 )
   *                              ( 1  0  0  1)  ( a3 )    ( a3 + a0 )
	 
   * Therefore the top components are
      
   *      ( b0r + i b0i )  =  ( {a0r + a3r} + i{a0i + a3i} )
   *      ( b1r + i b1i )     ( {a1r - a2r} + i{a1i - a2i} )
      
   * The bottom components of be may be reconstructed using the formula

   *      ( b2r + i b2i )  =  ( {a2r - a1r} + i{a2i - a1i} )  =  ( - b1r - i b1i )
   *      ( b3r + i b3i )     ( {a3r + a0r} + i{a3i + a0i} )     (   b0r + i b0i ) 
   */
  d.elem(0) = s1.elem(0) + s1.elem(3);
  d.elem(1) = s1.elem(1) - s1.elem(2);

  return d;
}
    
template<class T>
inline typename UnaryReturn<PSpinVector<T,4>, FnSpinProjectDir2Minus>::Type_t
spinProjectDir2Minus(const PSpinVector<T,4>& s1)
{
  typename UnaryReturn<PSpinVector<T,4>, FnSpinProjectDir2Minus>::Type_t  d;

  /*                              ( 1  0 -i  0)  ( a0 )    ( a0 - i a2 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1  0  i)  ( a1 )  = ( a1 + i a3 )
   *                    2         ( i  0  1  0)  ( a2 )    ( a2 + i a0 )
   *                              ( 0 -i  0  1)  ( a3 )    ( a3 - i a1 )

   * Therefore the top components are
      
   *      ( b0r + i b0i )  =  ( {a0r + a2i} + i{a0i - a2r} )
   *      ( b1r + i b1i )     ( {a1r - a3i} + i{a1i + a3r} )
      
   * The bottom components of be may be reconstructed using the formula

   *      ( b2r + i b2i )  =  ( {a2r - a0i} + i{a2i + a0r} )  =  ( - b0i + i b0r )
   *      ( b3r + i b3i )     ( {a3r + a1i} + i{a3i - a1r} )     (   b1i - i b1r )
   */
  d.elem(0) = s1.elem(0) - timesI(s1.elem(2));
  d.elem(1) = s1.elem(1) + timesI(s1.elem(3));

  return d;
}
    
template<class T>
inline typename UnaryReturn<PSpinVector<T,4>, FnSpinProjectDir3Minus>::Type_t
spinProjectDir3Minus(const PSpinVector<T,4>& s1)
{
  typename UnaryReturn<PSpinVector<T,4>, FnSpinProjectDir3Minus>::Type_t  d;

  /*                              ( 1  0 -1  0)  ( a0 )    ( a0 - a2 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1  0 -1)  ( a1 )  = ( a1 - a3 )
   *                    3         (-1  0  1  0)  ( a2 )    ( a2 - a0 )
   *                              ( 0 -1  0  1)  ( a3 )    ( a3 - a1 )
      
   * Therefore the top components are
      
   *      ( b0r + i b0i )  =  ( {a0r - a2r} + i{a0i - a2i} )
   *      ( b1r + i b1i )     ( {a1r - a3r} + i{a1i - a3i} )

   * The bottom components of be may be reconstructed using the formula

   *      ( b2r + i b2i )  =  ( {a2r - a0r} + i{a2i - a0i} )  =  ( - b0r - i b0i )
   *      ( b3r + i b3i )     ( {a3r - a1r} + i{a3i - a1i} )     ( - b1r - i b1i ) 
   */
  d.elem(0) = s1.elem(0) - s1.elem(2);
  d.elem(1) = s1.elem(1) - s1.elem(3);

  return d;
}

template<class T>
inline typename UnaryReturn<PSpinVector<T,4>, FnSpinProjectDir0Plus>::Type_t
spinProjectDir0Plus(const PSpinVector<T,4>& s1)
{
  typename UnaryReturn<PSpinVector<T,4>, FnSpinProjectDir0Plus>::Type_t  d;

  /*                              ( 1  0  0 +i)  ( a0 )    ( a0 + i a3 )
   *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1 +i  0)  ( a1 )  = ( a1 + i a2 )
   *                    0         ( 0 -i  1  0)  ( a2 )    ( a2 - i a1 )
   *                              (-i  0  0  1)  ( a3 )    ( a3 - i a0 )

   * Therefore the top components are

   *      ( b0r + i b0i )  =  ( {a0r - a3i} + i{a0i + a3r} )
   *      ( b1r + i b1i )     ( {a1r - a2i} + i{a1i + a2r} )

   * The bottom components of be may be reconstructed using the formula

   *      ( b2r + i b2i )  =  ( {a2r + a1i} + i{a2i - a1r} )  =  ( b1i - i b1r )
   *      ( b3r + i b3i )     ( {a3r + a0i} + i{a3i - a0r} )     ( b0i - i b0r ) 
   */
  d.elem(0) = s1.elem(0) + timesI(s1.elem(3));
  d.elem(1) = s1.elem(1) + timesI(s1.elem(2));

  return d;
}

template<class T>
inline typename UnaryReturn<PSpinVector<T,4>, FnSpinProjectDir1Plus>::Type_t
spinProjectDir1Plus(const PSpinVector<T,4>& s1)
{
  typename UnaryReturn<PSpinVector<T,4>, FnSpinProjectDir1Plus>::Type_t  d;

  /*                              ( 1  0  0 -1)  ( a0 )    ( a0 - a3 )
   *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  1  0)  ( a1 )  = ( a1 + a2 )
   *                    1         ( 0  1  1  0)  ( a2 )    ( a2 + a1 )
   *                              (-1  0  0  1)  ( a3 )    ( a3 - a0 )

   * Therefore the top components are

   *      ( b0r + i b0i )  =  ( {a0r - a3r} + i{a0i - a3i} )
   *      ( b1r + i b1i )     ( {a1r + a2r} + i{a1i + a2i} )

   * The bottom components of be may be reconstructed using the formula

   *      ( b2r + i b2i )  =  ( {a2r + a1r} + i{a2i + a1i} )  =  (   b1r + i b1i )
   *      ( b3r + i b3i )     ( {a3r - a0r} + i{a3i - a0i} )     ( - b0r - i b0i ) 
   */
  d.elem(0) = s1.elem(0) - s1.elem(3);
  d.elem(1) = s1.elem(1) + s1.elem(2);

  return d;
}

template<class T>
inline typename UnaryReturn<PSpinVector<T,4>, FnSpinProjectDir2Plus>::Type_t
spinProjectDir2Plus(const PSpinVector<T,4>& s1)
{
  typename UnaryReturn<PSpinVector<T,4>, FnSpinProjectDir2Plus>::Type_t  d;

  /*                              ( 1  0  i  0)  ( a0 )    ( a0 + i a2 )
   *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  0 -i)  ( a1 )  = ( a1 - i a3 )
   *                    2         (-i  0  1  0)  ( a2 )    ( a2 - i a0 )
   *                              ( 0  i  0  1)  ( a3 )    ( a3 + i a1 )

   * Therefore the top components are

   *      ( b0r + i b0i )  =  ( {a0r - a2i} + i{a0i + a2r} )
   *      ( b1r + i b1i )     ( {a1r + a3i} + i{a1i - a3r} )

   * The bottom components of be may be reconstructed using the formula

   *      ( b2r + i b2i )  =  ( {a2r + a0i} + i{a2i - a0r} )  =  (   b0i - i b0r )
   *      ( b3r + i b3i )     ( {a3r - a1i} + i{a3i + a1r} )     ( - b1i + i b1r ) 
   */
  d.elem(0) = s1.elem(0) + timesI(s1.elem(2));
  d.elem(1) = s1.elem(1) - timesI(s1.elem(3));

  return d;
}

template<class T>
inline typename UnaryReturn<PSpinVector<T,4>, FnSpinProjectDir3Plus>::Type_t
spinProjectDir3Plus(const PSpinVector<T,4>& s1)
{
  typename UnaryReturn<PSpinVector<T,4>, FnSpinProjectDir3Plus>::Type_t  d;

  /*                              ( 1  0  1  0)  ( a0 )    ( a0 + a2 )
   *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  0  1)  ( a1 )  = ( a1 + a3 )
   *                    3         ( 1  0  1  0)  ( a2 )    ( a2 + a0 )
   *                              ( 0  1  0  1)  ( a3 )    ( a3 + a1 )

   * Therefore the top components are

   *      ( b0r + i b0i )  =  ( {a0r + a2r} + i{a0i + a2i} )
   *      ( b1r + i b1i )     ( {a1r + a3r} + i{a1i + a3i} )

   * The bottom components of be may be reconstructed using the formula

   *      ( b2r + i b2i )  =  ( {a2r + a0r} + i{a2i + a0i} )  =  ( b0r + i b0i )
   *      ( b3r + i b3i )     ( {a3r + a1r} + i{a3i + a1i} )     ( b1r + i b1i ) 
   */
  d.elem(0) = s1.elem(0) + s1.elem(2);
  d.elem(1) = s1.elem(1) + s1.elem(3);

  return d;
}


// SpinVector<4> = SpinReconstruct(SpinVector<2>)
// There are 4 cases here for Nd=4 for each forward/backward direction
template<class T>
inline typename UnaryReturn<PSpinVector<T,2>, FnSpinReconstructDir0Minus>::Type_t
spinReconstructDir0Minus(const PSpinVector<T,2>& s1)
{
  typename UnaryReturn<PSpinVector<T,2>, FnSpinReconstructDir0Minus>::Type_t  d;

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  d.elem(2) = timesI(s1.elem(1));
  d.elem(3) = timesI(s1.elem(0));

  return d;
}

template<class T>
inline typename UnaryReturn<PSpinVector<T,2>, FnSpinReconstructDir1Minus>::Type_t
spinReconstructDir1Minus(const PSpinVector<T,2>& s1)
{
  typename UnaryReturn<PSpinVector<T,2>, FnSpinReconstructDir1Minus>::Type_t  d;

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  d.elem(2) = -s1.elem(1);
  d.elem(3) = s1.elem(0);

  return d;
}


template<class T>
inline typename UnaryReturn<PSpinVector<T,2>, FnSpinReconstructDir2Minus>::Type_t
spinReconstructDir2Minus(const PSpinVector<T,2>& s1)
{
  typename UnaryReturn<PSpinVector<T,2>, FnSpinReconstructDir2Minus>::Type_t  d;

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  d.elem(2) = timesI(s1.elem(0));
  d.elem(3) = timesMinusI(s1.elem(1));

  return d;
}

template<class T>
inline typename UnaryReturn<PSpinVector<T,2>, FnSpinReconstructDir3Minus>::Type_t
spinReconstructDir3Minus(const PSpinVector<T,2>& s1)
{
  typename UnaryReturn<PSpinVector<T,2>, FnSpinReconstructDir3Minus>::Type_t  d;

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  d.elem(2) = -s1.elem(0);
  d.elem(3) = -s1.elem(1);

  return d;
}

template<class T>
inline typename UnaryReturn<PSpinVector<T,2>, FnSpinReconstructDir0Plus>::Type_t
spinReconstructDir0Plus(const PSpinVector<T,2>& s1)
{
  typename UnaryReturn<PSpinVector<T,2>, FnSpinReconstructDir0Plus>::Type_t  d;

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  d.elem(2) = timesMinusI(s1.elem(1));
  d.elem(3) = timesMinusI(s1.elem(0));

  return d;
}

template<class T>
inline typename UnaryReturn<PSpinVector<T,2>, FnSpinReconstructDir1Plus>::Type_t
spinReconstructDir1Plus(const PSpinVector<T,2>& s1)
{
  typename UnaryReturn<PSpinVector<T,2>, FnSpinReconstructDir1Plus>::Type_t  d;

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  d.elem(2) = s1.elem(1);
  d.elem(3) = -s1.elem(0);

  return d;
}

template<class T>
inline typename UnaryReturn<PSpinVector<T,2>, FnSpinReconstructDir2Plus>::Type_t
spinReconstructDir2Plus(const PSpinVector<T,2>& s1)
{
  typename UnaryReturn<PSpinVector<T,2>, FnSpinReconstructDir2Plus>::Type_t  d;

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  d.elem(2) = timesMinusI(s1.elem(0));
  d.elem(3) = timesI(s1.elem(1));

  return d;
}

template<class T>
inline typename UnaryReturn<PSpinVector<T,2>, FnSpinReconstructDir3Plus>::Type_t
spinReconstructDir3Plus(const PSpinVector<T,2>& s1)
{
  typename UnaryReturn<PSpinVector<T,2>, FnSpinReconstructDir3Plus>::Type_t  d;

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  d.elem(2) = s1.elem(0);
  d.elem(3) = s1.elem(1);

  return d;
}

//-----------------------------------------------------------------------------
//! PSpinVector<T,4> = P_+ * PSpinVector<T,4>
template<class T>
inline typename UnaryReturn<PSpinVector<T,4>, FnChiralProjectPlus>::Type_t
chiralProjectPlus(const PSpinVector<T,4>& s1)
{
  typename UnaryReturn<PSpinVector<T,4>, FnChiralProjectPlus>::Type_t  d;

  d.elem(0) = s1.elem(0);
  d.elem(1) = s1.elem(1);
  zero_rep(d.elem(2));
  zero_rep(d.elem(3));

  return d;
}

//! PSpinVector<T,4> = P_- * PSpinVector<T,4>
template<class T>
inline typename UnaryReturn<PSpinVector<T,4>, FnChiralProjectMinus>::Type_t
chiralProjectMinus(const PSpinVector<T,4>& s1)
{
  typename UnaryReturn<PSpinVector<T,4>, FnChiralProjectMinus>::Type_t  d;

  zero_rep(d.elem(0));
  zero_rep(d.elem(1));
  d.elem(2) = s1.elem(2);
  d.elem(3) = s1.elem(3);

  return d;
}


/*! @} */   // end of group primspinvector

QDP_END_NAMESPACE();

