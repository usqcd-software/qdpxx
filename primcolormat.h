// -*- C++ -*-
// $Id: primcolormat.h,v 1.6 2002-11-07 19:25:50 edwards Exp $

/*! \file
 * \brief Primitive Color Matrix
 */


QDP_BEGIN_NAMESPACE(QDP);

//-------------------------------------------------------------------------------------
/*! \addtogroup primcolormatrix Color matrix primitive
 * \ingroup primmatrix
 *
 * Primitive type that transforms like a Color Matrix
 *
 * @{
 */


//! Primitive color Matrix class 
template <class T, int N> class PColorMatrix : public PMatrix<T, N, PColorMatrix>
{
public:
  //! PColorMatrix = PScalar
  /*! Fill with primitive scalar */
  template<class T1>
  inline
  PColorMatrix& operator=(const PScalar<T1>& rhs)
    {
      assign(rhs);
      return *this;
    }

  //! PColorMatrix = PColorMatrix
  /*! Set equal to another PMatrix */
  inline
  PColorMatrix& operator=(const PColorMatrix& rhs) 
    {
      assign(rhs);
      return *this;
    }

};

/*! @} */   // end of group primcolormatrix

//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

// Underlying word type
template<class T1, int N>
struct WordType<PColorMatrix<T1,N> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};

// Internally used scalars
template<class T, int N>
struct InternalScalar<PColorMatrix<T,N> > {
  typedef PScalar<typename InternalScalar<T>::Type_t>  Type_t;
};

//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

// Default unary(PColorMatrix) -> PColorMatrix
template<class T1, int N, class Op>
struct UnaryReturn<PColorMatrix<T1,N>, Op> {
  typedef PColorMatrix<typename UnaryReturn<T1, Op>::Type_t, N>  Type_t;
};

// Default binary(PScalar,PColorMatrix) -> PColorMatrix
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PScalar<T1>, PColorMatrix<T2,N>, Op> {
  typedef PColorMatrix<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PColorMatrix,PColorMatrix) -> PColorMatrix
template<class T1, class T2, int N, class Op>
struct BinaryReturn<PColorMatrix<T1,N>, PColorMatrix<T2,N>, Op> {
  typedef PColorMatrix<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PColorMatrix,PScalar) -> PColorMatrix
template<class T1, int N, class T2, class Op>
struct BinaryReturn<PColorMatrix<T1,N>, PScalar<T2>, Op> {
  typedef PColorMatrix<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};


// Assignment is different
template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrix<T1,N>, PColorMatrix<T2,N>, OpAssign > {
  typedef PColorMatrix<T1,N> &Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrix<T1,N>, PColorMatrix<T2,N>, OpAddAssign > {
  typedef PColorMatrix<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrix<T1,N>, PColorMatrix<T2,N>, OpSubtractAssign > {
  typedef PColorMatrix<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrix<T1,N>, PColorMatrix<T2,N>, OpMultiplyAssign > {
  typedef PColorMatrix<T1,N> &Type_t;
};
 

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrix<T1,N>, PScalar<T2>, OpAssign > {
  typedef PColorMatrix<T1,N> &Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrix<T1,N>, PScalar<T2>, OpAddAssign > {
  typedef PColorMatrix<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrix<T1,N>, PScalar<T2>, OpSubtractAssign > {
  typedef PColorMatrix<T1,N> &Type_t;
};
 
template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrix<T1,N>, PScalar<T2>, OpMultiplyAssign > {
  typedef PColorMatrix<T1,N> &Type_t;
};
 


// ColorMatrix
template<class T, int N>
struct UnaryReturn<PColorMatrix<T,N>, FnTrace > {
  typedef PScalar<typename UnaryReturn<T, FnTrace>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PColorMatrix<T,N>, FnRealTrace > {
  typedef PScalar<typename UnaryReturn<T, FnRealTrace>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PColorMatrix<T,N>, FnImagTrace > {
  typedef PScalar<typename UnaryReturn<T, FnImagTrace>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PColorMatrix<T,N>, FnNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T, int N>
struct UnaryReturn<PColorMatrix<T,N>, FnLocalNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrix<T1,N>, PColorMatrix<T2,N>, FnInnerproduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerproduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrix<T1,N>, PColorMatrix<T2,N>, FnLocalInnerproduct> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerproduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrix<T1,N>, PColorMatrix<T2,N>, FnInnerproductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerproductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N>
struct BinaryReturn<PColorMatrix<T1,N>, PColorMatrix<T2,N>, FnLocalInnerproductReal> {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerproductReal>::Type_t>  Type_t;
};





//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------

/*! \addtogroup primcolormatrix */
/*! @{ */

// trace = traceColor(source1)
/*! This only acts on color indices and is diagonal in all other indices */
template<class T, int N>
struct UnaryReturn<PColorMatrix<T,N>, FnTraceColor > {
  typedef PScalar<typename UnaryReturn<T, FnTraceColor>::Type_t>  Type_t;
};

template<class T, int N>
inline typename UnaryReturn<PColorMatrix<T,N>, FnTraceColor>::Type_t
traceColor(const PColorMatrix<T,N>& s1)
{
  typename UnaryReturn<PColorMatrix<T,N>, FnTraceColor>::Type_t  d;

  // Since the color index is eaten, do not need to pass on function by
  // calling trace(...) again
  d.elem() = s1.elem(0,0);
  for(int i=1; i < N; ++i)
    d.elem() += s1.elem(i,i);

  return d;
}


//-----------------------------------------------
// Peeking and poking
//! Extract color matrix components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T, int N>
struct UnaryReturn<PColorMatrix<T,N>, FnPeekColorMatrix > {
  typedef PScalar<typename UnaryReturn<T, FnPeekColorMatrix>::Type_t>  Type_t;
};

template<class T, int N>
inline typename UnaryReturn<PColorMatrix<T,N>, FnPeekColorMatrix>::Type_t
peekColor(const PColorMatrix<T,N>& l, int row, int col)
{
  typename UnaryReturn<PColorMatrix<T,N>, FnPeekColorMatrix>::Type_t  d;

  // Note, do not need to propagate down since the function is eaten at this level
  d.elem() = l.elem(row,col);
  return d;
}

//! Insert color matrix components
template<class T1, class T2, int N>
inline PColorMatrix<T1,N>&
pokeColor(PColorMatrix<T1,N>& l, const PScalar<T2>& r, int row, int col)
{
  // Note, do not need to propagate down since the function is eaten at this level
  l.elem(row,col) = r.elem();
  return l;
}


//-----------------------------------------------------------------------------
// Contraction for quark propagators
// QuarkContract 
//! dest  = QuarkContractXX_rep(Qprop1,Qprop2)
/*!
 * This is a really slow implementation for now - I just want it
 * to run and work. It was probably simpler just to completely 
 * unroll the loops which is what I want to do anyway...
 */
template<class T1, class T2>
inline typename BinaryReturn<PColorMatrix<T1,3>, PColorMatrix<T2,3>, FnQuarkContractXX>::Type_t
quarkcontractxx(const PColorMatrix<T1,3>& s1, const PColorMatrix<T2,3>& s2)
{
  typename BinaryReturn<PColorMatrix<T1,3>, PColorMatrix<T2,3>, FnQuarkContractXX>::Type_t  d;

  bool first = true;
  int antisym[3][3][3] = {{{0,0,0},{0,0,1},{0,-1,0}},
			  {{0,0,-1},{0,0,0},{1,0,0}},
			  {{0,1,0},{-1,0,0},{0,0,0}}};

  for(int k1=0; k1 < 3; ++k1)
    for(int j1=0; j1 < 3; ++j1)
      for(int i1=0; i1 < 3; ++i1)
      {
	int e1 = antisym[i1][j1][k1];
	
	if (e1 != 0)
	{
	  for(int k2=0; k2 < 3; ++k2)
	    for(int j2=0; j2 < 3; ++j2)
	      for(int i2=0; i2 < 3; ++i2)
	      {
		int e2 = e1*antisym[i2][j2][k2];

		if (e2 != 0)
		{
		  if (first)
		  {
		    switch(e2)
		    {
		    case 1:
		      d.elem(k2,k1) = s1.elem(i1,j1) * s2.elem(i2,j2);
		      break;
		    case -1:
		      d.elem(k2,k1) = -s1.elem(i1,j1) * s2.elem(i2,j2);
		      break;
		    }

		    first = false;
		  }
		  else
		  {
		    switch(e2)
		    {
		    case 1:
		      d.elem(k2,k1) += s1.elem(i1,j1) * s2.elem(i2,j2);
		      break;
		    case -1:
		      d.elem(k2,k1) -= s1.elem(i1,j1) * s2.elem(i2,j2);
		      break;
		    }
		  }
		}
	      }
	}
      }

  return d;
}

/*! @} */   // end of group primcolormatrix

QDP_END_NAMESPACE();

