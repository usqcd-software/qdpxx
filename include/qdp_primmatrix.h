// -*- C++ -*-
// $Id: qdp_primmatrix.h,v 1.4 2003-07-31 19:29:09 edwards Exp $

/*! \file
 * \brief Primitive Matrix
 */


QDP_BEGIN_NAMESPACE(QDP);

//-------------------------------------------------------------------------------------
/*! \addtogroup primmatrix Matrix primitive
 * \ingroup fiber
 *
 * Primitive type that transforms like a matrix
 *
 * @{
 */


//! Primitive Matrix class
/*!
 * All Matrix classes inherit this class
 * NOTE: For efficiency, there can be no virtual methods, so the data
 * portion is a part of the generic class, hence it is called a domain
 * and not a category
 */
template <class T, int N, template<class,int> class C> class PMatrix
{
public:
  PMatrix() {}
  ~PMatrix() {}

  typedef C<T,N>  CC;

  //! PMatrix = PScalar
  /*! Fill with primitive scalar */
  template<class T1>
  inline
  CC& assign(const PScalar<T1>& rhs)
    {
      for(int i=0; i < N; ++i)
	for(int j=0; j < N; ++j)
	  if (i == j)
	    elem(i,j) = rhs.elem();
	  else
	    zero_rep(elem(i,j));

      return static_cast<CC&>(*this);
    }

  //! PMatrix = PMatrix
  /*! Set equal to another PMatrix */
  inline
  CC& assign(const CC& rhs) 
    {
      for(int i=0; i < N; ++i)
	for(int j=0; j < N; ++j)
	  elem(i,j) = rhs.elem(i,j);

      return static_cast<CC&>(*this);
    }

  //! PMatrix += PMatrix
  inline
  CC& operator+=(const CC& rhs) 
    {
      for(int i=0; i < N; ++i)
	for(int j=0; j < N; ++j)
	  elem(i,j) += rhs.elem(i,j);

      return static_cast<CC&>(*this);
    }

  //! PMatrix -= PMatrix
  inline
  CC& operator-=(const CC& rhs) 
    {
      for(int i=0; i < N; ++i)
	for(int j=0; j < N; ++j)
	  elem(i,j) -= rhs.elem(i,j);

      return static_cast<CC&>(*this);
    }

  //! PMatrix += PScalar
  template<class T1>
  inline
  CC& operator+=(const PScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	for(int j=0; j < N; ++j)
	  elem(i,j) += rhs.elem();

      return static_cast<CC&>(*this);
    }

  //! PMatrix -= PScalar
  template<class T1>
  inline
  CC& operator-=(const PScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	for(int j=0; j < N; ++j)
	  elem(i,j) -= rhs.elem();

      return static_cast<CC&>(*this);
    }

  //! PMatrix *= PScalar
  template<class T1>
  inline
  CC& operator*=(const PScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	for(int j=0; j < N; ++j)
	  elem(i,j) *= rhs.elem();

      return static_cast<CC&>(*this);
    }

  //! PMatrix /= PScalar
  template<class T1>
  inline
  CC& operator/=(const PScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	for(int j=0; j < N; ++j)
	  elem(i,j) /= rhs.elem();

      return static_cast<CC&>(*this);
    }

  //! Deep copy here
#if defined(QDP_USE_ARRAY_INITIALIZER)
  /*! This is an array initializer form - may not be strictly legal */
  PMatrix(const PMatrix& a) : F(a.F) {}
#else
  /*! This is a copy form - legal but not necessarily efficient */
  PMatrix(const PMatrix& a)
    {
      for(int i=0; i < N; ++i)
	for(int j=0; j < N; ++j)
	  F[i][j] = a.F[i][j];
    }
#endif

public:
  T& elem(int i, int j) {return F[i][j];}
  const T& elem(int i, int j) const {return F[i][j];}

private:
  T F[N][N];
};


//! Nml output
template<class T, int N, template<class,int> class C>  
inline
NmlWriter& operator<<(NmlWriter& nml, const PMatrix<T,N,C>& d)
{
  nml.get() << "  [MATRIX]\n";
  for(int j=0; j < N-1; ++j)
    for(int i=0; i < N; ++i)
    {
      nml.get() << "\tRow = " << i << ", Column = " << j << " = ";
      nml << d.elem(i,j);
      nml.get() << ",\n";
    }
    
  int j=N-1;
  for(int i=0; i < N-1; ++i)
  {
    nml.get() << "\tRow = " << i << ", Column = " << j << " = ";
    nml << d.elem(i,j);
    nml.get() << ",\n";
  }
    
  int i = j;
  nml.get() << "\tRow = " << i << ", Column = " << j << " = ";
  nml << d.elem(i,j);

  return nml;
}


//! XML output
template<class T, int N, template<class,int> class C>  
inline
XMLWriter& operator<<(XMLWriter& xml, const PMatrix<T,N,C>& d)
{
  xml.openTag("Matrix");

  XMLWriterAPI::AttributeList alist;

  for(int i=0; i < N; ++i)
  {
    for(int j=0; j < N; ++j)
    {
      alist.clear();
      alist.push_back(XMLWriterAPI::Attribute("row", i));
      alist.push_back(XMLWriterAPI::Attribute("col", j));

      xml.openTag("elem", alist);
      xml << d.elem(i,j);
      xml.closeTag();
    }
  }

  xml.closeTag(); // Matrix
  return xml;
}

/*! @} */  // end of group primmatrix

//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

// Underlying word type
template<class T1, int N, template<class,int> class C>
struct WordType<PMatrix<T1,N,C> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};

// Internally used scalars
template<class T, int N, template<class,int> class C>
struct InternalScalar<PMatrix<T,N,C> > {
  typedef PScalar<typename InternalScalar<T>::Type_t>  Type_t;
};


//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------


// Default unary(PMatrix) -> PMatrix
template<class T, int N, template<class,int> class C, class Op>
struct UnaryReturn<PMatrix<T,N,C>, Op> {
  typedef C<typename UnaryReturn<T, Op>::Type_t, N>  Type_t;
};

// Default binary(PScalar,PMatrix) -> PMatrix
template<class T1, class T2, int N, template<class,int> class C, class Op>
struct BinaryReturn<PScalar<T1>, PMatrix<T2,N,C>, Op> {
  typedef C<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PMatrix,PMatrix) -> PMatrix
template<class T1, class T2, int N, template<class,int> class C, class Op>
struct BinaryReturn<PMatrix<T1,N,C>, PMatrix<T2,N,C>, Op> {
  typedef C<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PMatrix,PScalar) -> PMatrix
template<class T1, int N, class T2, template<class,int> class C, class Op>
struct BinaryReturn<PMatrix<T1,N,C>, PScalar<T2>, Op> {
  typedef C<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};



#if 0
template<class T1, class T2>
struct UnaryReturn<PScalar<T2>, OpCast<T1> > {
  typedef PScalar<typename UnaryReturn<T, OpCast>::Type_t>  Type_t;
//  typedef T1 Type_t;
};
#endif


// Assignment is different
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrix<T1,N,C>, PMatrix<T2,N,C>, OpAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrix<T1,N,C>, PMatrix<T2,N,C>, OpAddAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrix<T1,N,C>, PMatrix<T2,N,C>, OpSubtractAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrix<T1,N,C>, PScalar<T2>, OpAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrix<T1,N,C>, PScalar<T2>, OpAddAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrix<T1,N,C>, PScalar<T2>, OpSubtractAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrix<T1,N,C>, PScalar<T2>, OpMultiplyAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrix<T1,N,C>, PScalar<T2>, OpDivideAssign > {
  typedef C<T1,N> &Type_t;
};
 


//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------
/*! \addtogroup primmatrix */
/*! @{ */

// Primitive Matrices
template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrix<T1,N,C>, OpUnaryPlus>::Type_t
operator+(const PMatrix<T1,N,C>& l)
{
  typename UnaryReturn<PMatrix<T1,N,C>, OpUnaryPlus>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = +l.elem(i,j);

  return d;
}


template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrix<T1,N,C>, OpUnaryMinus>::Type_t
operator-(const PMatrix<T1,N,C>& l)
{
  typename UnaryReturn<PMatrix<T1,N,C>, OpUnaryMinus>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = -l.elem(i,j);

  return d;
}


template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrix<T1,N,C>, PMatrix<T2,N,C>, OpAdd>::Type_t
operator+(const PMatrix<T1,N,C>& l, const PMatrix<T2,N,C>& r)
{
  typename BinaryReturn<PMatrix<T1,N,C>, PMatrix<T2,N,C>, OpAdd>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = l.elem(i,j) + r.elem(i,j);

  return d;
}


template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrix<T1,N,C>, PMatrix<T2,N,C>, OpSubtract>::Type_t
operator-(const PMatrix<T1,N,C>& l, const PMatrix<T2,N,C>& r)
{
  typename BinaryReturn<PMatrix<T1,N,C>, PMatrix<T2,N,C>, OpSubtract>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = l.elem(i,j) - r.elem(i,j);

  return d;
}


template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrix<T1,N,C>, PScalar<T2>, OpMultiply>::Type_t
operator*(const PMatrix<T1,N,C>& l, const PScalar<T2>& r)
{
  typename BinaryReturn<PMatrix<T1,N,C>, PScalar<T2>, OpMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = l.elem(i,j) * r.elem();
  return d;
}


template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalar<T1>, PMatrix<T2,N,C>, OpMultiply>::Type_t
operator*(const PScalar<T1>& l, const PMatrix<T2,N,C>& r)
{
  typename BinaryReturn<PScalar<T1>, PMatrix<T2,N,C>, OpMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = l.elem() * r.elem(i,j);
  return d;
}


template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrix<T1,N,C>, PMatrix<T2,N,C>, OpMultiply>::Type_t
operator*(const PMatrix<T1,N,C>& l, const PMatrix<T2,N,C>& r)
{
  typename BinaryReturn<PMatrix<T1,N,C>, PMatrix<T2,N,C>, OpMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
    {
      d.elem(i,j) = l.elem(i,0) * r.elem(0,j);
      for(int k=1; k < N; ++k)
	d.elem(i,j) += l.elem(i,k) * r.elem(k,j);
    }

  return d;
}


template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrix<T1,N,C>, PScalar<T2>, OpDivide>::Type_t
operator/(const PMatrix<T1,N,C>& l, const PScalar<T2>& r)
{
  typename BinaryReturn<PMatrix<T1,N,C>, PScalar<T2>, OpDivide>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = l.elem(i,j) / r.elem();
  return d;
}



//-----------------------------------------------------------------------------
// Functions

// Adjoint
template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrix<T1,N,C>, FnAdjoint>::Type_t
adj(const PMatrix<T1,N,C>& l)
{
  typename UnaryReturn<PMatrix<T1,N,C>, FnAdjoint>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = adj(l.elem(j,i));

  return d;
}


// Conjugate
template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrix<T1,N,C>, FnConjugate>::Type_t
conj(const PMatrix<T1,N,C>& l)
{
  typename UnaryReturn<PMatrix<T1,N,C>, FnConjugate>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = conj(l.elem(i,j));

  return d;
}


// Transpose
template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrix<T1,N,C>, FnTranspose>::Type_t
transpose(const PMatrix<T1,N,C>& l)
{
  typename UnaryReturn<PMatrix<T1,N,C>, FnTranspose>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = transpose(l.elem(j,i));

  return d;
}


// TRACE
// trace = Trace(source1)
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrix<T,N,C>, FnTrace > {
  typedef PScalar<typename UnaryReturn<T, FnTrace>::Type_t>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrix<T,N,C>, FnTrace>::Type_t
trace(const PMatrix<T,N,C>& s1)
{
  typename UnaryReturn<PMatrix<T,N,C>, FnTrace>::Type_t  d;

  d.elem() = trace(s1.elem(0,0));
  for(int i=1; i < N; ++i)
    d.elem() += trace(s1.elem(i,i));

  return d;
}


// trace = Re(Trace(source1))
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrix<T,N,C>, FnRealTrace > {
  typedef PScalar<typename UnaryReturn<T, FnRealTrace>::Type_t>  Type_t;
};

template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrix<T1,N,C>, FnRealTrace>::Type_t
realTrace(const PMatrix<T1,N,C>& s1)
{
  typename UnaryReturn<PMatrix<T1,N,C>, FnRealTrace>::Type_t  d;

  d.elem() = realTrace(s1.elem(0,0));
  for(int i=1; i < N; ++i)
    d.elem() += realTrace(s1.elem(i,i));

  return d;
}


//! trace = Im(Trace(source1))
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrix<T,N,C>, FnImagTrace > {
  typedef PScalar<typename UnaryReturn<T, FnImagTrace>::Type_t>  Type_t;
};

template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrix<T1,N,C>, FnImagTrace>::Type_t
imagTrace(const PMatrix<T1,N,C>& s1)
{
  typename UnaryReturn<PMatrix<T1,N,C>, FnImagTrace>::Type_t  d;

  d.elem() = imagTrace(s1.elem(0,0));
  for(int i=1; i < N; ++i)
    d.elem() += imagTrace(s1.elem(i,i));

  return d;
}


//! trace = traceColor(source1)   [this is an identity in general]
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrix<T,N,C>, FnTraceColor>::Type_t
traceColor(const PMatrix<T,N,C>& s1)
{
  typename UnaryReturn<PMatrix<T,N,C>, FnTraceColor>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = traceColor(s1.elem(i,j));

  return d;
}


//! trace = traceSpin(source1)   [this is an identity in general]
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrix<T,N,C>, FnTraceSpin>::Type_t
traceSpin(const PMatrix<T,N,C>& s1)
{
  typename UnaryReturn<PMatrix<T,N,C>, FnTraceSpin>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = traceSpin(s1.elem(i,j));

  return d;
}


//! PMatrix = Re(PMatrix)
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrix<T,N,C>, FnReal>::Type_t
real(const PMatrix<T,N,C>& s1)
{
  typename UnaryReturn<PMatrix<T,N,C>, FnReal>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = real(s1.elem(i,j));

  return d;
}


//! PMatrix = Im(PMatrix)
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrix<T,N,C>, FnImag>::Type_t
imag(const PMatrix<T,N,C>& s1)
{
  typename UnaryReturn<PMatrix<T,N,C>, FnImag>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = imag(s1.elem(i,j));

  return d;
}


//! PMatrix<T> = (PMatrix<T> , PMatrix<T>)
template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrix<T1,N,C>, PMatrix<T2,N,C>, FnCmplx>::Type_t
cmplx(const PMatrix<T1,N,C>& s1, const PMatrix<T2,N,C>& s2)
{
  typename BinaryReturn<PMatrix<T1,N,C>, PMatrix<T2,N,C>, FnCmplx>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = cmplx(s1.elem(i,j), s2.elem(i,j));

  return d;
}




// Functions
//! PMatrix = i * PMatrix
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrix<T,N,C>, FnTimesI>::Type_t
timesI(const PMatrix<T,N,C>& s1)
{
  typename UnaryReturn<PMatrix<T,N,C>, FnTimesI>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = timesI(s1.elem(i,j));

  return d;
}

//! PMatrix = -i * PMatrix
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrix<T,N,C>, FnTimesMinusI>::Type_t
timesMinusI(const PMatrix<T,N,C>& s1)
{
  typename UnaryReturn<PMatrix<T,N,C>, FnTimesMinusI>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = timesMinusI(s1.elem(i,j));

  return d;
}

//! Extract color vector components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrix<T,N,C>, FnPeekColorVector>::Type_t
peekColor(const PMatrix<T,N,C>& l, int row)
{
  typename UnaryReturn<PMatrix<T,N,C>, FnPeekColorVector>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = peekColor(l.elem(i,j),row);
  return d;
}

//! Extract color matrix components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrix<T,N,C>, FnPeekColorMatrix>::Type_t
peekColor(const PMatrix<T,N,C>& l, int row, int col)
{
  typename UnaryReturn<PMatrix<T,N,C>, FnPeekColorMatrix>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = peekColor(l.elem(i,j),row,col);
  return d;
}

//! Extract spin vector components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrix<T,N,C>, FnPeekSpinVector>::Type_t
peekSpin(const PMatrix<T,N,C>& l, int row)
{
  typename UnaryReturn<PMatrix<T,N,C>, FnPeekSpinVector>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = peekSpin(l.elem(i,j),row);
  return d;
}

//! Extract spin matrix components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrix<T,N,C>, FnPeekSpinMatrix>::Type_t
peekSpin(const PMatrix<T,N,C>& l, int row, int col)
{
  typename UnaryReturn<PMatrix<T,N,C>, FnPeekSpinMatrix>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = peekSpin(l.elem(i,j),row,col);
  return d;
}


//! Insert color vector components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T1, class T2, int N, template<class,int> class C>
inline PMatrix<T1,N,C>&
pokeColor(PMatrix<T1,N,C>& l, const PMatrix<T2,N,C>& r, int row)
{
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      pokeColor(l.elem(i,j),r.elem(i,j),row);
  return l;
}

//! Insert color matrix components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T1, class T2, int N, template<class,int> class C>
inline PMatrix<T1,N,C>&
pokeColor(PMatrix<T1,N,C>& l, const PMatrix<T2,N,C>& r, int row, int col)
{
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      pokeColor(l.elem(i,j),r.elem(i,j),row,col);
  return l;
}

//! Insert spin vector components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T1, class T2, int N, template<class,int> class C>
inline PMatrix<T1,N,C>&
pokeSpin(PMatrix<T1,N,C>& l, const PMatrix<T2,N,C>& r, int row)
{
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      pokeSpin(l.elem(i,j),r.elem(i,j),row);
  return l;
}

//! Insert spin matrix components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T1, class T2, int N, template<class,int> class C>
inline PMatrix<T1,N,C>&
pokeSpin(PMatrix<T1,N,C>& l, const PMatrix<T2,N,C>& r, int row, int col)
{
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      pokeSpin(l.elem(i,j),r.elem(i,j),row,col);
  return l;
}



//! dest = 0
template<class T, int N, template<class,int> class C> 
inline void 
zero_rep(PMatrix<T,N,C>& dest) 
{
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      zero_rep(dest.elem(i,j));
}


//! dest = (mask) ? s1 : dest
template<class T, class T1, int N, template<class,int> class C> 
inline void 
copymask(PMatrix<T,N,C>& d, const PScalar<T1>& mask, const PMatrix<T,N,C>& s1) 
{
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      copymask(d.elem(i,j),mask.elem(),s1.elem(i,j));
}


//! dest  = random  
template<class T, int N, template<class,int> class C, class T1, class T2>
inline void
fill_random(PMatrix<T,N,C>& d, T1& seed, T2& skewed_seed, const T1& seed_mult)
{
  // The skewed_seed is the starting seed to use
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      fill_random(d.elem(i,j), seed, skewed_seed, seed_mult);
}

//! dest  = gaussian
template<class T, int N, template<class,int> class C>
inline void
fill_gaussian(PMatrix<T,N,C>& d, PMatrix<T,N,C>& r1, PMatrix<T,N,C>& r2)
{
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      fill_gaussian(d.elem(i,j), r1.elem(i,j), r2.elem(i,j));
}



#if 0
// Global sum over site indices only
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrix<T,N,C>, FnSum > {
  typedef C<typename UnaryReturn<T, FnSum>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrix<T,N,C>, FnSum>::Type_t
sum(const PMatrix<T,N,C>& s1)
{
  typename UnaryReturn<PMatrix<T,N,C>, FnSum>::Type_t  d;

  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = sum(s1.elem(i,j));

  return d;
}
#endif


// InnerProduct (norm-seq) global sum = sum(tr(adj(s1)*s1))
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrix<T,N,C>, FnNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T, int N, template<class,int> class C>
struct UnaryReturn<PMatrix<T,N,C>, FnLocalNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PMatrix<T,N,C>, FnLocalNorm2>::Type_t
localNorm2(const PMatrix<T,N,C>& s1)
{
  typename UnaryReturn<PMatrix<T,N,C>, FnLocalNorm2>::Type_t  d;

  d.elem() = localNorm2(s1.elem(0,0));
  for(int j=1; j < N; ++j)
    d.elem() += localNorm2(s1.elem(0,j));

  for(int i=1; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem() += localNorm2(s1.elem(i,j));

  return d;
}


//! PScalar<T> = InnerProduct(adj(PMatrix<T1>)*PMatrix<T1>)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrix<T1,N,C>, PMatrix<T2,N,C>, FnInnerProduct > {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrix<T1,N,C>, PMatrix<T2,N,C>, FnLocalInnerProduct > {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrix<T1,N,C>, PMatrix<T2,N,C>, FnLocalInnerProduct>::Type_t
localInnerProduct(const PMatrix<T1,N,C>& s1, const PMatrix<T2,N,C>& s2)
{
  typename BinaryReturn<PMatrix<T1,N,C>, PMatrix<T2,N,C>, FnLocalInnerProduct>::Type_t  d;

  d.elem() = localInnerProduct(s1.elem(0,0), s2.elem(0,0));
  for(int k=1; k < N; ++k)
    d.elem() += localInnerProduct(s1.elem(k,0), s2.elem(k,0));

  for(int j=1; j < N; ++j)
    for(int k=0; k < N; ++k)
      d.elem() += localInnerProduct(s1.elem(k,j), s2.elem(k,j));

  return d;
}


//! PScalar<T> = InnerProductReal(adj(PMatrix<T1>)*PMatrix<T1>)
/*!
 * return  realpart of InnerProduct(adj(s1)*s2)
 */
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrix<T1,N,C>, PMatrix<T2,N,C>, FnInnerProductReal > {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PMatrix<T1,N,C>, PMatrix<T2,N,C>, FnLocalInnerProductReal > {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrix<T1,N,C>, PMatrix<T2,N,C>, FnLocalInnerProductReal>::Type_t
localInnerProductReal(const PMatrix<T1,N,C>& s1, const PMatrix<T2,N,C>& s2)
{
  typename BinaryReturn<PMatrix<T1,N,C>, PMatrix<T2,N,C>, FnLocalInnerProductReal>::Type_t  d;

  d.elem() = localInnerProductReal(s1.elem(0,0), s2.elem(0,0));
  for(int k=1; k < N; ++k)
    d.elem() += localInnerProductReal(s1.elem(k,0), s2.elem(k,0));

  for(int j=1; j < N; ++j)
    for(int k=0; k < N; ++k)
      d.elem() += localInnerProductReal(s1.elem(k,j), s2.elem(k,j));

  return d;
}


//! PMatrix<T> = where(PScalar, PMatrix, PMatrix)
/*!
 * Where is the ? operation
 * returns  (a) ? b : c;
 */
template<class T1, class T2, class T3, int N, template<class,int> class C>
inline typename BinaryReturn<PMatrix<T2,N,C>, PMatrix<T3,N,C>, FnWhere>::Type_t
where(const PScalar<T1>& a, const PMatrix<T2,N,C>& b, const PMatrix<T3,N,C>& c)
{
  typename BinaryReturn<PMatrix<T2,N,C>, PMatrix<T3,N,C>, FnWhere>::Type_t  d;

  // Not optimal - want to have where outside assignment
  for(int i=0; i < N; ++i)
    for(int j=0; j < N; ++j)
      d.elem(i,j) = where(a.elem(), b.elem(i,j), c.elem(i,j));

  return d;
}

/*! @} */  // end of group primmatrix

QDP_END_NAMESPACE();

