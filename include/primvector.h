// -*- C++ -*-
// $Id: primvector.h,v 1.8 2002-10-25 03:33:26 edwards Exp $

/*! \file
 * \brief Primitive Vector
 */


QDP_BEGIN_NAMESPACE(QDP);

//-------------------------------------------------------------------------------------
/*! \addtogroup primvector Vector primitive
 * \ingroup fiber
 *
 * Primitive type that transforms like a vector
 *
 * @{
 */

//! Primitive Vector class
/*!
 * All vector classes inherit this class
 * NOTE: For efficiency, there can be no virtual methods, so the data
 * portion is a part of the generic class, hence it is called a domain
 * and not a category
 */
template <class T, int N, template<class,int> class C> class PVector
{
public:
  PVector() {}
  ~PVector() {}

  typedef C<T,N>  CC;

#if 0
  //! PVector = PScalar
  /*! Fill with a primitive scalar */
  template<class T1>
  inline
  PVector& operator=(const PScalar<T1>& rhs)
    {
      //! This should check on rhs !!!
      for(int i=0; i < N; ++i)
	elem(i) = rhs.elem();

      return *this;
    }
#endif


  //! PVector = PVector
  /*! Set equal to another PVector */
  inline
  CC& assign(const CC& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) = rhs.elem(i);

      return static_cast<CC&>(*this);
    }

  //! PVector = PVector
  /*! Set equal to another PVector */
  inline
  CC& operator=(const CC& rhs) 
    {
      return assign(rhs);
    }

  //! PVector += PVector
  inline
  CC& operator+=(const CC& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) += rhs.elem(i);

      return static_cast<CC&>(*this);
    }

  //! PVector -= PVector
  inline
  CC& operator-=(const CC& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) -= rhs.elem(i);

      return static_cast<CC&>(*this);
    }

  //! PVector *= PScalar
  template<class T1>
  inline
  CC& operator*=(const PScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) *= rhs.elem();

      return static_cast<CC&>(*this);
    }


  //! Deep copy constructor
  PVector(const PVector& a) : F(a.F) {}

public:
  T& elem(int i) {return F[i];}
  const T& elem(int i) const {return F[i];}

private:
  T F[N];
};


//! Ascii output
template<class T, int N, template<class,int> class C> 
NmlWriter& operator<<(NmlWriter& nml, const PVector<T,N,C>& d)
{
  nml.get() << "  [VECTOR]\n";
  for(int i=0; i < N-1; ++i)
  {
    nml.get() << "\tRow = " << i << " = ";
    nml << d.elem(i);
    nml.get() << " ,\n";
  }
    
  int i = N-1;
  nml.get() << "\tRow = " << i << " = ";
  nml << d.elem(i);
    
  return nml;
}

/*! @} */  // end of group primvector


//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

// Underlying word type
template<class T1, int N, template<class,int> class C>
struct WordType<PVector<T1,N,C> > 
{
  typedef typename WordType<T1>::Type_t  Type_t;
};

// Internally used scalars
template<class T, int N, template<class,int> class C>
struct InternalScalar<PVector<T,N,C> > {
  typedef PScalar<typename InternalScalar<T>::Type_t>  Type_t;
};

//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

// Default unary(PVector) -> PVector
template<class T1, int N, template<class,int> class C, class Op>
struct UnaryReturn<PVector<T1,N,C>, Op> {
  typedef C<typename UnaryReturn<T1, Op>::Type_t, N>  Type_t;
};
// Default binary(PScalar,PVector) -> PVector
template<class T1, class T2, int N, template<class,int> class C, class Op>
struct BinaryReturn<PScalar<T1>, PVector<T2,N,C>, Op> {
  typedef C<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PMatrix,PVector) -> PVector
template<class T1, class T2, int N, template<class,int> class C1, 
  template<class,int> class C2, class Op>
struct BinaryReturn<PMatrix<T1,N,C1>, PVector<T2,N,C2>, Op> {
  typedef C2<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PVector,PScalar) -> PVector
template<class T1, class T2, int N, template<class,int> class C, class Op>
struct BinaryReturn<PVector<T1,N,C>, PScalar<T2>, Op> {
  typedef C<typename BinaryReturn<T1, T2, Op>::Type_t, N>  Type_t;
};

// Default binary(PVector,PVector) -> PVector
template<class T1, class T2, int N, template<class,int> class C, class Op>
struct BinaryReturn<PVector<T1,N,C>, PVector<T2,N,C>, Op> {
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
struct BinaryReturn<PVector<T1,N,C>, PVector<T2,N,C>, OpAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PVector<T1,N,C>, PVector<T2,N,C>, OpAddAssign > {
  typedef C<T1,N> &Type_t;
};
 
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PVector<T1,N,C>, PVector<T2,N,C>, OpSubtractAssign > {
  typedef C<T1,N> &Type_t;
};
 



//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------

/*! \addtogroup primvector */
/*! @{ */

// Primitive Vectors

template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PVector<T1,N,C>, OpUnaryPlus>::Type_t
operator+(const PVector<T1,N,C>& l)
{
  typename UnaryReturn<PVector<T1,N,C>, OpUnaryPlus>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = +l.elem(i);
  return d;
}


template<class T1, int N, template<class,int> class C>
inline typename UnaryReturn<PVector<T1,N,C>, OpUnaryMinus>::Type_t
operator-(const PVector<T1,N,C>& l)
{
  typename UnaryReturn<PVector<T1,N,C>, OpUnaryMinus>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = -l.elem(i);
  return d;
}


template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PVector<T1,N,C>, PVector<T2,N,C>, OpAdd>::Type_t
operator+(const PVector<T1,N,C>& l, const PVector<T2,N,C>& r)
{
  typename BinaryReturn<PVector<T1,N,C>, PVector<T2,N,C>, OpAdd>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem(i) + r.elem(i);
  return d;
}


template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PVector<T1,N,C>, PVector<T2,N,C>, OpSubtract>::Type_t
operator-(const PVector<T1,N,C>& l, const PVector<T2,N,C>& r)
{
  typename BinaryReturn<PVector<T1,N,C>, PVector<T2,N,C>, OpSubtract>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem(i) - r.elem(i);
  return d;
}


template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PVector<T1,N,C>, PScalar<T2>, OpMultiply>::Type_t
operator*(const PVector<T1,N,C>& l, const PScalar<T2>& r)
{
  typename BinaryReturn<PVector<T1,N,C>, PScalar<T2>, OpMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem(i) * r.elem();
  return d;
}


template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PScalar<T1>, PVector<T2,N,C>, OpMultiply>::Type_t
operator*(const PScalar<T1>& l, const PVector<T2,N,C>& r)
{
  typename BinaryReturn<PScalar<T1>, PVector<T2,N,C>, OpMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = l.elem() * r.elem(i);
  return d;
}


template<class T1, class T2, int N, template<class,int> class C1, template<class,int> class C2>
inline typename BinaryReturn<PMatrix<T1,N,C1>, PVector<T2,N,C2>, OpMultiply>::Type_t
operator*(const PMatrix<T1,N,C1>& l, const PVector<T2,N,C2>& r)
{
  typename BinaryReturn<PMatrix<T1,N,C1>, PVector<T2,N,C2>, OpMultiply>::Type_t  d;

  for(int i=0; i < N; ++i)
  {
    d.elem(i) = l.elem(i,0) * r.elem(0);
    for(int j=1; j < N; ++j)
      d.elem(i) += l.elem(i,j) * r.elem(j);
  }

  return d;
}



//! PVector = Re(PVector)
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PVector<T,N,C>, FnReal>::Type_t
real(const PVector<T,N,C>& s1)
{
  typename UnaryReturn<PVector<T,N,C>, FnReal>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = real(s1.elem(i));

  return d;
}


//! PVector = Im(PVector)
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PVector<T,N,C>, FnImag>::Type_t
imag(const PVector<T,N,C>& s1)
{
  typename UnaryReturn<PVector<T,N,C>, FnImag>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = imag(s1.elem(i));

  return d;
}


//! PVector<T> = (PVector<T> , PVector<T>)
template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PVector<T1,N,C>, PVector<T2,N,C>, FnCmplx>::Type_t
cmplx(const PVector<T1,N,C>& s1, const PVector<T2,N,C>& s2)
{
  typename BinaryReturn<PVector<T1,N,C>, PVector<T2,N,C>, FnCmplx>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = cmplx(s1.elem(i), s2.elem(i));

  return d;
}



// Functions
//! PVector = i * PVector
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PVector<T,N,C>, FnMultiplyI>::Type_t
multiplyI(const PVector<T,N,C>& s1)
{
  typename UnaryReturn<PVector<T,N,C>, FnMultiplyI>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = multiplyI(s1.elem(i));

  return d;
}

//! PVector = -i * PVector
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PVector<T,N,C>, FnMultiplyMinusI>::Type_t
multiplyMinusI(const PVector<T,N,C>& s1)
{
  typename UnaryReturn<PVector<T,N,C>, FnMultiplyMinusI>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = multiplyMinusI(s1.elem(i));

  return d;
}


//! Extract color vector components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PVector<T,N,C>, FnPeekColorVector>::Type_t
peekColor(const PVector<T,N,C>& l, int row)
{
  typename UnaryReturn<PVector<T,N,C>, FnPeekColorVector>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = peekColor(l.elem(i),row);
  return d;
}

//! Extract color matrix components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PVector<T,N,C>, FnPeekColorMatrix>::Type_t
peekColor(const PVector<T,N,C>& l, int row, int col)
{
  typename UnaryReturn<PVector<T,N,C>, FnPeekColorMatrix>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = peekColor(l.elem(i),row,col);
  return d;
}

//! Extract spin vector components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PVector<T,N,C>, FnPeekSpinVector>::Type_t
peekSpin(const PVector<T,N,C>& l, int row)
{
  typename UnaryReturn<PVector<T,N,C>, FnPeekSpinVector>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = peekSpin(l.elem(i),row);
  return d;
}

//! Extract spin matrix components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PVector<T,N,C>, FnPeekSpinMatrix>::Type_t
peekSpin(const PVector<T,N,C>& l, int row, int col)
{
  typename UnaryReturn<PVector<T,N,C>, FnPeekSpinMatrix>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = peekSpin(l.elem(i),row,col);
  return d;
}


//! Insert color vector components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T1, class T2, int N, template<class,int> class C>
inline PVector<T1,N,C>&
pokeColor(PVector<T1,N,C>& l, const PVector<T2,N,C>& r, int row)
{
  for(int i=0; i < N; ++i)
    l.elem(i) = pokeColor(l.elem(i),r.elem(i),row);
  return l;
}

//! Insert color matrix components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T1, class T2, int N, template<class,int> class C>
inline PVector<T1,N,C>&
pokeColor(PVector<T1,N,C>& l, const PVector<T2,N,C>& r, int row, int col)
{
  for(int i=0; i < N; ++i)
    l.elem(i) = pokeColor(l.elem(i),r.elem(i),row,col);
  return l;
}

//! Insert spin vector components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T1, class T2, int N, template<class,int> class C>
inline PVector<T1,N,C>&
pokeSpin(PVector<T1,N,C>& l, const PVector<T2,N,C>& r, int row)
{
  for(int i=0; i < N; ++i)
    l.elem(i) = pokeSpin(l.elem(i),r.elem(i),row);
  return l;
}

//! Insert spin matrix components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T1, class T2, int N, template<class,int> class C>
inline PVector<T1,N,C>&
pokeSpin(PVector<T1,N,C>& l, const PVector<T2,N,C>& r, int row, int col)
{
  for(int i=0; i < N; ++i)
    l.elem(i) = pokeSpin(l.elem(i),r.elem(i),row,col);
  return l;
}



//! dest = 0
template<class T, int N, template<class,int> class C> 
void zero_rep(PVector<T,N,C>& dest) 
{
  for(int i=0; i < N; ++i)
    zero_rep(dest.elem(i));
}

//! dest = (mask) ? s1 : dest
template<class T, class T1, int N, template<class,int> class C> 
void copymask(PVector<T,N,C>& d, const PScalar<T1>& mask, const PVector<T,N,C>& s1) 
{
  for(int i=0; i < N; ++i)
    copymask(d.elem(i),mask.elem(),s1.elem(i));
}


//! dest  = random  
template<class T, int N, template<class,int> class C, class T1, class T2>
inline void
fill_random(PVector<T,N,C>& d, T1& seed, T2& skewed_seed, const T1& seed_mult)
{
  // Loop over rows the slowest
  for(int i=0; i < N; ++i)
    fill_random(d.elem(i), seed, skewed_seed, seed_mult);
}


//! dest  = gaussian
template<class T, int N, template<class,int> class C>
void
fill_gaussian(PVector<T,N,C>& d, PVector<T,N,C>& r1, PVector<T,N,C>& r2)
{
  for(int i=0; i < N; ++i)
    fill_gaussian(d.elem(i), r1.elem(i), r2.elem(i));
}


#if 0
// Global sum over site indices only
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PVector<T,N,C>, FnSum > {
  typedef C<typename UnaryReturn<T, FnSum>::Type_t, N>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PVector<T,N,C>, FnSum>::Type_t
sum(const PVector<T,N,C>& s1)
{
  typename UnaryReturn<PVector<T,N,C>, FnSum>::Type_t  d;

  for(int i=0; i < N; ++i)
    d.elem(i) = sum(s1.elem(i));

  return d;
}
#endif


// Innerproduct (norm-seq) global sum = sum(tr(conj(s1)*s1))
template<class T, int N, template<class,int> class C>
struct UnaryReturn<PVector<T,N,C>, FnNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T, int N, template<class,int> class C>
struct UnaryReturn<PVector<T,N,C>, FnLocalNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T, int N, template<class,int> class C>
inline typename UnaryReturn<PVector<T,N,C>, FnLocalNorm2>::Type_t
localNorm2(const PVector<T,N,C>& s1)
{
  typename UnaryReturn<PVector<T,N,C>, FnLocalNorm2>::Type_t  d;

  d.elem() = localNorm2(s1.elem(0));
  for(int i=1; i < N; ++i)
    d.elem() += localNorm2(s1.elem(i));

  return d;
}


//! PScalar<T> = Innerproduct(Conj(PVector<T1>)*PVector<T1>)
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PVector<T1,N,C>, PVector<T2,N,C>, FnInnerproduct > {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerproduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PVector<T1,N,C>, PVector<T2,N,C>, FnLocalInnerproduct > {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerproduct>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PVector<T1,N,C>, PVector<T2,N,C>, FnLocalInnerproduct>::Type_t
localInnerproduct(const PVector<T1,N,C>& s1, const PVector<T1,N,C>& s2)
{
  typename BinaryReturn<PVector<T1,N,C>, PVector<T2,N,C>, FnLocalInnerproduct>::Type_t  d;

  d.elem() = localInnerproduct(s1.elem(0), s2.elem(0));
  for(int i=1; i < N; ++i)
    d.elem() += localInnerproduct(s1.elem(i), s2.elem(i));

  return d;
}


//! PScalar<T> = InnerproductReal(Conj(PVector<T1>)*PVector<T1>)
/*!
 * return  realpart of Innerproduct(Conj(s1)*s2)
 */
template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PVector<T1,N,C>, PVector<T2,N,C>, FnInnerproductReal > {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerproductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
struct BinaryReturn<PVector<T1,N,C>, PVector<T2,N,C>, FnLocalInnerproductReal > {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerproductReal>::Type_t>  Type_t;
};

template<class T1, class T2, int N, template<class,int> class C>
inline typename BinaryReturn<PVector<T1,N,C>, PVector<T2,N,C>, FnLocalInnerproductReal>::Type_t
localInnerproductReal(const PVector<T1,N,C>& s1, const PVector<T1,N,C>& s2)
{
  typename BinaryReturn<PVector<T1,N,C>, PVector<T2,N,C>, FnLocalInnerproductReal>::Type_t  d;

  d.elem() = localInnerproductReal(s1.elem(0), s2.elem(0));
  for(int i=1; i < N; ++i)
    d.elem() += localInnerproductReal(s1.elem(i), s2.elem(i));

  return d;
}


//! PVector<T> = where(PScalar, PVector, PVector)
/*!
 * Where is the ? operation
 * returns  (a) ? b : c;
 */
template<class T1, class T2, class T3, int N, template<class,int> class C>
inline typename BinaryReturn<PVector<T2,N,C>, PVector<T3,N,C>, FnWhere>::Type_t
where(const PScalar<T1>& a, const PVector<T2,N,C>& b, const PVector<T3,N,C>& c)
{
  typename BinaryReturn<PVector<T2,N,C>, PVector<T3,N,C>, FnWhere>::Type_t  d;

  // Not optimal - want to have where outside assignment
  for(int i=0; i < N; ++i)
    d.elem(i) = where(a.elem(), b.elem(i), c.elem(i));

  return d;
}

/*! @} */  // end of group primvector

QDP_END_NAMESPACE();

