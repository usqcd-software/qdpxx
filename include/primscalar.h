// -*- C++ -*-
// $Id: primscalar.h,v 1.15 2002-11-23 02:23:24 edwards Exp $

/*! \file
 * \brief Primitive Scalar
 */

QDP_BEGIN_NAMESPACE(QDP);

//-------------------------------------------------------------------------------------
/*! \addtogroup primscalar Scalar primitive
 * \ingroup fiber
 *
 * Primitive Scalar is a placeholder for no primitive structure
 *
 * @{
 */

//! Primitive Scalar
/*! Placeholder for no primitive structure */
template<class T> class PScalar
{
public:
  PScalar() {}
  ~PScalar() {}

  //---------------------------------------------------------
  //! construct dest = const
  PScalar(const typename WordType<T>::Type_t rhs) : F(rhs) {}


  //! construct dest = const
  template<class T1>
  PScalar(const PScalar<T1> rhs) : F(rhs) {}

  //---------------------------------------------------------
#if 0
  //! dest = const
  /*! Fill with an integer constant. Will be promoted to underlying word type */
  inline
  PScalar& operator=(const typename WordType<T>::Type_t& rhs)
    {
      elem() = rhs;
      return *this;
    }
#endif

  //! PScalar = PScalar
  /*! Set equal to another PScalar */
  template<class T1>
  inline
  PScalar& operator=(const PScalar<T1>& rhs) 
    {
      elem() = rhs.elem();
      return *this;
    }

  //! PScalar += PScalar
  template<class T1>
  inline
  PScalar& operator+=(const PScalar<T1>& rhs) 
    {
      elem() += rhs.elem();
      return *this;
    }

  //! PScalar -= PScalar
  template<class T1>
  inline
  PScalar& operator-=(const PScalar<T1>& rhs) 
    {
      elem() -= rhs.elem();
      return *this;
    }

  //! PScalar *= PScalar
  template<class T1>
  inline
  PScalar& operator*=(const PScalar<T1>& rhs) 
    {
      elem() *= rhs.elem();
      return *this;
    }

  //! PScalar /= PScalar
  template<class T1>
  inline
  PScalar& operator/=(const PScalar<T1>& rhs) 
    {
      elem() /= rhs.elem();
      return *this;
    }

  //! PScalar %= PScalar
  template<class T1>
  inline
  PScalar& operator%=(const PScalar<T1>& rhs) 
    {
      elem() %= rhs.elem();
      return *this;
    }

  //! PScalar |= PScalar
  template<class T1>
  inline
  PScalar& operator|=(const PScalar<T1>& rhs) 
    {
      elem() |= rhs.elem();
      return *this;
    }

  //! PScalar &= PScalar
  template<class T1>
  inline
  PScalar& operator&=(const PScalar<T1>& rhs) 
    {
      elem() &= rhs.elem();
      return *this;
    }

  //! PScalar ^= PScalar
  template<class T1>
  inline
  PScalar& operator^=(const PScalar<T1>& rhs) 
    {
      elem() ^= rhs.elem();
      return *this;
    }

  //! PScalar <<= PScalar
  template<class T1>
  inline
  PScalar& operator<<=(const PScalar<T1>& rhs) 
    {
      elem() <<= rhs.elem();
      return *this;
    }

  //! PScalar >>= PScalar
  template<class T1>
  inline
  PScalar& operator>>=(const PScalar<T1>& rhs) 
    {
      elem() >>= rhs.elem();
      return *this;
    }

  //! Deep copies here
  PScalar(const PScalar& a): F(a.F) {/* fprintf(stderr,"copy PScalar\n"); */}

public:
  T& elem() {return F;}
  const T& elem() const {return F;}

private:
  T F;
};


// Output
//! Ascii output
template<class T>
inline
ostream& operator<<(ostream& s, const PScalar<T>& d)
{
  return s << d.elem();
}

//! Namelist output
template<class T>
inline
NmlWriter& operator<<(NmlWriter& nml, const PScalar<T>& d)
{
  return nml << d.elem();
}

/*! @} */  // end of group primscalar


//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

// Underlying word type
template<class T>
struct WordType<PScalar<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};

// Internally used scalars
template<class T>
struct InternalScalar<PScalar<T> > {
  typedef PScalar<typename InternalScalar<T>::Type_t>  Type_t;
};

// Internally used real scalars
template<class T>
struct RealScalar<PScalar<T> > {
  typedef PScalar<typename RealScalar<T>::Type_t>  Type_t;
};


//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

// Default unary(PScalar) -> PScalar
template<class T1, class Op>
struct UnaryReturn<PScalar<T1>, Op> {
  typedef PScalar<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};

// Default binary(PScalar,PScalar) -> PScalar
template<class T1, class T2, class Op>
struct BinaryReturn<PScalar<T1>, PScalar<T2>, Op> {
  typedef PScalar<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


#if 0
template<class T1, class T2>
struct UnaryReturn<PScalar<T2>, OpCast<T1> > {
  typedef PScalar<typename UnaryReturn<T, OpCast>::Type_t>  Type_t;
//  typedef T1 Type_t;
};
#endif

// Assignment is different
template<class T1, class T2 >
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpAssign > {
  typedef PScalar<T1> &Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpAddAssign > {
  typedef PScalar<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpSubtractAssign > {
  typedef PScalar<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpMultiplyAssign > {
  typedef PScalar<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpDivideAssign > {
  typedef PScalar<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpModAssign > {
  typedef PScalar<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpBitwiseOrAssign > {
  typedef PScalar<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpBitwiseAndAssign > {
  typedef PScalar<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpBitwiseXorAssign > {
  typedef PScalar<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpLeftShiftAssign > {
  typedef PScalar<T1> &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpRightShiftAssign > {
  typedef PScalar<T1> &Type_t;
};
 



//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------

/*! \addtogroup primscalar */
/*! @{ */

// Primitive Scalars
template<class T>
struct UnaryReturn<PScalar<T>, OpNot > {
  typedef PScalar<typename UnaryReturn<T, OpNot>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<PScalar<T1>, OpNot>::Type_t
operator!(const PScalar<T1>& l)
{
  typename UnaryReturn<PScalar<T1>, OpNot>::Type_t  d;

  d.elem() = ! l.elem();
  return d;
}


template<class T1>
inline typename UnaryReturn<PScalar<T1>, OpUnaryPlus>::Type_t
operator+(const PScalar<T1>& l)
{
  typename UnaryReturn<PScalar<T1>, OpUnaryPlus>::Type_t  d;

  d.elem() = +l.elem();
  return d;
}


template<class T1>
inline typename UnaryReturn<PScalar<T1>, OpUnaryMinus>::Type_t
operator-(const PScalar<T1>& l)
{
  typename UnaryReturn<PScalar<T1>, OpUnaryMinus>::Type_t  d;

  d.elem() = -l.elem();
  return d;
}


template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpAdd>::Type_t
operator+(const PScalar<T1>& l, const PScalar<T2>& r)
{
  typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpAdd>::Type_t  d;

  d.elem() = l.elem() + r.elem();
  return d;
}


template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpSubtract>::Type_t
operator-(const PScalar<T1>& l, const PScalar<T2>& r)
{
  typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpSubtract>::Type_t  d;

  d.elem() = l.elem() - r.elem();
  return d;
}


template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpMultiply>::Type_t
operator*(const PScalar<T1>& l, const PScalar<T2>& r)
{
  typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpMultiply>::Type_t  d;

  d.elem() = l.elem() * r.elem();
  return d;
}


template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpDivide>::Type_t
operator/(const PScalar<T1>& l, const PScalar<T2>& r)
{
  typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpDivide>::Type_t  d;

  d.elem() = l.elem() / r.elem();
  return d;
}



template<class T1, class T2 >
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpLeftShift > {
  typedef PScalar<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};
 

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpLeftShift>::Type_t
operator<<(const PScalar<T1>& l, const PScalar<T2>& r)
{
  typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpLeftShift>::Type_t  d;

  d.elem() = l.elem() << r.elem();
  return d;
}

template<class T1, class T2 >
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpRightShift > {
  typedef PScalar<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};
 

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpRightShift>::Type_t
operator>>(const PScalar<T1>& l, const PScalar<T2>& r)
{
  typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpRightShift>::Type_t  d;

  d.elem() = l.elem() >> r.elem();
  return d;
}


template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpMod>::Type_t
operator%(const PScalar<T1>& l, const PScalar<T2>& r)
{
  typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpMod>::Type_t  d;

  d.elem() = l.elem() % r.elem();
  return d;
}

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpBitwiseXor>::Type_t
operator^(const PScalar<T1>& l, const PScalar<T2>& r)
{
  typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpBitwiseXor>::Type_t  d;

  d.elem() = l.elem() ^ r.elem();
  return d;
}

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpBitwiseAnd>::Type_t
operator&(const PScalar<T1>& l, const PScalar<T2>& r)
{
  typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpBitwiseAnd>::Type_t  d;

  d.elem() = l.elem() & r.elem();
  return d;
}

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpBitwiseOr>::Type_t
operator|(const PScalar<T1>& l, const PScalar<T2>& r)
{
  typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpBitwiseOr>::Type_t  d;

  d.elem() = l.elem() | r.elem();
  return d;
}


// Comparisons
template<class T1, class T2 >
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpLT > {
  typedef PScalar<typename BinaryReturn<T1, T2, OpLT>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpLT>::Type_t
operator<(const PScalar<T1>& l, const PScalar<T2>& r)
{
  typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpLT>::Type_t  d;

  d.elem() = l.elem() < r.elem();
  return d;
}


template<class T1, class T2 >
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpLE > {
  typedef PScalar<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpLE>::Type_t
operator<=(const PScalar<T1>& l, const PScalar<T2>& r)
{
  typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpLE>::Type_t  d;

  d.elem() = l.elem() <= r.elem();
  return d;
}


template<class T1, class T2 >
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpGT > {
  typedef PScalar<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpGT>::Type_t
operator>(const PScalar<T1>& l, const PScalar<T2>& r)
{
  typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpGT>::Type_t  d;

  d.elem() = l.elem() > r.elem();
  return d;
}


template<class T1, class T2 >
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpGE > {
  typedef PScalar<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpGE>::Type_t
operator>=(const PScalar<T1>& l, const PScalar<T2>& r)
{
  typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpGE>::Type_t  d;

  d.elem() = l.elem() >= r.elem();
  return d;
}


template<class T1, class T2 >
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpEQ > {
  typedef PScalar<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpEQ>::Type_t
operator==(const PScalar<T1>& l, const PScalar<T2>& r)
{
  typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpEQ>::Type_t  d;

  d.elem() = l.elem() == r.elem();
  return d;
}


template<class T1, class T2 >
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpNE > {
  typedef PScalar<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpNE>::Type_t
operator!=(const PScalar<T1>& l, const PScalar<T2>& r)
{
  typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpNE>::Type_t  d;

  d.elem() = l.elem() != r.elem();
  return d;
}


//-----------------------------------------------------------------------------
// Functions

// Conjugate
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnConj>::Type_t
conj(const PScalar<T1>& s1)
{
  typename UnaryReturn<PScalar<T1>, FnConj>::Type_t  d;

  d.elem() = conj(s1.elem());
  return d;
}


// Transpose
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnTranspose>::Type_t
transpose(const PScalar<T1>& s1)
{
  typename UnaryReturn<PScalar<T1>, FnTranspose>::Type_t  d;

  d.elem() = transpose(s1.elem());
  return d;
}


// TRACE
// trace = Trace(source1)
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnTrace>::Type_t
trace(const PScalar<T1>& s1)
{
  typename UnaryReturn<PScalar<T1>, FnTrace>::Type_t  d;

  d.elem() = trace(s1.elem());
  return d;
}


// trace = Re(Trace(source1))
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnRealTrace>::Type_t
realTrace(const PScalar<T1>& s1)
{
  typename UnaryReturn<PScalar<T1>, FnRealTrace>::Type_t  d;

  d.elem() = realTrace(s1.elem());
  return d;
}


// trace = Im(Trace(source1))
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnImagTrace>::Type_t
imagTrace(const PScalar<T1>& s1)
{
  typename UnaryReturn<PScalar<T1>, FnImagTrace>::Type_t  d;

  d.elem() = imagTrace(s1.elem());
  return d;
}


// trace = colorTrace(source1)
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnTraceColor>::Type_t
traceColor(const PScalar<T1>& s1)
{
  typename UnaryReturn<PScalar<T1>, FnTraceColor>::Type_t  d;

  d.elem() = traceColor(s1.elem());
  return d;
}


// trace = traceSpin(source1)
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnTraceSpin>::Type_t
traceSpin(const PScalar<T1>& s1)
{
  typename UnaryReturn<PScalar<T1>, FnTraceSpin>::Type_t  d;

  d.elem() = traceSpin(s1.elem());
  return d;
}


// PScalar = Re(PScalar)
template<class T>
inline typename UnaryReturn<PScalar<T>, FnReal>::Type_t
real(const PScalar<T>& s1)
{
  typename UnaryReturn<PScalar<T>, FnReal>::Type_t  d;

  d.elem() = real(s1.elem());
  return d;
}


// PScalar = Im(PScalar)
template<class T>
inline typename UnaryReturn<PScalar<T>, FnImag>::Type_t
imag(const PScalar<T>& s1)
{
  typename UnaryReturn<PScalar<T>, FnImag>::Type_t  d;

  d.elem() = imag(s1.elem());
  return d;
}


// ArcCos
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnArcCos>::Type_t
acos(const PScalar<T1>& s1)
{
  typename UnaryReturn<PScalar<T1>, FnArcCos>::Type_t  d;

  d.elem() = acos(s1.elem());
  return d;
}

// ArcSin
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnArcSin>::Type_t
asin(const PScalar<T1>& s1)
{
  typename UnaryReturn<PScalar<T1>, FnArcSin>::Type_t  d;

  d.elem() = asin(s1.elem());
  return d;
}

// ArcTan
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnArcTan>::Type_t
atan(const PScalar<T1>& s1)
{
  typename UnaryReturn<PScalar<T1>, FnArcTan>::Type_t  d;

  d.elem() = atan(s1.elem());
  return d;
}

// Cos
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnCos>::Type_t
cos(const PScalar<T1>& s1)
{
  typename UnaryReturn<PScalar<T1>, FnCos>::Type_t  d;

  d.elem() = cos(s1.elem());
  return d;
}

// Exp
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnExp>::Type_t
exp(const PScalar<T1>& s1)
{
  typename UnaryReturn<PScalar<T1>, FnExp>::Type_t  d;

  d.elem() = exp(s1.elem());
  return d;
}

// Fabs
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnFabs>::Type_t
fabs(const PScalar<T1>& s1)
{
  typename UnaryReturn<PScalar<T1>, FnFabs>::Type_t  d;

  d.elem() = fabs(s1.elem());
  return d;
}

// Log
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnLog>::Type_t
log(const PScalar<T1>& s1)
{
  typename UnaryReturn<PScalar<T1>, FnLog>::Type_t  d;

  d.elem() = log(s1.elem());
  return d;
}

// Sin
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnSin>::Type_t
sin(const PScalar<T1>& s1)
{
  typename UnaryReturn<PScalar<T1>, FnSin>::Type_t  d;

  d.elem() = sin(s1.elem());
  return d;
}

// Sqrt
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnSqrt>::Type_t
sqrt(const PScalar<T1>& s1)
{
  typename UnaryReturn<PScalar<T1>, FnSqrt>::Type_t  d;

  d.elem() = sqrt(s1.elem());
  return d;
}

// Tan
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnTan>::Type_t
tan(const PScalar<T1>& s1)
{
  typename UnaryReturn<PScalar<T1>, FnTan>::Type_t  d;

  d.elem() = tan(s1.elem());
  return d;
}



//! PScalar<T> = pow(PScalar<T> , PScalar<T>)
template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, FnPow>::Type_t
pow(const PScalar<T1>& s1, const PScalar<T2>& s2)
{
  typename BinaryReturn<PScalar<T1>, PScalar<T2>, FnPow>::Type_t  d;

  d.elem() = pow(s1.elem(), s2.elem());
  return d;
}

//! PScalar<T> = atan2(PScalar<T> , PScalar<T>)
template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, FnArcTan2>::Type_t
atan2(const PScalar<T1>& s1, const PScalar<T2>& s2)
{
  typename BinaryReturn<PScalar<T1>, PScalar<T2>, FnArcTan2>::Type_t  d;

  d.elem() = atan2(s1.elem(), s2.elem());
  return d;
}


//! PScalar<T> = (PScalar<T> , PScalar<T>)
template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, FnCmplx>::Type_t
cmplx(const PScalar<T1>& s1, const PScalar<T2>& s2)
{
  typename BinaryReturn<PScalar<T1>, PScalar<T2>, FnCmplx>::Type_t  d;

  d.elem() = cmplx(s1.elem(), s2.elem());
  return d;
}



// Global Functions
// PScalar = i * PScalar
template<class T>
inline typename UnaryReturn<PScalar<T>, FnMultiplyI>::Type_t
multiplyI(const PScalar<T>& s1)
{
  typename UnaryReturn<PScalar<T>, FnMultiplyI>::Type_t  d;

  d.elem() = multiplyI(s1.elem());
  return d;
}

// PScalar = -i * PScalar
template<class T>
inline typename UnaryReturn<PScalar<T>, FnMultiplyMinusI>::Type_t
multiplyMinusI(const PScalar<T>& s1)
{
  typename UnaryReturn<PScalar<T>, FnMultiplyMinusI>::Type_t  d;

  d.elem() = multiplyMinusI(s1.elem());
  return d;
}


//! dest [float type] = source [seed type]
template<class T>
inline typename UnaryReturn<PScalar<T>, FnSeedToFloat>::Type_t
seedToFloat(const PScalar<T>& s1)
{
  typename UnaryReturn<PScalar<T>, FnSeedToFloat>::Type_t  d;

  d.elem() = seedToFloat(s1.elem());
  return d;
}


//! Extract color vector components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T>
inline typename UnaryReturn<PScalar<T>, FnPeekColorVector>::Type_t
peekColor(const PScalar<T>& l, int row)
{
  typename UnaryReturn<PScalar<T>, FnPeekColorVector>::Type_t  d;

  d.elem() = peekColor(l.elem(),row);
  return d;
}

//! Extract color matrix components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T>
inline typename UnaryReturn<PScalar<T>, FnPeekColorMatrix>::Type_t
peekColor(const PScalar<T>& l, int row, int col)
{
  typename UnaryReturn<PScalar<T>, FnPeekColorMatrix>::Type_t  d;

  d.elem() = peekColor(l.elem(),row,col);
  return d;
}

//! Extract spin vector components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T>
inline typename UnaryReturn<PScalar<T>, FnPeekSpinVector>::Type_t
peekSpin(const PScalar<T>& l, int row)
{
  typename UnaryReturn<PScalar<T>, FnPeekSpinVector>::Type_t  d;

  d.elem() = peekSpin(l.elem(),row);
  return d;
}

//! Extract spin matrix components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T>
inline typename UnaryReturn<PScalar<T>, FnPeekSpinMatrix>::Type_t
peekSpin(const PScalar<T>& l, int row, int col)
{
  typename UnaryReturn<PScalar<T>, FnPeekSpinMatrix>::Type_t  d;

  d.elem() = peekSpin(l.elem(),row,col);
  return d;
}


//! Insert color vector components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T1, class T2>
inline PScalar<T1>&
pokeColor(PScalar<T1>& l, const PScalar<T2>& r, int row)
{
  pokeColor(l.elem(),r.elem(),row);
  return l;
}

//! Insert color matrix components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T1, class T2>
inline PScalar<T1>&
pokeColor(PScalar<T1>& l, const PScalar<T2>& r, int row, int col)
{
  pokeColor(l.elem(),r.elem(),row,col);
  return l;
}

//! Insert spin vector components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T1, class T2>
inline PScalar<T1>&
pokeSpin(PScalar<T1>& l, const PScalar<T2>& r, int row)
{
  pokeSpin(l.elem(),r.elem(),row);
  return l;
}

//! Insert spin matrix components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T1, class T2>
inline PScalar<T1>&
pokeSpin(PScalar<T1>& l, const PScalar<T2>& r, int row, int col)
{
  pokeSpin(l.elem(),r.elem(),row,col);
  return l;
}



//! dest = (mask) ? s1 : dest
template<class T, class T1> 
void copymask(PScalar<T>& d, const PScalar<T1>& mask, const PScalar<T>& s1) 
{
  copymask(d.elem(),mask.elem(),s1.elem());
}

//! dest  = random  
template<class T, class T1, class T2>
inline void
fill_random(PScalar<T>& d, T1& seed, T2& skewed_seed, const T1& seed_mult)
{
  fill_random(d.elem(), seed, skewed_seed, seed_mult);
}


//! dest  = gaussian  
template<class T>
inline void
fill_gaussian(PScalar<T>& d, PScalar<T>& r1, PScalar<T>& r2)
{
  fill_gaussian(d.elem(), r1.elem(), r2.elem());
}


#if 0
// Global sum over site indices only
template<class T>
struct UnaryReturn<PScalar<T>, FnSum > {
  typedef PScalar<typename UnaryReturn<T, FnSum>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<PScalar<T>, FnSum>::Type_t
sum(const PScalar<T>& s1)
{
  typename UnaryReturn<PScalar<T>, FnSum>::Type_t  d;

  d.elem() = sum(s1.elem());
  return d;
}
#endif


// Innerproduct (norm-seq) global sum = sum(tr(conj(s1)*s1))
template<class T>
struct UnaryReturn<PScalar<T>, FnNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<PScalar<T>, FnLocalNorm2 > {
  typedef PScalar<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<PScalar<T>, FnLocalNorm2>::Type_t
localNorm2(const PScalar<T>& s1)
{
  typename UnaryReturn<PScalar<T>, FnLocalNorm2>::Type_t  d;

  d.elem() = localNorm2(s1.elem());
  return d;
}



//! PScalar<T> = Innerproduct(Conj(PScalar<T1>)*PScalar<T2>)
template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PScalar<T2>, FnInnerproduct > {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerproduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PScalar<T2>, FnLocalInnerproduct > {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerproduct>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, FnLocalInnerproduct>::Type_t
localInnerproduct(const PScalar<T1>& s1, const PScalar<T2>& s2)
{
  typename BinaryReturn<PScalar<T1>, PScalar<T2>, FnLocalInnerproduct>::Type_t  d;

  d.elem() = localInnerproduct(s1.elem(), s2.elem());
  return d;
}


//! PScalar<T> = Innerproduct_real(Conj(PMatrix<T1>)*PMatrix<T1>)
template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PScalar<T2>, FnInnerproductReal > {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerproductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PScalar<T2>, FnLocalInnerproductReal > {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerproductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, FnLocalInnerproductReal>::Type_t
localInnerproductReal(const PScalar<T1>& s1, const PScalar<T2>& s2)
{
  typename BinaryReturn<PScalar<T1>, PScalar<T2>, FnLocalInnerproductReal>::Type_t  d;

  d.elem() = localInnerproductReal(s1.elem(), s2.elem());
  return d;
}


//! PScalar<T> = where(PScalar, PScalar, PScalar)
/*!
 * Where is the ? operation
 * returns  (a) ? b : c;
 */
template<class T1, class T2, class T3>
inline typename BinaryReturn<PScalar<T2>, PScalar<T3>, FnWhere>::Type_t
where(const PScalar<T1>& a, const PScalar<T2>& b, const PScalar<T3>& c)
{
  typename BinaryReturn<PScalar<T2>, PScalar<T3>, FnWhere>::Type_t  d;

  d.elem() = where(a.elem(), b.elem(), c.elem());
  return d;
}


//-----------------------------------------------------------------------------
//! conversion routines
template<class T> 
inline int 
toInt(const PScalar<T>& s) 
{
  return toInt(s.elem());
}



//-----------------------------------------------------------------------------
// Other operations
//! dest = 0
template<class T> 
inline void 
zero_rep(PScalar<T>& dest) 
{
  zero_rep(dest.elem());
}

//! dest [some type] = source [some type]
template<class T, class T1>
inline void 
cast_rep(T& d, const PScalar<T1>& s1)
{
  cast_rep(d, s1.elem());
}

//! dest [some type] = source [some type]
template<class T, class T1>
inline void 
cast_rep(PScalar<T>& d, const PScalar<T1>& s1)
{
  cast_rep(d.elem(), s1.elem());
}

/*! @} */  // end of group primscalar

QDP_END_NAMESPACE();

