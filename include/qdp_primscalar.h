// -*- C++ -*-
// $Id: qdp_primscalar.h,v 1.3 2003-08-01 18:41:11 edwards Exp $

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
  PScalar(const typename WordType<T>::Type_t& rhs) : F(rhs) {}

  //! construct dest = rhs
  template<class T1>
  PScalar(const PScalar<T1>& rhs) : F(rhs.elem()) {}

  //! construct dest = rhs
  template<class T1>
  PScalar(const T1& rhs) : F(rhs) {}

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


// Input
//! Ascii input
template<class T>
inline
istream& operator>>(istream& s, PScalar<T>& d)
{
  s >> d.elem();
  return s;
}

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

//! XML output
template<class T>
inline
XMLWriter& operator<<(XMLWriter& xml, const PScalar<T>& d)
{
  return xml << d.elem();
}

//! XML input
template<class T>
inline
void read(XMLReader& xml, const string& path, PScalar<T>& d)
{
  read(xml, path, d.elem());
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
  return ! l.elem();
}


template<class T1>
inline typename UnaryReturn<PScalar<T1>, OpUnaryPlus>::Type_t
operator+(const PScalar<T1>& l)
{
  return +l.elem();
}


template<class T1>
inline typename UnaryReturn<PScalar<T1>, OpUnaryMinus>::Type_t
operator-(const PScalar<T1>& l)
{
  return -l.elem();
}


template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpAdd>::Type_t
operator+(const PScalar<T1>& l, const PScalar<T2>& r)
{
  return l.elem() + r.elem();
}


template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpSubtract>::Type_t
operator-(const PScalar<T1>& l, const PScalar<T2>& r)
{
  return l.elem() - r.elem();
}


template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpMultiply>::Type_t
operator*(const PScalar<T1>& l, const PScalar<T2>& r)
{
  return  l.elem() * r.elem();
}


template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpDivide>::Type_t
operator/(const PScalar<T1>& l, const PScalar<T2>& r)
{
  return l.elem() / r.elem();
}



template<class T1, class T2 >
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpLeftShift > {
  typedef PScalar<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};
 

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpLeftShift>::Type_t
operator<<(const PScalar<T1>& l, const PScalar<T2>& r)
{
  return l.elem() << r.elem();
}

template<class T1, class T2 >
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpRightShift > {
  typedef PScalar<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};
 

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpRightShift>::Type_t
operator>>(const PScalar<T1>& l, const PScalar<T2>& r)
{
  return l.elem() >> r.elem();
}


template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpMod>::Type_t
operator%(const PScalar<T1>& l, const PScalar<T2>& r)
{
  return l.elem() % r.elem();
}

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpBitwiseXor>::Type_t
operator^(const PScalar<T1>& l, const PScalar<T2>& r)
{
  return l.elem() ^ r.elem();
}

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpBitwiseAnd>::Type_t
operator&(const PScalar<T1>& l, const PScalar<T2>& r)
{
  return l.elem() & r.elem();
}

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpBitwiseOr>::Type_t
operator|(const PScalar<T1>& l, const PScalar<T2>& r)
{
  return l.elem() | r.elem();
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
  return l.elem() < r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpLE > {
  typedef PScalar<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpLE>::Type_t
operator<=(const PScalar<T1>& l, const PScalar<T2>& r)
{
  return l.elem() <= r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpGT > {
  typedef PScalar<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpGT>::Type_t
operator>(const PScalar<T1>& l, const PScalar<T2>& r)
{
  return l.elem() > r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpGE > {
  typedef PScalar<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpGE>::Type_t
operator>=(const PScalar<T1>& l, const PScalar<T2>& r)
{
  return l.elem() >= r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpEQ > {
  typedef PScalar<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpEQ>::Type_t
operator==(const PScalar<T1>& l, const PScalar<T2>& r)
{
  return l.elem() == r.elem();
}


template<class T1, class T2 >
struct BinaryReturn<PScalar<T1>, PScalar<T2>, OpNE > {
  typedef PScalar<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, OpNE>::Type_t
operator!=(const PScalar<T1>& l, const PScalar<T2>& r)
{
  return l.elem() != r.elem();
}


//-----------------------------------------------------------------------------
// Functions

// Adjoint
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnAdjoint>::Type_t
adj(const PScalar<T1>& s1)
{
  return adj(s1.elem());
}


// Conjugate
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnConjugate>::Type_t
conj(const PScalar<T1>& s1)
{
  return conj(s1.elem());
}


// Transpose
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnTranspose>::Type_t
transpose(const PScalar<T1>& s1)
{
  return transpose(s1.elem());
}


// TRACE
// trace = Trace(source1)
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnTrace>::Type_t
trace(const PScalar<T1>& s1)
{
  return trace(s1.elem());
}


// trace = Re(Trace(source1))
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnRealTrace>::Type_t
realTrace(const PScalar<T1>& s1)
{
  return realTrace(s1.elem());
}


// trace = Im(Trace(source1))
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnImagTrace>::Type_t
imagTrace(const PScalar<T1>& s1)
{
  return imagTrace(s1.elem());
}


// trace = colorTrace(source1)
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnTraceColor>::Type_t
traceColor(const PScalar<T1>& s1)
{
  return traceColor(s1.elem());
}


// trace = traceSpin(source1)
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnTraceSpin>::Type_t
traceSpin(const PScalar<T1>& s1)
{
  return traceSpin(s1.elem());
}


// PScalar = Re(PScalar)
template<class T>
inline typename UnaryReturn<PScalar<T>, FnReal>::Type_t
real(const PScalar<T>& s1)
{
  return real(s1.elem());
}


// PScalar = Im(PScalar)
template<class T>
inline typename UnaryReturn<PScalar<T>, FnImag>::Type_t
imag(const PScalar<T>& s1)
{
  return imag(s1.elem());
}


// ArcCos
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnArcCos>::Type_t
acos(const PScalar<T1>& s1)
{
  return acos(s1.elem());
}

// ArcSin
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnArcSin>::Type_t
asin(const PScalar<T1>& s1)
{
  return asin(s1.elem());
}

// ArcTan
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnArcTan>::Type_t
atan(const PScalar<T1>& s1)
{
  return atan(s1.elem());
}

// Cos
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnCos>::Type_t
cos(const PScalar<T1>& s1)
{
  return cos(s1.elem());
}

// Exp
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnExp>::Type_t
exp(const PScalar<T1>& s1)
{
  return exp(s1.elem());
}

// Fabs
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnFabs>::Type_t
fabs(const PScalar<T1>& s1)
{
  return fabs(s1.elem());
}

// Log
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnLog>::Type_t
log(const PScalar<T1>& s1)
{
  return log(s1.elem());
}

// Sin
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnSin>::Type_t
sin(const PScalar<T1>& s1)
{
  return sin(s1.elem());
}

// Sqrt
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnSqrt>::Type_t
sqrt(const PScalar<T1>& s1)
{
  return sqrt(s1.elem());
}

// Tan
template<class T1>
inline typename UnaryReturn<PScalar<T1>, FnTan>::Type_t
tan(const PScalar<T1>& s1)
{
  return tan(s1.elem());
}



//! PScalar<T> = pow(PScalar<T> , PScalar<T>)
template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, FnPow>::Type_t
pow(const PScalar<T1>& s1, const PScalar<T2>& s2)
{
  return pow(s1.elem(), s2.elem());
}

//! PScalar<T> = atan2(PScalar<T> , PScalar<T>)
template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, FnArcTan2>::Type_t
atan2(const PScalar<T1>& s1, const PScalar<T2>& s2)
{
  return atan2(s1.elem(), s2.elem());
}


//! PScalar<T> = (PScalar<T> , PScalar<T>)
template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, FnCmplx>::Type_t
cmplx(const PScalar<T1>& s1, const PScalar<T2>& s2)
{
  return cmplx(s1.elem(), s2.elem());
}



// Global Functions
// PScalar = i * PScalar
template<class T>
inline typename UnaryReturn<PScalar<T>, FnTimesI>::Type_t
timesI(const PScalar<T>& s1)
{
  return timesI(s1.elem());
}

// PScalar = -i * PScalar
template<class T>
inline typename UnaryReturn<PScalar<T>, FnTimesMinusI>::Type_t
timesMinusI(const PScalar<T>& s1)
{
  return timesMinusI(s1.elem());
}


//! dest [float type] = source [seed type]
template<class T>
inline typename UnaryReturn<PScalar<T>, FnSeedToFloat>::Type_t
seedToFloat(const PScalar<T>& s1)
{
  return seedToFloat(s1.elem());
}


//! Extract color vector components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T>
inline typename UnaryReturn<PScalar<T>, FnPeekColorVector>::Type_t
peekColor(const PScalar<T>& l, int row)
{
  return peekColor(l.elem(),row);
}

//! Extract color matrix components 
/*! Generically, this is an identity operation. Defined differently under color */
template<class T>
inline typename UnaryReturn<PScalar<T>, FnPeekColorMatrix>::Type_t
peekColor(const PScalar<T>& l, int row, int col)
{
  return peekColor(l.elem(),row,col);
}

//! Extract spin vector components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T>
inline typename UnaryReturn<PScalar<T>, FnPeekSpinVector>::Type_t
peekSpin(const PScalar<T>& l, int row)
{
  return peekSpin(l.elem(),row);
}

//! Extract spin matrix components 
/*! Generically, this is an identity operation. Defined differently under spin */
template<class T>
inline typename UnaryReturn<PScalar<T>, FnPeekSpinMatrix>::Type_t
peekSpin(const PScalar<T>& l, int row, int col)
{
  return peekSpin(l.elem(),row,col);
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


//-----------------------------------------------------------------------------
// Contraction for color matrices
// colorContract 
//! dest  = colorContract(Qprop1,Qprop2,Qprop3)
/*!
 * This routine is completely unrolled for 3 colors
 */
template<class T1, class T2, class T3>
struct TrinaryReturn<PScalar<T1>, PScalar<T2>, PScalar<T3>, FnColorContract> {
  typedef PScalar<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};

template<class T1, class T2, class T3>
inline typename TrinaryReturn<PScalar<T1>, PScalar<T2>, PScalar<T3>, FnColorContract>::Type_t
colorContract(const PScalar<T1>& s1, const PScalar<T2>& s2, const PScalar<T3>& s3)
{
  return colorContract(s1.elem(), s2.elem(), s3.elem());
}




//! dest = (mask) ? s1 : dest
template<class T, class T1> 
inline void 
copymask(PScalar<T>& d, const PScalar<T1>& mask, const PScalar<T>& s1) 
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
  return sum(s1.elem());
}
#endif


// InnerProduct (norm-seq) global sum = sum(tr(adj(s1)*s1))
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
  return localNorm2(s1.elem());
}



//! PScalar<T> = InnerProduct(adj(PScalar<T1>)*PScalar<T2>)
template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PScalar<T2>, FnInnerProduct > {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PScalar<T2>, FnLocalInnerProduct > {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, FnLocalInnerProduct>::Type_t
localInnerProduct(const PScalar<T1>& s1, const PScalar<T2>& s2)
{
  return localInnerProduct(s1.elem(), s2.elem());
}


//! PScalar<T> = InnerProductReal(adj(PMatrix<T1>)*PMatrix<T1>)
template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PScalar<T2>, FnInnerProductReal > {
  typedef PScalar<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<PScalar<T1>, PScalar<T2>, FnLocalInnerProductReal > {
  typedef PScalar<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<PScalar<T1>, PScalar<T2>, FnLocalInnerProductReal>::Type_t
localInnerProductReal(const PScalar<T1>& s1, const PScalar<T2>& s2)
{
  return localInnerProductReal(s1.elem(), s2.elem());
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
  return where(a.elem(), b.elem(), c.elem());
}


//-----------------------------------------------------------------------------
//! QDP Int to int primitive in conversion routine
template<class T> 
inline int 
toInt(const PScalar<T>& s) 
{
  return toInt(s.elem());
}

//! QDP Real to float primitive in conversion routine
template<class T> 
inline float
toFloat(const PScalar<T>& s) 
{
  return toFloat(s.elem());
}

//! QDP Double to double primitive in conversion routine
template<class T> 
inline double
toDouble(const PScalar<T>& s) 
{
  return toDouble(s.elem());
}

//! QDP Boolean to bool primitive in conversion routine
template<class T> 
inline bool
toBool(const PScalar<T>& s) 
{
  return toBool(s.elem());
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

