// -*- C++ -*-
// $Id: reality.h,v 1.1 2002-09-12 18:22:16 edwards Exp $
//
// QDP data parallel interface
//

QDP_BEGIN_NAMESPACE(QDP);

//-------------------------------------------------------------------------------------
//! Scalar reality (not complex)
template<class T> class RScalar
{
public:
  RScalar() {}
  ~RScalar() {}

  //! construct dest = const
  RScalar(const typename WordType<T>::Type_t& rhs) : F(rhs) {}


  //! construct dest = const
  template<class T1>
  RScalar(const RScalar<T1>& rhs)
    {
      elem() = rhs.elem();
    }

#if 0
  //! dest = const
  /*! Fill with a constant. Will be promoted to underlying word type */
  inline
  RScalar& operator=(const typename WordType<T>::Type_t& rhs)
    {
      elem() = rhs;
      return *this;
    }
#endif

  //! RScalar = RScalar
  /*! Set equal to another RScalar */
  template<class T1>
  inline
  RScalar& operator=(const RScalar<T1>& rhs) 
    {
      elem() = rhs.elem();
      return *this;
    }

  //! RScalar += RScalar
  template<class T1>
  inline
  RScalar& operator+=(const RScalar<T1>& rhs) 
    {
      elem() += rhs.elem();
      return *this;
    }

  //! RScalar -= RScalar
  template<class T1>
  inline
  RScalar& operator-=(const RScalar<T1>& rhs) 
    {
      elem() -= rhs.elem();
      return *this;
    }

  //! RScalar *= RScalar
  template<class T1>
  inline
  RScalar& operator*=(const RScalar<T1>& rhs) 
    {
      elem() *= rhs.elem();
      return *this;
    }

  //! RScalar /= RScalar
  template<class T1>
  inline
  RScalar& operator/=(const RScalar<T1>& rhs) 
    {
      elem() /= rhs.elem();
      return *this;
    }

  //! RScalar %= RScalar
  template<class T1>
  inline
  RScalar& operator%=(const RScalar<T1>& rhs) 
    {
      elem() %= rhs.elem();
      return *this;
    }

  //! RScalar |= RScalar
  template<class T1>
  inline
  RScalar& operator|=(const RScalar<T1>& rhs) 
    {
      elem() |= rhs.elem();
      return *this;
    }

  //! RScalar &= RScalar
  template<class T1>
  inline
  RScalar& operator&=(const RScalar<T1>& rhs) 
    {
      elem() &= rhs.elem();
      return *this;
    }

  //! RScalar ^= RScalar
  template<class T1>
  inline
  RScalar& operator^=(const RScalar<T1>& rhs) 
    {
      elem() ^= rhs.elem();
      return *this;
    }

  //! RScalar <<= RScalar
  template<class T1>
  inline
  RScalar& operator<<=(const RScalar<T1>& rhs) 
    {
      elem() <<= rhs.elem();
      return *this;
    }

  //! RScalar >>= RScalar
  template<class T1>
  inline
  RScalar& operator>>=(const RScalar<T1>& rhs) 
    {
      elem() >>= rhs.elem();
      return *this;
    }


  //! Do deep copies here
  RScalar(const RScalar& a): F(a.F) {}

public:
  T& elem() {return F;}
  const T& elem() const {return F;}

private:
  T F;
};

 
//! Ascii output
template<class T>  ostream& operator<<(ostream& s, const RScalar<T>& d)
{
  return s << d.elem();
}



//-------------------------------------------------------------------------------------
//! Reality complex
/*! All fields are either complex or scalar reality */
template<class T> class RComplex
{
public:
  RComplex() {}
  ~RComplex() {}

#if 0
  //! dest = int
  /*! Fill with an integer constant. Will be promoted to underlying word type */
  inline
  RComplex& operator=(const typename WordType<T>::Type_t& rhs)
    {
      real() = rhs;
      zero(imag());
      return *this;
    }
#endif

  //! RComplex = RScalar
  /*! Set the real part and zero the imag part */
  template<class T1>
  inline
  RComplex& operator=(const RScalar<T1>& rhs) 
    {
      real() = rhs.elem();
      zero(imag());
      return *this;
    }

  //! RComplex += RScalar
  template<class T1>
  inline
  RComplex& operator+=(const RScalar<T1>& rhs) 
    {
      real() += rhs.elem();
      return *this;
    }

  //! RComplex -= RScalar
  template<class T1>
  inline
  RComplex& operator-=(const RScalar<T1>& rhs) 
    {
      real() -= rhs.elem();
      return *this;
    }

  //! RComplex *= RScalar
  template<class T1>
  inline
  RComplex& operator*=(const RScalar<T1>& rhs) 
    {
      real() *= rhs.elem();
      imag() *= rhs.elem();
      return *this;
    }

  //! RComplex /= RScalar
  template<class T1>
  inline
  RComplex& operator/=(const RScalar<T1>& rhs) 
    {
      real() /= rhs.elem();
      imag() /= rhs.elem();
      return *this;
    }



  //! RComplex = RComplex
  /*! Set equal to another RComplex */
  template<class T1>
  inline
  RComplex& operator=(const RComplex<T1>& rhs) 
    {
      real() = rhs.real();
      imag() = rhs.imag();
      return *this;
    }

  //! RComplex += RComplex
  template<class T1>
  inline
  RComplex& operator+=(const RComplex<T1>& rhs) 
    {
      real() += rhs.real();
      imag() += rhs.imag();
      return *this;
    }

  //! RComplex -= RComplex
  template<class T1>
  inline
  RComplex& operator-=(const RComplex<T1>& rhs) 
    {
      real() -= rhs.real();
      imag() -= rhs.imag();
      return *this;
    }

  //! RComplex *= RComplex
  template<class T1>
  inline
  RComplex& operator*=(const RComplex<T1>& rhs) 
    {
      RComplex<T> d;
      d = *this * rhs;

      real() = d.real();
      imag() = d.imag();
      return *this;
    }

  //! RComplex /= RComplex
  template<class T1>
  inline
  RComplex& operator/=(const RComplex<T1>& rhs) 
    {
      RComplex<T> d;
      d = *this / rhs;

      real() = d.real();
      imag() = d.imag();
      return *this;
    }


  //! Deep copy constructor
  RComplex(const RComplex& a): re(a.re), im(a.im) {}

public:
  T& real() {return re;}
  const T& real() const {return re;}

  T& imag() {return im;}
  const T& imag() const {return im;}

private:
  T re;
  T im;
};

//! Ascii output
template<class T>  ostream& operator<<(ostream& s, const RComplex<T>& d)
{
  return s << "( " << d.real() << " , " << d.imag() << " )";
}



//-----------------------------------------------------------------------------
// Traits classes 
//-----------------------------------------------------------------------------

// Underlying word type
template<class T>
struct WordType<RScalar<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};

template<class T>
struct WordType<RComplex<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};


// Internally used scalars
template<class T>
struct InternalScalar<RScalar<T> > {
  typedef RScalar<typename InternalScalar<T>::Type_t>  Type_t;
};

template<class T>
struct InternalScalar<RComplex<T> > {
  typedef RScalar<typename InternalScalar<T>::Type_t>  Type_t;
};


//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

// Default unary(RScalar) -> RScalar
template<class T1, class Op>
struct UnaryReturn<RScalar<T1>, Op> {
  typedef RScalar<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};

// Default unary(RComplex) -> RComplex
template<class T1, class Op>
struct UnaryReturn<RComplex<T1>, Op> {
  typedef RComplex<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};

// Default binary(RScalar,RScalar) -> RScalar
template<class T1, class T2, class Op>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, Op> {
  typedef RScalar<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};

// Default binary(RComplex,RComplex) -> RComplex
template<class T1, class T2, class Op>
struct BinaryReturn<RComplex<T1>, RComplex<T2>, Op> {
  typedef RComplex<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};

// Default binary(RScalar,RComplex) -> RComplex
template<class T1, class T2, class Op>
struct BinaryReturn<RScalar<T1>, RComplex<T2>, Op> {
  typedef RComplex<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};

// Default binary(RComplex,RScalar) -> RComplex
template<class T1, class T2, class Op>
struct BinaryReturn<RComplex<T1>, RScalar<T2>, Op> {
  typedef RComplex<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


// RScalar
#if 0
template<class T1, class T2>
struct UnaryReturn<RScalar<T2>, OpCast<T1> > {
  typedef RScalar<typename UnaryReturn<T, OpCast>::Type_t>  Type_t;
//  typedef T1 Type_t;
};
#endif


template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpAddAssign > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpAddAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpSubtractAssign > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpSubtractAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpMultiplyAssign > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpMultiplyAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpDivideAssign > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpDivideAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpModAssign > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpModAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpBitwiseOrAssign > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpBitwiseOrAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpBitwiseAndAssign > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpBitwiseAndAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpBitwiseXorAssign > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpBitwiseXorAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpLeftShiftAssign > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpLeftShiftAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpRightShiftAssign > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpRightShiftAssign>::Type_t>  Type_t;
};
 

// RScalar
// Gamma algebra
template<int N, int m, class T2, class OpGammaConstMultiply>
struct BinaryReturn<GammaConst<N,m>, RScalar<T2>, OpGammaConstMultiply> {
  typedef RScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, int m, class OpMultiplyGammaConst>
struct BinaryReturn<RScalar<T2>, GammaConst<N,m>, OpMultiplyGammaConst> {
  typedef RScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpGammaTypeMultiply>
struct BinaryReturn<GammaType<N>, RScalar<T2>, OpGammaTypeMultiply> {
  typedef RScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpMultiplyGammaType>
struct BinaryReturn<RScalar<T2>, GammaType<N>, OpMultiplyGammaType> {
  typedef RScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};



// RComplex
// Gamma algebra
template<int N, int m, class T2, class OpGammaConstMultiply>
struct BinaryReturn<GammaConst<N,m>, RComplex<T2>, OpGammaConstMultiply> {
  typedef RComplex<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, int m, class OpMultiplyGammaConst>
struct BinaryReturn<RComplex<T2>, GammaConst<N,m>, OpMultiplyGammaConst> {
  typedef RComplex<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpGammaTypeMultiply>
struct BinaryReturn<GammaType<N>, RComplex<T2>, OpGammaTypeMultiply> {
  typedef RComplex<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpMultiplyGammaType>
struct BinaryReturn<RComplex<T2>, GammaType<N>, OpMultiplyGammaType> {
  typedef RComplex<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};


// Assignment is different
template<class T1, class T2 >
struct BinaryReturn<RComplex<T1>, RComplex<T2>, OpAssign > {
//  typedef RComplex<T1> &Type_t;
  typedef RComplex<typename BinaryReturn<T1, T2, OpAssign>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<RComplex<T1>, RComplex<T2>, OpAddAssign > {
  typedef RComplex<typename BinaryReturn<T1, T2, OpAddAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplex<T1>, RComplex<T2>, OpSubtractAssign > {
  typedef RComplex<typename BinaryReturn<T1, T2, OpSubtractAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplex<T1>, RComplex<T2>, OpMultiplyAssign > {
  typedef RComplex<typename BinaryReturn<T1, T2, OpMultiplyAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplex<T1>, RComplex<T2>, OpDivideAssign > {
  typedef RComplex<typename BinaryReturn<T1, T2, OpDivideAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplex<T1>, RComplex<T2>, OpModAssign > {
  typedef RComplex<typename BinaryReturn<T1, T2, OpModAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplex<T1>, RComplex<T2>, OpBitwiseOrAssign > {
  typedef RComplex<typename BinaryReturn<T1, T2, OpBitwiseOrAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplex<T1>, RComplex<T2>, OpBitwiseAndAssign > {
  typedef RComplex<typename BinaryReturn<T1, T2, OpBitwiseAndAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplex<T1>, RComplex<T2>, OpBitwiseXorAssign > {
  typedef RComplex<typename BinaryReturn<T1, T2, OpBitwiseXorAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplex<T1>, RComplex<T2>, OpLeftShiftAssign > {
  typedef RComplex<typename BinaryReturn<T1, T2, OpLeftShiftAssign>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<RComplex<T1>, RComplex<T2>, OpRightShiftAssign > {
  typedef RComplex<typename BinaryReturn<T1, T2, OpRightShiftAssign>::Type_t>  Type_t;
};
 





//-----------------------------------------------------------------------------
// Operators
//-----------------------------------------------------------------------------

// Scalar Reality
template<class T>
struct UnaryReturn<RScalar<T>, OpNot > {
  typedef RScalar<typename UnaryReturn<T, OpNot>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RScalar<T1>, OpNot>::Type_t
operator!(const RScalar<T1>& l)
{
  typename UnaryReturn<RScalar<T1>, OpNot>::Type_t  d;

  d.elem() = ! l.elem();
  return d;
}


template<class T1>
inline typename UnaryReturn<RScalar<T1>, OpUnaryPlus>::Type_t
operator+(const RScalar<T1>& l)
{
  typename UnaryReturn<RScalar<T1>, OpUnaryPlus>::Type_t  d;

  d.elem() = +l.elem();
  return d;
}


template<class T1>
inline typename UnaryReturn<RScalar<T1>, OpUnaryMinus>::Type_t
operator-(const RScalar<T1>& l)
{
  typename UnaryReturn<RScalar<T1>, OpUnaryMinus>::Type_t  d;

  d.elem() = -l.elem();
  return d;
}


template<class T1, class T2>
inline typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpAdd>::Type_t
operator+(const RScalar<T1>& l, const RScalar<T2>& r)
{
  typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpAdd>::Type_t  d;

  d.elem() = l.elem()+r.elem();
  return d;
}


template<class T1, class T2>
inline typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpSubtract>::Type_t
operator-(const RScalar<T1>& l, const RScalar<T2>& r)
{
  typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpSubtract>::Type_t  d;

  d.elem() = l.elem() - r.elem();
  return d;
}


template<class T1, class T2>
inline typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpMultiply>::Type_t
operator*(const RScalar<T1>& l, const RScalar<T2>& r)
{
  typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpMultiply>::Type_t  d;

  d.elem() = l.elem() * r.elem();
  return d;
}


template<class T1, class T2>
inline typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpDivide>::Type_t
operator/(const RScalar<T1>& l, const RScalar<T2>& r)
{
  typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpDivide>::Type_t  d;

  d.elem() = l.elem() / r.elem();
  return d;
}



template<class T1, class T2 >
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpLeftShift > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};
 

template<class T1, class T2>
inline typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpLeftShift>::Type_t
operator<<(const RScalar<T1>& l, const RScalar<T2>& r)
{
  typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpLeftShift>::Type_t  d;

  d.elem() = l.elem() << r.elem();
  return d;
}


template<class T1, class T2 >
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpRightShift > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};
 

template<class T1, class T2>
inline typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpRightShift>::Type_t
operator>>(const RScalar<T1>& l, const RScalar<T2>& r)
{
  typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpRightShift>::Type_t  d;

  d.elem() = l.elem() >> r.elem();
  return d;
}


template<class T1, class T2 >
inline typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpMod>::Type_t
operator%(const RScalar<T1>& l, const RScalar<T2>& r)
{
  typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpMod>::Type_t  d;

  d.elem() = l.elem() % r.elem();
  return d;
}

template<class T1, class T2 >
inline typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpBitwiseXor>::Type_t
operator^(const RScalar<T1>& l, const RScalar<T2>& r)
{
  typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpBitwiseXor>::Type_t  d;

  d.elem() = l.elem() ^ r.elem();
  return d;
}

template<class T1, class T2 >
inline typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpBitwiseAnd>::Type_t
operator&(const RScalar<T1>& l, const RScalar<T2>& r)
{
  typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpBitwiseAnd>::Type_t  d;

  d.elem() = l.elem() & r.elem();
  return d;
}

template<class T1, class T2>
inline typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpBitwiseOr>::Type_t
operator|(const RScalar<T1>& l, const RScalar<T2>& r)
{
  typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpBitwiseOr>::Type_t  d;

  d.elem() = l.elem() | r.elem();
  return d;
}



// Comparisons
template<class T1, class T2 >
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpLT > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpLT>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpLT>::Type_t
operator<(const RScalar<T1>& l, const RScalar<T2>& r)
{
  typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpLT>::Type_t  d;

  d.elem() = l.elem() < r.elem();
  return d;
}


template<class T1, class T2 >
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpLE > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpLE>::Type_t
operator<=(const RScalar<T1>& l, const RScalar<T2>& r)
{
  typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpLE>::Type_t  d;

  d.elem() = l.elem() <= r.elem();
  return d;
}


template<class T1, class T2 >
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpGT > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpGT>::Type_t
operator>(const RScalar<T1>& l, const RScalar<T2>& r)
{
  typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpGT>::Type_t  d;

  d.elem() = l.elem() > r.elem();
  return d;
}


template<class T1, class T2 >
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpGE > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpGE>::Type_t
operator>=(const RScalar<T1>& l, const RScalar<T2>& r)
{
  typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpGE>::Type_t  d;

  d.elem() = l.elem() >= r.elem();
  return d;
}


template<class T1, class T2 >
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpEQ > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpEQ>::Type_t
operator==(const RScalar<T1>& l, const RScalar<T2>& r)
{
  typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpEQ>::Type_t  d;

  d.elem() = l.elem() == r.elem();
  return d;
}


template<class T1, class T2 >
struct BinaryReturn<RScalar<T1>, RScalar<T2>, OpNE > {
  typedef RScalar<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpNE>::Type_t
operator!=(const RScalar<T1>& l, const RScalar<T2>& r)
{
  typename BinaryReturn<RScalar<T1>, RScalar<T2>, OpNE>::Type_t  d;

  d.elem() = l.elem() != r.elem();
  return d;
}



//-----------------------------------------------------------------------------
// Functions

// Conjugate
template<class T1>
inline typename UnaryReturn<RScalar<T1>, FnConj>::Type_t
conj(const RScalar<T1>& s1)
{
  typename UnaryReturn<RScalar<T1>, FnConj>::Type_t  d;

  d.elem() = conj(s1.elem());
  return d;
}



// TRACE
// trace = Trace(source1)
template<class T>
struct UnaryReturn<RScalar<T>, FnTrace > {
  typedef RScalar<typename UnaryReturn<T, FnTrace>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RScalar<T1>, FnTrace>::Type_t
trace(const RScalar<T1>& s1)
{
  typename UnaryReturn<RScalar<T1>, FnTrace>::Type_t  d;

  d.elem() = trace(s1.elem());
  return d;
}


// trace = Re(Trace(source1))
template<class T>
struct UnaryReturn<RScalar<T>, FnTraceReal > {
  typedef RScalar<typename UnaryReturn<T, FnTraceReal>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RScalar<T1>, FnTraceReal>::Type_t
trace_real(const RScalar<T1>& s1)
{
  typename UnaryReturn<RScalar<T1>, FnTraceReal>::Type_t  d;

  d.elem() = trace_real(s1.elem());
  return d;
}


// trace = Im(Trace(source1))
template<class T>
struct UnaryReturn<RScalar<T>, FnTraceImag > {
  typedef RScalar<typename UnaryReturn<T, FnTraceImag>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RScalar<T1>, FnTraceImag>::Type_t
trace_imag(const RScalar<T1>& s1)
{
  typename UnaryReturn<RScalar<T1>, FnTraceImag>::Type_t  d;

  d.elem() = trace_imag(s1.elem());
  return d;
}


// RScalar = Re(RScalar)  [identity]
template<class T>
inline typename UnaryReturn<RScalar<T>, FnReal>::Type_t
real(const RScalar<T>& s1)
{
  typename UnaryReturn<RScalar<T>, FnReal>::Type_t  d;

  d.elem() = s1.elem();
  return d;
}


// RScalar = Im(RScalar) [this is zero]
template<class T>
inline typename UnaryReturn<RScalar<T>, FnImag>::Type_t
imag(const RScalar<T>& s1)
{
  typename UnaryReturn<RScalar<T>, FnImag>::Type_t  d;

  zero(d.elem());
  return d;
}


// ArcCos
template<class T1>
inline typename UnaryReturn<RScalar<T1>, FnArcCos>::Type_t
arccos(const RScalar<T1>& s1)
{
  typename UnaryReturn<RScalar<T1>, FnArcCos>::Type_t  d;

  d.elem() = arccos(s1.elem());
  return d;
}

// ArcSin
template<class T1>
inline typename UnaryReturn<RScalar<T1>, FnArcSin>::Type_t
arcsin(const RScalar<T1>& s1)
{
  typename UnaryReturn<RScalar<T1>, FnArcSin>::Type_t  d;

  d.elem() = arcsin(s1.elem());
  return d;
}

// ArcTan
template<class T1>
inline typename UnaryReturn<RScalar<T1>, FnArcTan>::Type_t
arctan(const RScalar<T1>& s1)
{
  typename UnaryReturn<RScalar<T1>, FnArcTan>::Type_t  d;

  d.elem() = arctan(s1.elem());
  return d;
}

// Cos
template<class T1>
inline typename UnaryReturn<RScalar<T1>, FnCos>::Type_t
cos(const RScalar<T1>& s1)
{
  typename UnaryReturn<RScalar<T1>, FnCos>::Type_t  d;

  d.elem() = cos(s1.elem());
  return d;
}

// Exp
template<class T1>
inline typename UnaryReturn<RScalar<T1>, FnExp>::Type_t
exp(const RScalar<T1>& s1)
{
  typename UnaryReturn<RScalar<T1>, FnExp>::Type_t  d;

  d.elem() = exp(s1.elem());
  return d;
}

// Fabs
template<class T1>
inline typename UnaryReturn<RScalar<T1>, FnFabs>::Type_t
fabs(const RScalar<T1>& s1)
{
  typename UnaryReturn<RScalar<T1>, FnFabs>::Type_t  d;

  d.elem() = fabs(s1.elem());
  return d;
}

// Log
template<class T1>
inline typename UnaryReturn<RScalar<T1>, FnLog>::Type_t
log(const RScalar<T1>& s1)
{
  typename UnaryReturn<RScalar<T1>, FnLog>::Type_t  d;

  d.elem() = log(s1.elem());
  return d;
}

// Sin
template<class T1>
inline typename UnaryReturn<RScalar<T1>, FnSin>::Type_t
sin(const RScalar<T1>& s1)
{
  typename UnaryReturn<RScalar<T1>, FnSin>::Type_t  d;

  d.elem() = sin(s1.elem());
  return d;
}

// Sqrt
template<class T1>
inline typename UnaryReturn<RScalar<T1>, FnSqrt>::Type_t
sqrt(const RScalar<T1>& s1)
{
  typename UnaryReturn<RScalar<T1>, FnSqrt>::Type_t  d;

  d.elem() = sqrt(s1.elem());
  return d;
}

// Tan
template<class T1>
inline typename UnaryReturn<RScalar<T1>, FnTan>::Type_t
tan(const RScalar<T1>& s1)
{
  typename UnaryReturn<RScalar<T1>, FnTan>::Type_t  d;

  d.elem() = tan(s1.elem());
  return d;
}



//! dest = (mask) ? s1 : dest
template<class T, class T1> 
inline
void copymask(RScalar<T>& d, const RScalar<T1>& mask, const RScalar<T>& s1) 
{
  copymask(d.elem(),mask.elem(),s1.elem());
}

//! dest [float type] = source [seed type]
template<class T, class T1>
inline
void seed_to_float(RScalar<T>& d, const RScalar<T1>& s1)
{
  seed_to_float(d.elem(), s1.elem());
}

//! dest [float type] = source [int type]
template<class T, class T1>
inline
void cast_rep(T& d, const RScalar<T1>& s1)
{
  cast_rep(d, s1.elem());
}


// Global sum over site indices only
template<class T>
struct UnaryReturn<RScalar<T>, FnSum > {
  typedef RScalar<typename UnaryReturn<T, FnSum>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<RScalar<T>, FnSum>::Type_t
sum(const RScalar<T>& s1)
{
  typename UnaryReturn<RScalar<T>, FnSum>::Type_t  d;

  d.elem() = sum(s1.elem());
  return d;
}


// Innerproduct (norm-seq) global sum = sum(tr(conj(s1)*s1))
template<class T>
struct UnaryReturn<RScalar<T>, FnSumSq > {
  typedef RScalar<typename UnaryReturn<T, FnSumSq>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<RScalar<T>, FnSumSq>::Type_t
sumsq(const RScalar<T>& s1)
{
  typename UnaryReturn<RScalar<T>, FnSumSq>::Type_t  d;

  d.elem() = sumsq(s1.elem());
  return d;
}



//! RScalar<T> = Innerproduct(Conj(RScalar<T1>)*RScalar<T2>)
template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, FnInnerproduct > {
  typedef RScalar<typename BinaryReturn<T1, T2, FnInnerproduct>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalar<T1>, RScalar<T2>, FnInnerproduct>::Type_t
innerproduct(const RScalar<T1>& s1, const RScalar<T2>& s2)
{
  typename BinaryReturn<RScalar<T1>, RScalar<T2>, FnInnerproduct>::Type_t  d;

  d.elem() = innerproduct(s1.elem(), s2.elem());
  return d;
}


//! RScalar<T> = Innerproduct_real(Conj(PMatrix<T1>)*PMatrix<T1>)
template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, FnInnerproduct_real > {
  typedef RScalar<typename BinaryReturn<T1, T2, FnInnerproduct_real>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalar<T1>, RScalar<T2>, FnInnerproduct_real>::Type_t
innerproduct_real(const RScalar<T1>& s1, const RScalar<T2>& s2)
{
  typename BinaryReturn<RScalar<T1>, RScalar<T2>, FnInnerproduct_real>::Type_t  d;

  d.elem() = innerproduct_real(s1.elem(), s2.elem());
  return d;
}


//-----------------------------------------------------------------------------
// Broadcast operations
//! dest = 0
template<class T> 
inline
void zero(RScalar<T>& dest) 
{
  zero(dest.elem());
}

//! dest  = random  
template<class T, class T1, class T2>
inline void
fill_random(RScalar<T>& d, T1& seed, T2& skewed_seed, const T1& seed_mult)
{
  fill_random(d.elem(), seed, skewed_seed, seed_mult);
}



//! dest  = gaussian  
/*! Real form of complex polar method */
template<class T>
inline void
fill_gaussian(RScalar<T>& d, RScalar<T>& r1, RScalar<T>& r2)
{
  typedef typename InternalScalar<T>::Type_t  S;

  // r1 and r2 are the input random numbers needed

  /* Stage 2: get the cos of the second number  */
  T  g_r;

  r2.elem() *= S(6.283185307);
  g_r = cos(r2.elem());
    
  /* Stage 4: get  sqrt(-2.0 * log(u1)) */
  r1.elem() = sqrt(-S(2.0) * log(r1.elem()));

  /* Stage 5:   g_r = sqrt(-2*log(u1))*cos(2*pi*u2) */
  /* Stage 5:   g_i = sqrt(-2*log(u1))*sin(2*pi*u2) */
  d.elem() = r1.elem() * g_r;
}





//-----------------------------------------------------------------------------
// Complex Reality
//-----------------------------------------------------------------------------

template<class T1>
inline typename UnaryReturn<RComplex<T1>, OpUnaryPlus>::Type_t
operator+(const RComplex<T1>& l)
{
  typename UnaryReturn<RComplex<T1>, OpUnaryPlus>::Type_t  d;

  d.real() = +l.real();
  d.imag() = +l.imag();
  return d;
}


template<class T1>
inline typename UnaryReturn<RComplex<T1>, OpUnaryMinus>::Type_t
operator-(const RComplex<T1>& l)
{
  typename UnaryReturn<RComplex<T1>, OpUnaryMinus>::Type_t  d;

  d.real() = -l.real();
  d.imag() = -l.imag();
  return d;
}


template<class T1, class T2>
inline typename BinaryReturn<RComplex<T1>, RComplex<T2>, OpAdd>::Type_t
operator+(const RComplex<T1>& l, const RComplex<T2>& r)
{
  typename BinaryReturn<RComplex<T1>, RComplex<T2>, OpAdd>::Type_t  d;

  d.real() = l.real()+r.real();
  d.imag() = l.imag()+r.imag();
  return d;
}


template<class T1, class T2>
inline typename BinaryReturn<RComplex<T1>, RComplex<T2>, OpSubtract>::Type_t
operator-(const RComplex<T1>& l, const RComplex<T2>& r)
{
  typename BinaryReturn<RComplex<T1>, RComplex<T2>, OpSubtract>::Type_t  d;

  d.real() = l.real() - r.real();
  d.imag() = l.imag() - r.imag();
  return d;
}


template<class T1, class T2>
inline typename BinaryReturn<RComplex<T1>, RComplex<T2>, OpMultiply>::Type_t
operator*(const RComplex<T1>& l, const RComplex<T2>& r)
{
  typename BinaryReturn<RComplex<T1>, RComplex<T2>, OpMultiply>::Type_t  d;

  d.real() = l.real()*r.real() - l.imag()*r.imag();
  d.imag() = l.real()*r.imag() + l.imag()*r.real();
  return d;
}


template<class T1, class T2>
inline typename BinaryReturn<RScalar<T1>, RComplex<T2>, OpMultiply>::Type_t
operator*(const RScalar<T1>& l, const RComplex<T2>& r)
{
  typename BinaryReturn<RScalar<T1>, RComplex<T2>, OpMultiply>::Type_t  d;

  d.real() = l.elem()*r.real();
  d.imag() = l.elem()*r.imag();
  return d;
}

template<class T1, class T2>
inline typename BinaryReturn<RComplex<T1>, RScalar<T2>, OpMultiply>::Type_t
operator*(const RComplex<T1>& l, const RScalar<T2>& r)
{
  typename BinaryReturn<RComplex<T1>, RScalar<T2>, OpMultiply>::Type_t  d;

  d.real() = l.real()*r.elem();
  d.imag() = l.real()*r.elem();
  return d;
}


template<class T1, class T2>
inline typename BinaryReturn<RComplex<T1>, RComplex<T2>, OpDivide>::Type_t
operator/(const RComplex<T1>& l, const RComplex<T2>& r)
{
  typename BinaryReturn<RComplex<T1>, RComplex<T2>, OpDivide>::Type_t  d;

  typedef typename InternalScalar<T2>::Type_t  S;
  S tmp = S(1.0) / (r.real()*r.real() + r.imag()*r.imag());

  d.real() = (l.real()*r.real() + l.imag()*r.imag()) * tmp;
  d.imag() = (l.imag()*r.real() - l.real()*r.imag()) * tmp;
  return d;
}


template<class T1, class T2>
inline typename BinaryReturn<RComplex<T1>, RScalar<T2>, OpDivide>::Type_t
operator/(const RComplex<T1>& l, const RScalar<T2>& r)
{
  typename BinaryReturn<RComplex<T1>, RScalar<T2>, OpDivide>::Type_t  d;

  d.real() = l.real() / r.elem();
  d.imag() = l.imag() / r.elem();
  return d;
}


template<class T1, class T2>
inline typename BinaryReturn<RScalar<T1>, RComplex<T2>, OpDivide>::Type_t
operator/(const RScalar<T1>& l, const RComplex<T2>& r)
{
  typename BinaryReturn<RScalar<T1>, RComplex<T2>, OpDivide>::Type_t  d;

  typedef typename InternalScalar<T2>::Type_t  S;
  S tmp = S(1.0) / (r.real()*r.real() + r.imag()*r.imag());

  d.real() = l.elem() * r.real() * tmp;
  d.imag() = -l.elem() * r.imag() * tmp;
  return d;
}



//-----------------------------------------------------------------------------
// Functions

// Conjugate
template<class T1>
inline typename UnaryReturn<RComplex<T1>, FnConj>::Type_t
conj(const RComplex<T1>& l)
{
  typename UnaryReturn<RComplex<T1>, FnConj>::Type_t  d;

  d.real() = l.real();
  d.imag() = -l.imag();
  return d;
}

// TRACE
// trace = Trace(source1)
template<class T>
struct UnaryReturn<RComplex<T>, FnTrace > {
  typedef RComplex<typename UnaryReturn<T, FnTrace>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RComplex<T1>, FnTrace>::Type_t
trace(const RComplex<T1>& s1)
{
  typename UnaryReturn<RComplex<T1>, FnTrace>::Type_t  d;

  d.real() = trace(s1.real());
  d.imag() = trace(s1.imag());
  return d;
}


// trace = Re(Trace(source1))
template<class T>
struct UnaryReturn<RComplex<T>, FnTraceReal > {
  typedef RScalar<typename UnaryReturn<T, FnTraceReal>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RComplex<T1>, FnTraceReal>::Type_t
trace_real(const RComplex<T1>& s1)
{
  typename UnaryReturn<RComplex<T1>, FnTraceReal>::Type_t  d;

  d.elem() = trace(s1.real());
  return d;
}


// trace = Im(Trace(source1))
template<class T>
struct UnaryReturn<RComplex<T>, FnTraceImag > {
  typedef RScalar<typename UnaryReturn<T, FnTraceImag>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RComplex<T1>, FnTraceImag>::Type_t
trace_imag(const RComplex<T1>& s1)
{
  typename UnaryReturn<RComplex<T1>, FnTraceImag>::Type_t  d;

  d.elem() = trace(s1.imag());
  return d;
}


// RScalar = Re(RComplex)
template<class T>
struct UnaryReturn<RComplex<T>, FnReal > {
  typedef RScalar<typename UnaryReturn<T, FnReal>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RComplex<T1>, FnReal>::Type_t
real(const RComplex<T1>& s1)
{
  typename UnaryReturn<RComplex<T1>, FnReal>::Type_t  d;

  d.elem() = s1.real();
  return d;
}

// RScalar = Im(RComplex)
template<class T>
struct UnaryReturn<RComplex<T>, FnImag > {
  typedef RScalar<typename UnaryReturn<T, FnImag>::Type_t>  Type_t;
};

template<class T1>
inline typename UnaryReturn<RComplex<T1>, FnImag>::Type_t
imag(const RComplex<T1>& s1)
{
  typename UnaryReturn<RComplex<T1>, FnImag>::Type_t  d;

  d.elem() = s1.imag();
  return d;
}


//! RComplex<T> = (RScalar<T> , RScalar<T>)
template<class T1, class T2>
struct BinaryReturn<RScalar<T1>, RScalar<T2>, FnCmplx > {
  typedef RComplex<typename BinaryReturn<T1, T2, FnCmplx>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RScalar<T1>, RScalar<T2>, FnCmplx>::Type_t
cmplx(const RScalar<T1>& s1, const RScalar<T2>& s2)
{
  typename BinaryReturn<RScalar<T1>, RScalar<T2>, FnCmplx>::Type_t  d;

  d.real() = s1.elem();
  d.imag() = s2.elem();
  return d;
}



// RComplex = i * RScalar
template<class T>
struct UnaryReturn<RScalar<T>, FnMultiplyI > {
  typedef RComplex<typename UnaryReturn<T, FnMultiplyI>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<RScalar<T>, FnMultiplyI>::Type_t
multiplyI(const RScalar<T>& s1)
{
  typename UnaryReturn<RScalar<T>, FnMultiplyI>::Type_t  d;

  zero(d.real());
  d.imag() = s1.elem();
  return d;
}

// RComplex = i * RComplex
template<class T>
inline typename UnaryReturn<RComplex<T>, FnMultiplyI>::Type_t
multiplyI(const RComplex<T>& s1)
{
  typename UnaryReturn<RComplex<T>, FnMultiplyI>::Type_t  d;

  d.real() = -s1.imag();
  d.imag() =  s1.real();
  return d;
}


// RComplex = -i * RScalar
template<class T>
struct UnaryReturn<RScalar<T>, FnMultiplyMinusI > {
  typedef RComplex<typename UnaryReturn<T, FnMultiplyMinusI>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<RScalar<T>, FnMultiplyMinusI>::Type_t
multiplyMinusI(const RScalar<T>& s1)
{
  typename UnaryReturn<RScalar<T>, FnMultiplyMinusI>::Type_t  d;

  zero(d.real());
  d.imag() = -s1.elem();
  return d;
}


// RComplex = -i * RComplex
template<class T>
inline typename UnaryReturn<RComplex<T>, FnMultiplyMinusI>::Type_t
multiplyMinusI(const RComplex<T>& s1)
{
  typename UnaryReturn<RComplex<T>, FnMultiplyMinusI>::Type_t  d;

  d.real() =  s1.imag();
  d.imag() = -s1.real();
  return d;
}



//! dest = (mask) ? s1 : dest
template<class T, class T1> 
inline
void copymask(RComplex<T>& d, const RScalar<T1>& mask, const RComplex<T>& s1) 
{
  copymask(d.real(),mask.elem(),s1.real());
  copymask(d.imag(),mask.elem(),s1.imag());
}


// Global sum over site indices only
template<class T>
struct UnaryReturn<RComplex<T>, FnSum > {
  typedef RComplex<typename UnaryReturn<T, FnSum>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<RComplex<T>, FnSum>::Type_t
sum(const RComplex<T>& s1)
{
  typename UnaryReturn<RComplex<T>, FnSum>::Type_t  d;

  d.real() = sum(s1.real());
  d.imag() = sum(s1.imag());
  return d;
}


// Innerproduct (norm-seq) global sum = sum(tr(conj(s1)*s1))
template<class T>
struct UnaryReturn<RComplex<T>, FnSumSq > {
  typedef RScalar<typename UnaryReturn<T, FnSumSq>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<RComplex<T>, FnSumSq>::Type_t
sumsq(const RComplex<T>& s1)
{
  typename UnaryReturn<RComplex<T>, FnSumSq>::Type_t  d;

  d.elem() = sum(s1.real()*s1.real() + s1.imag()*s1.imag());
  return d;
}



//! RComplex<T> = Innerproduct(Conj(RComplex<T1>)*RComplex<T2>)
template<class T1, class T2>
struct BinaryReturn<RComplex<T1>, RComplex<T2>, FnInnerproduct > {
  typedef RComplex<typename BinaryReturn<T1, T2, FnInnerproduct>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RComplex<T1>, RComplex<T2>, FnInnerproduct>::Type_t
innerproduct(const RComplex<T1>& l, const RComplex<T2>& r)
{
  typename BinaryReturn<RComplex<T1>, RComplex<T2>, FnInnerproduct>::Type_t  d;

  d.real() = sum(l.real()*r.real() + l.imag()*r.imag());
  d.imag() = sum(l.real()*r.imag() - l.imag()*r.real());
  return d;
}


//! RScalar<T> = Innerproduct_real(Conj(RComplex<T1>)*RComplex<T1>)
template<class T1, class T2>
struct BinaryReturn<RComplex<T1>, RComplex<T2>, FnInnerproduct_real > {
  typedef RScalar<typename BinaryReturn<T1, T2, FnInnerproduct_real>::Type_t>  Type_t;
};

template<class T1, class T2>
inline typename BinaryReturn<RComplex<T1>, RComplex<T2>, FnInnerproduct_real>::Type_t
innerproduct_real(const RComplex<T1>& s1, const RComplex<T2>& s2)
{
  typename BinaryReturn<RComplex<T1>, RComplex<T2>, FnInnerproduct_real>::Type_t  d;

  d.elem() = sum(s1.real()*s2.real() + s1.imag()*s2.imag());
  return d;
}


//-----------------------------------------------------------------------------
// Broadcast operations
//! dest = 0
template<class T> 
inline
void zero(RComplex<T>& dest) 
{
  zero(dest.real());
  zero(dest.imag());
}


//! dest  = random  
template<class T, class T1, class T2>
inline void
fill_random(RComplex<T>& d, T1& seed, T2& skewed_seed, const T1& seed_mult)
{
  fill_random(d.real(), seed, skewed_seed, seed_mult);
  fill_random(d.imag(), seed, skewed_seed, seed_mult);
}


//! dest  = gaussian
/*! RComplex polar method */
template<class T>
inline void
fill_gaussian(RComplex<T>& d, RComplex<T>& r1, RComplex<T>& r2)
{
  typedef typename InternalScalar<T>::Type_t  S;

  // r1 and r2 are the input random numbers needed

  /* Stage 2: get the cos of the second number  */
  T  g_r, g_i;

  r2.real() *= S(6.283185307);
  g_r = cos(r2.real());
  g_i = sin(r2.real());
    
  /* Stage 4: get  sqrt(-2.0 * log(u1)) */
  r1.real() = sqrt(-S(2.0) * log(r1.real()));

  /* Stage 5:   g_r = sqrt(-2*log(u1))*cos(2*pi*u2) */
  /* Stage 5:   g_i = sqrt(-2*log(u1))*sin(2*pi*u2) */
  d.real() = r1.real() * g_r;
  d.imag() = r1.real() * g_i;
}




QDP_END_NAMESPACE();
