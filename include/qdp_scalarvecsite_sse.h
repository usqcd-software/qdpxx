// -*- C++ -*-
// $Id: qdp_scalarvecsite_sse.h,v 1.12 2003-09-02 03:02:12 edwards Exp $

/*! @file
 * @brief Intel SSE optimizations
 *
 * SSE optimizations of basic operations
 */

#ifndef QDP_SCALARVECSITE_SSE_H
#define QDP_SCALARVECSITE_SSE_H

// These SSE asm instructions are only supported under GCC/G++
#if defined(__GNUC__)

QDP_BEGIN_NAMESPACE(QDP);

/*! @defgroup optimizations  Optimizations
 *
 * Optimizations for basic QDP operations
 *
 * @{
 */

// Use this def just to safe some typing later on in the file
#define ILatticeFloat  ILattice<float,4>
#define RComplexFloat  RComplex<ILattice<float,4> >


typedef float v4sf __attribute__ ((aligned (16),mode(V4SF)));


#if 0
// NOTE: the   operator+(v4sf,v4sf) first exists in gcc 3.3.X, not 3.2.X

// v4sf + v4sf
inline v4sf
operator+(v4sf l, v4sf r)
{
  v4sf tmp = __builtin_ia32_addps(l, r);
  return tmp;
}


// v4sf - v4sf
inline v4sf
operator-(v4sf l, v4sf r)
{
  return __builtin_ia32_subps(l, r);
}


// v4sf * v4sf
inline v4sf
operator*(v4sf l, v4sf r)
{
  return __builtin_ia32_mulps(l, r);
}


// v4sf / v4sf
inline v4sf
operator/(v4sf l, v4sf r)
{
  return __builtin_ia32_divps(l, r);
}
#endif





#if 1
//! Specialized Inner lattice class
/*! Uses sse  */
template<> class ILattice<float, 4>
{
public:
  typedef float  T;
  static const int N = 4;

  ILattice() {}
  ~ILattice() {}

  //---------------------------------------------------------
  //! construct dest = const
  ILattice(const WordType<float>::Type_t& rhs)
    {
      for(int i=0; i < N; ++i)
	elem(i) = rhs;
    }

  //! construct dest = rhs
  template<class T1>
  ILattice(const ILattice<T1,N>& rhs)
    {
      for(int i=0; i < N; ++i)
	elem(i) = rhs.elem(i);
    }

  //! construct dest = rhs
  template<class T1>
  ILattice(const T1& rhs)
    {
      for(int i=0; i < N; ++i)
	elem(i) = rhs;
    }


  //---------------------------------------------------------
  //! ILattice = IScalar
  /*! Set equal to an IScalar */
  template<class T1>
  inline
  ILattice& operator=(const IScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) = rhs.elem();

      return *this;
    }

  //! ILattice += IScalar
  template<class T1>
  inline
  ILattice& operator+=(const IScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) += rhs.elem();

      return *this;
    }

  //! ILattice -= IScalar
  template<class T1>
  inline
  ILattice& operator-=(const IScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) -= rhs.elem();

      return *this;
    }

  //! ILattice *= IScalar
  template<class T1>
  inline
  ILattice& operator*=(const IScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) *= rhs.elem();

      return *this;
    }

  //! ILattice /= IScalar
  template<class T1>
  inline
  ILattice& operator/=(const IScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) /= rhs.elem();

      return *this;
    }


  //---------------------------------------------------------
  //! ILattice = ILattice
  /*! Set equal to another ILattice */
  inline
  ILattice& operator=(const ILattice& rhs) 
    {
      F.v = rhs.F.v;
      return *this;
    }

  //! ILattice += ILattice
  inline
  ILattice& operator+=(const ILattice& rhs) 
    {
      F.v = __builtin_ia32_addps(F.v, rhs.F.v);
      return *this;
    }

  //! ILattice -= ILattice
  inline
  ILattice& operator-=(const ILattice& rhs) 
    {
      F.v = __builtin_ia32_subps(F.v, rhs.F.v);
      return *this;
    }

  //! ILattice *= ILattice
  inline
  ILattice& operator*=(const ILattice& rhs) 
    {
      F.v = __builtin_ia32_mulps(F.v, rhs.F.v);
      return *this;
    }

  //! ILattice /= ILattice
  inline
  ILattice& operator/=(const ILattice& rhs) 
    {
      F.v = __builtin_ia32_divps(F.v, rhs.F.v);
      return *this;
    }


  //! Deep copy constructor
  ILattice(const ILattice& a)
    {
      // fprintf(stderr,"copy ILattice\n");
      F.v = a.F.v;
    }


public:
  //! The backdoor
  /*! 
   * Used by optimization routines (e.g., SSE) that need the memory address of data.
   * BTW: to make this a friend would be a real pain since functions are templatized.
   */
  inline T* data() {return F.a;}


public:
  T& elem(int i) {return F.a[i];}
  const T& elem(int i) const {return F.a[i];}

  v4sf& elem_v() {return F.v;}
  const v4sf elem_v() const {return F.v;}

private:
  // SSE attributes
  union {
    v4sf v;
    T    a[4];
  } F  QDP_ALIGN16;

};
#endif




//--------------------------------------------------------------------------------------
// Optimized version of  
//    ILatticeFloat <- ILatticeFloat + ILatticeFloat
inline BinaryReturn<ILatticeFloat, ILatticeFloat, OpAdd>::Type_t
operator+(const ILatticeFloat& l, const ILatticeFloat& r)
{
  BinaryReturn<ILatticeFloat, ILatticeFloat, OpAdd>::Type_t  d;

//  cout << "I+I" << endl;

  d.elem_v() = __builtin_ia32_addps(l.elem_v(), r.elem_v());

  return d;
}


// Optimized version of  
//    ILatticeFloat <- ILatticeFloat - ILatticeFloat
inline BinaryReturn<ILatticeFloat, ILatticeFloat, OpSubtract>::Type_t
operator-(const ILatticeFloat& l, const ILatticeFloat& r)
{
  BinaryReturn<ILatticeFloat, ILatticeFloat, OpSubtract>::Type_t  d;

//  cout << "I-I" << endl;

  d.elem_v() = __builtin_ia32_subps(l.elem_v(), r.elem_v());

  return d;
}


// Optimized version of  
//    ILatticeFloat <- ILatticeFloat * ILatticeFloat
inline BinaryReturn<ILatticeFloat, ILatticeFloat, OpMultiply>::Type_t
operator*(const ILatticeFloat& l, const ILatticeFloat& r)
{
  BinaryReturn<ILatticeFloat, ILatticeFloat, OpMultiply>::Type_t  d;

//  cout << "I*I" << endl;

  d.elem_v() = __builtin_ia32_mulps(l.elem_v(), r.elem_v());

  return d;
}


// Optimized version of  
//    ILatticeFloat <- ILatticeFloat / ILatticeFloat
inline BinaryReturn<ILatticeFloat, ILatticeFloat, OpDivide>::Type_t
operator/(const ILatticeFloat& l, const ILatticeFloat& r)
{
  BinaryReturn<ILatticeFloat, ILatticeFloat, OpDivide>::Type_t  d;

//  cout << "I/I" << endl;

  d.elem_v() = __builtin_ia32_mulps(l.elem_v(), r.elem_v());

  return d;
}




//--------------------------------------------------------------------------------------
// Optimized version of  
//    RComplexFloat <- RComplexFloat + RComplexFloat
inline BinaryReturn<RComplexFloat, RComplexFloat, OpAdd>::Type_t
operator+(const RComplexFloat& l, const RComplexFloat& r)
{
  BinaryReturn<RComplexFloat, RComplexFloat, OpAdd>::Type_t  d;

//  cout << "C+C" << endl;

  d.real().elem_v() = __builtin_ia32_addps(l.real().elem_v(), r.real().elem_v());
  d.imag().elem_v() = __builtin_ia32_addps(l.imag().elem_v(), r.imag().elem_v());

  return d;
}


// Optimized version of  
//    RComplexFloat <- RComplexFloat - RComplexFloat
inline BinaryReturn<RComplexFloat, RComplexFloat, OpSubtract>::Type_t
operator-(const RComplexFloat& l, const RComplexFloat& r)
{
  BinaryReturn<RComplexFloat, RComplexFloat, OpSubtract>::Type_t  d;

//  cout << "C-C" << endl;

  d.real().elem_v() = __builtin_ia32_subps(l.real().elem_v(), r.real().elem_v());
  d.imag().elem_v() = __builtin_ia32_subps(l.imag().elem_v(), r.imag().elem_v());

  return d;
}


// Optimized version of  
//    RComplexFloat <- RComplexFloat * RComplexFloat
inline BinaryReturn<RComplexFloat, RComplexFloat, OpMultiply>::Type_t
operator*(const RComplexFloat& l, const RComplexFloat& r)
{
  BinaryReturn<RComplexFloat, RComplexFloat, OpMultiply>::Type_t  d;

//  cout << "C*C" << endl;

  v4sf tmp1 = __builtin_ia32_mulps(l.real().elem_v(), r.real().elem_v());
  v4sf tmp2 = __builtin_ia32_mulps(l.imag().elem_v(), r.imag().elem_v());
  d.real().elem_v() = __builtin_ia32_subps(tmp1, tmp2);

  v4sf tmp3 = __builtin_ia32_mulps(l.real().elem_v(), r.imag().elem_v());
  v4sf tmp4 = __builtin_ia32_mulps(l.imag().elem_v(), r.real().elem_v());
  d.imag().elem_v() = __builtin_ia32_addps(tmp3, tmp4);

  return d;
}

// Optimized version of  
//    RComplexFloat <- adj(RComplexFloat) * RComplexFloat
inline BinaryReturn<RComplexFloat, RComplexFloat, OpAdjMultiply>::Type_t
adjMultiply(const RComplexFloat& l, const RComplexFloat& r)
{
  BinaryReturn<RComplexFloat, RComplexFloat, OpAdjMultiply>::Type_t  d;

//  cout << "adj(C)*C" << endl;

  v4sf tmp1 = __builtin_ia32_mulps(l.real().elem_v(), r.real().elem_v());
  v4sf tmp2 = __builtin_ia32_mulps(l.imag().elem_v(), r.imag().elem_v());
  d.real().elem_v() = __builtin_ia32_addps(tmp1, tmp2);

  v4sf tmp3 = __builtin_ia32_mulps(l.real().elem_v(), r.imag().elem_v());
  v4sf tmp4 = __builtin_ia32_mulps(l.imag().elem_v(), r.real().elem_v());
  d.imag().elem_v() = __builtin_ia32_subps(tmp3, tmp4);

  return d;
}

// Optimized  RComplex*adj(RComplex)
inline BinaryReturn<RComplexFloat, RComplexFloat, OpMultiplyAdj>::Type_t
multiplyAdj(const RComplexFloat& l, const RComplexFloat& r)
{
  BinaryReturn<RComplexFloat, RComplexFloat, OpMultiplyAdj>::Type_t  d;

  v4sf tmp1 = __builtin_ia32_mulps(l.real().elem_v(), r.real().elem_v());
  v4sf tmp2 = __builtin_ia32_mulps(l.imag().elem_v(), r.imag().elem_v());
  d.real().elem_v() = __builtin_ia32_addps(tmp1, tmp2);

  v4sf tmp3 = __builtin_ia32_mulps(l.imag().elem_v(), r.real().elem_v());
  v4sf tmp4 = __builtin_ia32_mulps(l.real().elem_v(), r.imag().elem_v());
  d.imag().elem_v() = __builtin_ia32_subps(tmp3, tmp4);

  return d;
}

// Optimized  adj(RComplex)*adj(RComplex)
inline BinaryReturn<RComplexFloat, RComplexFloat, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(const RComplexFloat& l, const RComplexFloat& r)
{
  BinaryReturn<RComplexFloat, RComplexFloat, OpAdjMultiplyAdj>::Type_t  d;

  typedef struct
  {
    unsigned int c[4];
  } sse_mask __attribute__ ((aligned (16)));
  
  static sse_mask _sse_sgn __attribute__ ((unused)) ={0x80000000, 0x80000000, 0x80000000, 0x80000000};

  v4sf tmp1 = __builtin_ia32_mulps(l.real().elem_v(), r.real().elem_v());
  v4sf tmp2 = __builtin_ia32_mulps(l.imag().elem_v(), r.imag().elem_v());
  d.real().elem_v() = __builtin_ia32_subps(tmp1, tmp2);

  v4sf tmp3 = __builtin_ia32_mulps(l.real().elem_v(), r.imag().elem_v());
  v4sf tmp4 = __builtin_ia32_mulps(l.imag().elem_v(), r.real().elem_v());
  v4sf tmp5 = __builtin_ia32_addps(tmp3, tmp4);
//  d.imag().elem_v() = __builtin_ia32_xorps(tmp5, _sse_sgn.v);
  v4sf tmp6 = __builtin_ia32_loadaps((float*)&_sse_sgn);
  d.imag().elem_v() = __builtin_ia32_xorps(tmp5, tmp6);

  return d;
}







#if 1

//--------------------------------------------------------------------------------------
#if 0
#define PREFETCH(addr)  __asm__ __volatile__("prefetcht0 %0"::"m"(*(addr)))
#else
#define PREFETCH(addr)
#endif

#define _inline_ssevec_mult_su3_nn(cc,aa,bb,j) \
{ \
__asm__ __volatile__ (                    \
              "movlps %0, %%xmm0 \n\t"    \
              "movaps %0,%%xmm0\n\t"      \
              "movaps %%xmm0,%%xmm1\n\t"  \
              "mulps  %2,%%xmm1\n\t"      \
              "movaps %%xmm0,%%xmm2\n\t"  \
              "mulps  %3,%%xmm2\n\t"      \
              "movaps %%xmm0,%%xmm3\n\t"  \
              "mulps  %4,%%xmm3\n\t"      \
              "movaps %%xmm0,%%xmm4\n\t"  \
              "mulps  %5,%%xmm4\n\t"      \
              "movaps %%xmm0,%%xmm5\n\t"  \
              "mulps  %6,%%xmm5\n\t"      \
              "mulps  %1,%%xmm0\n\t"      \
	      :                           \
	      : "m" (*(bb+0+8*j)),     \
		"m" (*(aa+0)),     \
		"m" (*(aa+4)),     \
		"m" (*(aa+24)),     \
		"m" (*(aa+28)),     \
		"m" (*(aa+48)),     \
		"m" (*(aa+52)));    \
__asm__ __volatile__ (                    \
              "movaps %0,%%xmm6\n\t"      \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %2,%%xmm7\n\t"      \
              "subps  %%xmm7,%%xmm0\n\t"  \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %1,%%xmm7\n\t"      \
              "addps  %%xmm7,%%xmm1\n\t"  \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %4,%%xmm7\n\t"      \
              "subps  %%xmm7,%%xmm2\n\t"  \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %3,%%xmm7\n\t"      \
              "addps  %%xmm7,%%xmm3\n\t"  \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %6,%%xmm7\n\t"      \
              "subps  %%xmm7,%%xmm4\n\t"  \
              "mulps  %5,%%xmm6\n\t"      \
              "addps  %%xmm6,%%xmm5\n\t"  \
	      :                           \
	      : "m" (*(bb+4+8*j)),     \
		"m" (*(aa+0)),     \
		"m" (*(aa+4)),     \
		"m" (*(aa+24)),     \
		"m" (*(aa+28)),     \
		"m" (*(aa+48)),     \
		"m" (*(aa+52)));    \
__asm__ __volatile__ (                    \
              "movaps %0,%%xmm6\n\t"      \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %1,%%xmm7\n\t"      \
              "addps  %%xmm7,%%xmm0\n\t"  \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %2,%%xmm7\n\t"      \
              "addps  %%xmm7,%%xmm1\n\t"  \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %3,%%xmm7\n\t"      \
              "addps  %%xmm7,%%xmm2\n\t"  \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %4,%%xmm7\n\t"      \
              "addps  %%xmm7,%%xmm3\n\t"  \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %5,%%xmm7\n\t"      \
              "addps  %%xmm7,%%xmm4\n\t"  \
              "mulps  %6,%%xmm6\n\t"      \
              "addps  %%xmm6,%%xmm5\n\t"  \
	      :                           \
	      : "m" (*(bb+24+8*j)),     \
		"m" (*(aa+8)),     \
		"m" (*(aa+12)),     \
		"m" (*(aa+32)),     \
		"m" (*(aa+36)),     \
		"m" (*(aa+56)),     \
		"m" (*(aa+60)));    \
__asm__ __volatile__ (                    \
              "movaps %0,%%xmm6\n\t"      \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %2,%%xmm7\n\t"      \
              "subps  %%xmm7,%%xmm0\n\t"  \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %1,%%xmm7\n\t"      \
              "addps  %%xmm7,%%xmm1\n\t"  \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %4,%%xmm7\n\t"      \
              "subps  %%xmm7,%%xmm2\n\t"  \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %3,%%xmm7\n\t"      \
              "addps  %%xmm7,%%xmm3\n\t"  \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %6,%%xmm7\n\t"      \
              "subps  %%xmm7,%%xmm4\n\t"  \
              "mulps  %5,%%xmm6\n\t"      \
              "addps  %%xmm6,%%xmm5\n\t"  \
	      :                           \
	      : "m" (*(bb+28+8*j)),     \
		"m" (*(aa+8)),     \
		"m" (*(aa+12)),     \
		"m" (*(aa+32)),     \
		"m" (*(aa+36)),     \
		"m" (*(aa+56)),     \
		"m" (*(aa+60)));    \
__asm__ __volatile__ (                    \
              "movaps %0,%%xmm6\n\t"      \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %1,%%xmm7\n\t"      \
              "addps  %%xmm7,%%xmm0\n\t"  \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %2,%%xmm7\n\t"      \
              "addps  %%xmm7,%%xmm1\n\t"  \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %3,%%xmm7\n\t"      \
              "addps  %%xmm7,%%xmm2\n\t"  \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %4,%%xmm7\n\t"      \
              "addps  %%xmm7,%%xmm3\n\t"  \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %5,%%xmm7\n\t"      \
              "addps  %%xmm7,%%xmm4\n\t"  \
              "mulps  %6,%%xmm6\n\t"      \
              "addps  %%xmm6,%%xmm5\n\t"  \
	      :                           \
	      : "m" (*(bb+48+8*j)),     \
		"m" (*(aa+16)),     \
		"m" (*(aa+20)),     \
		"m" (*(aa+40)),     \
		"m" (*(aa+44)),     \
		"m" (*(aa+64)),     \
		"m" (*(aa+68)));    \
__asm__ __volatile__ (                    \
              "movaps %0,%%xmm6\n\t"      \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %2,%%xmm7\n\t"      \
              "subps  %%xmm7,%%xmm0\n\t"  \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %1,%%xmm7\n\t"      \
              "addps  %%xmm7,%%xmm1\n\t"  \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %4,%%xmm7\n\t"      \
              "subps  %%xmm7,%%xmm2\n\t"  \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %3,%%xmm7\n\t"      \
              "addps  %%xmm7,%%xmm3\n\t"  \
              "movaps %%xmm6,%%xmm7\n\t"  \
              "mulps  %6,%%xmm7\n\t"      \
              "subps  %%xmm7,%%xmm4\n\t"  \
              "mulps  %5,%%xmm6\n\t"      \
              "addps  %%xmm6,%%xmm5\n\t"  \
	      :                           \
	      : "m" (*(bb+52+8*j)),     \
		"m" (*(aa+16)),     \
		"m" (*(aa+20)),     \
		"m" (*(aa+40)),     \
		"m" (*(aa+44)),     \
		"m" (*(aa+64)),     \
		"m" (*(aa+68)));    \
__asm__ __volatile__ (                    \
              "movaps %%xmm0,%0\n\t"      \
              "movaps %%xmm1,%1\n\t"      \
              "movaps %%xmm2,%2\n\t"      \
              "movaps %%xmm3,%3\n\t"      \
              "movaps %%xmm4,%4\n\t"      \
              "movaps %%xmm5,%5\n\t"      \
	      : "=m" (*(cc+0+8*j)),    \
		"=m" (*(cc+4+8*j)),    \
		"=m" (*(cc+24+8*j)),    \
		"=m" (*(cc+28+8*j)),    \
		"=m" (*(cc+48+8*j)),    \
		"=m" (*(cc+52+8*j)));   \
}



// Optimized version of  
//    PColorMatrix<RComplexFloat,3> <- PColorMatrix<RComplexFloat,3> * PColorMatrix<RComplexFloat,3>
inline BinaryReturn<PColorMatrix<RComplexFloat,3>, 
  PColorMatrix<RComplexFloat,3>, OpMultiply>::Type_t
operator*(const PColorMatrix<RComplexFloat,3>& l, 
	  const PColorMatrix<RComplexFloat,3>& r)
{
  BinaryReturn<PColorMatrix<RComplexFloat,3>, 
    PColorMatrix<RComplexFloat,3>, OpMultiply>::Type_t  d;

  float *dd = (float*)&d;
  float *ll = (float*)&l;
  float *rr = (float*)&r;

  _inline_ssevec_mult_su3_nn(dd,ll,rr,0);
  _inline_ssevec_mult_su3_nn(dd,ll,rr,1);
  _inline_ssevec_mult_su3_nn(dd,ll,rr,2);

  return d;
}


#if 0

// Specialization to optimize the case   
//    LatticeColorMatrix = LatticeColorMatrix * LatticeColorMatrix
template<>
void evaluate(OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >& d, 
	      const OpAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	      Reference<QDPType<PScalar<PColorMatrix<RComplexFloat, 3> >, 
	      OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > > >, 
	      Reference<QDPType<PScalar<PColorMatrix<RComplexFloat, 3> >, 
	      OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > > > >,
	      OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > >& rhs,
	      const OrderedSubset& s)
{
//  cout << "call single site QDP_M_eq_M_times_M" << endl;

  const LatticeColorMatrix& l = static_cast<const LatticeColorMatrix&>(rhs.expression().left());
  const LatticeColorMatrix& r = static_cast<const LatticeColorMatrix&>(rhs.expression().right());

  const int istart = s.start() >> INNER_LOG;
  const int iend   = s.end()   >> INNER_LOG;

  for(int i=istart; i <= iend; ++i) 
  {
    float *dd = (float*)&(d.elem(i).elem());
    float *ll = (float*)&(l.elem(i).elem());
    float *rr = (float*)&(r.elem(i).elem());

    _inline_ssevec_mult_su3_nn(dd,ll,rr,0);
    _inline_ssevec_mult_su3_nn(dd,ll,rr,1);
    _inline_ssevec_mult_su3_nn(dd,ll,rr,2);
  }
}
#endif


#endif



/*! @} */   // end of group optimizations

QDP_END_NAMESPACE();

#endif  // defined(__GNUC__)

#endif
