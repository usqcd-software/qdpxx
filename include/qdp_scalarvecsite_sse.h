// -*- C++ -*-
// $Id: qdp_scalarvecsite_sse.h,v 1.9 2003-08-21 06:46:54 edwards Exp $

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
#define RComplexFloat  RComplex<ILattice<float,4> >


#if 0
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
#if 0
  //! dest = const
  /*! Fill with an integer constant. Will be promoted to underlying word type */
  inline
  ILattice& operator=(const typename WordType<T>::Type_t& rhs)
    {
      for(int i=0; i < N; ++i)
	elem(i) = rhs;

      return *this;
    }
#endif

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

  //! ILattice %= IScalar
  template<class T1>
  inline
  ILattice& operator%=(const IScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) %= rhs.elem();

      return *this;
    }

  //! ILattice |= IScalar
  template<class T1>
  inline
  ILattice& operator|=(const IScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) |= rhs.elem();

      return *this;
    }

  //! ILattice &= IScalar
  template<class T1>
  inline
  ILattice& operator&=(const IScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) &= rhs.elem();

      return *this;
    }

  //! ILattice ^= IScalar
  template<class T1>
  inline
  ILattice& operator^=(const IScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) ^= rhs.elem();

      return *this;
    }

  //! ILattice <<= IScalar
  template<class T1>
  inline
  ILattice& operator<<=(const IScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) <<= rhs.elem();

      return *this;
    }

  //! ILattice >>= IScalar
  template<class T1>
  inline
  ILattice& operator>>=(const IScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) >>= rhs.elem();

      return *this;
    }

  //---------------------------------------------------------
  //! ILattice = ILattice
  /*! Set equal to another ILattice */
  inline
  ILattice& operator=(const ILattice& rhs) 
    {
      F = rhs.F;

      return *this;
    }

  //! ILattice += ILattice
  inline
  ILattice& operator+=(const ILattice& rhs) 
    {
      F += rhs.F;

      return *this;
    }

  //! ILattice -= ILattice
  template<class T1>
  inline
  ILattice& operator-=(const ILattice& rhs) 
    {
      F -= rhs.F;

      return *this;
    }

  //! ILattice *= ILattice
  template<class T1>
  inline
  ILattice& operator*=(const ILattice<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) *= rhs.elem(i);

      return *this;
    }

  //! ILattice /= ILattice
  template<class T1>
  inline
  ILattice& operator/=(const ILattice<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) /= rhs.elem(i);

      return *this;
    }

  //! ILattice %= ILattice
  template<class T1>
  inline
  ILattice& operator%=(const ILattice<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) %= rhs.elem(i);

      return *this;
    }

  //! ILattice |= ILattice
  template<class T1>
  inline
  ILattice& operator|=(const ILattice<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) |= rhs.elem(i);

      return *this;
    }

  //! ILattice &= ILattice
  template<class T1>
  inline
  ILattice& operator&=(const ILattice<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) &= rhs.elem(i);

      return *this;
    }

  //! ILattice ^= ILattice
  template<class T1>
  inline
  ILattice& operator^=(const ILattice<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) ^= rhs.elem(i);

      return *this;
    }

  //! ILattice <<= ILattice
  template<class T1>
  inline
  ILattice& operator<<=(const ILattice<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) <<= rhs.elem(i);

      return *this;
    }

  //! ILattice >>= ILattice
  template<class T1>
  inline
  ILattice& operator>>=(const ILattice<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) >>= rhs.elem(i);

      return *this;
    }

#if 0
  // NOTE: intentially avoid defining a copy constructor - let the compiler
  // generate one via the bit copy mechanism. This effectively achieves
  // the first form of the if below (QDP_USE_ARRAY_INITIALIZER) without having
  // to use that syntax which is not strictly legal in C++.
#endif

  //! Deep copy constructor
#if defined(QDP_USE_ARRAY_INITIALIZER)
  ILattice(const ILattice& a) : F(a.F) {}
#else
  /*! This is a copy form - legal but not necessarily efficient */
  ILattice(const ILattice& a)
    {
      // fprintf(stderr,"copy ILattice\n");
      F = a.F;
    }
#endif

public:
  //! The backdoor
  /*! 
   * Used by optimization routines (e.g., SSE) that need the memory address of data.
   * BTW: to make this a friend would be a real pain since functions are templatized.
   */
//  inline T* data() const {return &F;}


public:
  T& elem(int i) {return *(&F + i);}
  const T& elem(int i) const {return *(&F + i);}

private:
  // SSE attributes
  typedef float v4sf __attribute__ ((mode(V4SF)));
  v4sf F;

} QDP_ALIGN16;   // possibly force alignment
#endif




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


#if 1

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

  const int istart = s.start() >> INNER_LEN;
  const int iend   = s.end()   >> INNER_LEN;

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





/*! @} */   // end of group optimizations

QDP_END_NAMESPACE();

#endif  // defined(__GNUC__)

#endif
