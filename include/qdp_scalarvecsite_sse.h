// -*- C++ -*-
// $Id: qdp_scalarvecsite_sse.h,v 1.5 2003-08-21 04:45:50 edwards Exp $

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
	      : "m" ((bb).elem(0,j).real()),     \
		"m" ((aa).elem(0,0).real()),     \
		"m" ((aa).elem(0,0).imag()),     \
		"m" ((aa).elem(1,0).real()),     \
		"m" ((aa).elem(1,0).imag()),     \
		"m" ((aa).elem(2,0).real()),     \
		"m" ((aa).elem(2,0).imag()));    \
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
	      : "m" ((bb).elem(0,j).imag()),     \
		"m" ((aa).elem(0,0).real()),     \
		"m" ((aa).elem(0,0).imag()),     \
		"m" ((aa).elem(1,0).real()),     \
		"m" ((aa).elem(1,0).imag()),     \
		"m" ((aa).elem(2,0).real()),     \
		"m" ((aa).elem(2,0).imag()));    \
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
	      : "m" ((bb).elem(1,j).real()),     \
		"m" ((aa).elem(0,1).real()),     \
		"m" ((aa).elem(0,1).imag()),     \
		"m" ((aa).elem(1,1).real()),     \
		"m" ((aa).elem(1,1).imag()),     \
		"m" ((aa).elem(2,1).real()),     \
		"m" ((aa).elem(2,1).imag()));    \
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
	      : "m" ((bb).elem(1,j).imag()),     \
		"m" ((aa).elem(0,1).real()),     \
		"m" ((aa).elem(0,1).imag()),     \
		"m" ((aa).elem(1,1).real()),     \
		"m" ((aa).elem(1,1).imag()),     \
		"m" ((aa).elem(2,1).real()),     \
		"m" ((aa).elem(2,1).imag()));    \
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
	      : "m" ((bb).elem(2,j).real()),     \
		"m" ((aa).elem(0,2).real()),     \
		"m" ((aa).elem(0,2).imag()),     \
		"m" ((aa).elem(1,2).real()),     \
		"m" ((aa).elem(1,2).imag()),     \
		"m" ((aa).elem(2,2).real()),     \
		"m" ((aa).elem(2,2).imag()));    \
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
	      : "m" ((bb).elem(2,j).imag()),     \
		"m" ((aa).elem(0,2).real()),     \
		"m" ((aa).elem(0,2).imag()),     \
		"m" ((aa).elem(1,2).real()),     \
		"m" ((aa).elem(1,2).imag()),     \
		"m" ((aa).elem(2,2).real()),     \
		"m" ((aa).elem(2,2).imag()));    \
__asm__ __volatile__ (                    \
              "movaps %%xmm0,%0\n\t"      \
              "movaps %%xmm1,%1\n\t"      \
              "movaps %%xmm2,%2\n\t"      \
              "movaps %%xmm3,%3\n\t"      \
              "movaps %%xmm4,%4\n\t"      \
              "movaps %%xmm5,%5\n\t"      \
	      : "=m" ((cc).elem(0,j).real()),    \
		"=m" ((cc).elem(0,j).imag()),    \
		"=m" ((cc).elem(1,j).real()),    \
		"=m" ((cc).elem(1,j).imag()),    \
		"=m" ((cc).elem(2,j).real()),    \
		"=m" ((cc).elem(2,j).imag()));   \
}



// Optimized version of  
//    PColorMatrix<RComplexFloat,3> <- PColorMatrix<RComplexFloat,3> * PColorMatrix<RComplexFloat,3>
template<>
inline BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
  PMatrix<RComplexFloat,3,PColorMatrix>, OpMultiply>::Type_t
operator*<>(const PMatrix<RComplexFloat,3,PColorMatrix>& l, 
	    const PMatrix<RComplexFloat,3,PColorMatrix>& r)
{
  BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
    PMatrix<RComplexFloat,3,PColorMatrix>, OpMultiply>::Type_t  d;

  _inline_ssevec_mult_su3_nn(d,l,r,0);
  _inline_ssevec_mult_su3_nn(d,l,r,1);
  _inline_ssevec_mult_su3_nn(d,l,r,2);

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
// cout << "call single site QDP_M_eq_M_times_M" << endl;

  const LatticeColorMatrix& l = static_cast<const LatticeColorMatrix&>(rhs.expression().left());
  const LatticeColorMatrix& r = static_cast<const LatticeColorMatrix&>(rhs.expression().right());

  const int istart = s.start() >> INNER_LEN;
  const int iend   = s.end()   >> INNER_LEN;

  for(int i=istart; i <= iend; ++i) 
  {
    _inline_ssevec_mult_su3_nn(d.elem(i).elem(),l.elem(i).elem(),r.elem(i).elem(),0);
    _inline_ssevec_mult_su3_nn(d.elem(i).elem(),l.elem(i).elem(),r.elem(i).elem(),1);
    _inline_ssevec_mult_su3_nn(d.elem(i).elem(),l.elem(i).elem(),r.elem(i).elem(),2);
  }
}
#endif





/*! @} */   // end of group optimizations

QDP_END_NAMESPACE();

#endif  // defined(__GNUC__)

#endif
