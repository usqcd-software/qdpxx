// $Id: linalg1.cc,v 1.9 2003-08-05 03:58:22 edwards Exp $

#include <time.h>

#include "qdp.h"
#include "linalg.h"

namespace QDP {


#if 0
template<class T1, class T2, template<class,int> class C>
inline typename BinaryReturn<PScalar<T1>, PVector<T2,2,C>, OpMultiply>::Type_t
operator*(const PScalar<T1>& l, const PVector<T2,2,C>& r)
{
  typename BinaryReturn<PScalar<T1>, PVector<T2,2,C>, OpMultiply>::Type_t  d;

  d.elem(0) = l.elem() * r.elem(0);
  d.elem(1) = l.elem() * r.elem(1);
  return d;
}
#endif

#if 0
template<class T1, class T2, template<class,int> class C1, template<class,int> class C2>
inline typename BinaryReturn<PMatrix<T1,3,C1>, PVector<T2,3,C2>, OpMultiply>::Type_t
operator*(const PMatrix<T1,3,C1>& l, const PVector<T2,3,C2>& r)
{
  typename BinaryReturn<PMatrix<T1,3,C1>, PVector<T2,3,C2>, OpMultiply>::Type_t  d;

  d.elem(0) = l.elem(0,0)*r.elem(0) + l.elem(0,1)*r.elem(1) + l.elem(0,2)*r.elem(2);
  d.elem(1) = l.elem(1,0)*r.elem(0) + l.elem(1,1)*r.elem(1) + l.elem(1,2)*r.elem(2);
  d.elem(2) = l.elem(2,0)*r.elem(0) + l.elem(2,1)*r.elem(1) + l.elem(2,2)*r.elem(2);

  return d;
}
#endif

#if 0
template<>
inline BinaryReturn<PMatrix<RComplex<float>,3,PColorMatrix>, 
  PMatrix<RComplex<float>,3,PColorMatrix>, OpMultiply>::Type_t
operator*<>(const PMatrix<RComplex<float>,3,PColorMatrix>& l, 
	    const PMatrix<RComplex<float>,3,PColorMatrix>& r)
{
  BinaryReturn<PMatrix<RComplex<float>,3,PColorMatrix>, 
    PMatrix<RComplex<float>,3,PColorMatrix>, OpMultiply>::Type_t  d;

  d.elem(0,0) = l.elem(0,0)*r.elem(0,0) + l.elem(0,1)*r.elem(1,0) + l.elem(0,2)*r.elem(2,0);
  d.elem(1,0) = l.elem(1,0)*r.elem(0,0) + l.elem(1,1)*r.elem(1,0) + l.elem(1,2)*r.elem(2,0);
  d.elem(2,0) = l.elem(2,0)*r.elem(0,0) + l.elem(2,1)*r.elem(1,0) + l.elem(2,2)*r.elem(2,0);
  
  d.elem(0,1) = l.elem(0,0)*r.elem(0,1) + l.elem(0,1)*r.elem(1,1) + l.elem(0,2)*r.elem(2,1);
  d.elem(1,1) = l.elem(1,0)*r.elem(0,1) + l.elem(1,1)*r.elem(1,1) + l.elem(1,2)*r.elem(2,1);
  d.elem(2,1) = l.elem(2,0)*r.elem(0,1) + l.elem(2,1)*r.elem(1,1) + l.elem(2,2)*r.elem(2,1);
  
  d.elem(0,2) = l.elem(0,0)*r.elem(0,2) + l.elem(0,1)*r.elem(1,2) + l.elem(0,2)*r.elem(2,2);
  d.elem(1,2) = l.elem(1,0)*r.elem(0,2) + l.elem(1,1)*r.elem(1,2) + l.elem(1,2)*r.elem(2,2);
  d.elem(2,2) = l.elem(2,0)*r.elem(0,2) + l.elem(2,1)*r.elem(1,2) + l.elem(2,2)*r.elem(2,2);

  return d;
}
#endif


#if 0
template<>
inline BinaryReturn<PMatrix<RComplex<float>,3,PColorMatrix>, 
  PMatrix<RComplex<float>,3,PColorMatrix>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj<>(const PMatrix<RComplex<float>,3,PColorMatrix>& l, 
		 const PMatrix<RComplex<float>,3,PColorMatrix>& r)
{
  BinaryReturn<PMatrix<RComplex<float>,3,PColorMatrix>, 
    PMatrix<RComplex<float>,3,PColorMatrix>, OpAdjMultiplyAdj>::Type_t  d;

//cerr << "primmat: adjMultiplyAdj" << endl;

  d.elem(0,0) = adjMultiplyAdj(l.elem(0,0),r.elem(0,0)) 
              + adjMultiplyAdj(l.elem(1,0),r.elem(0,1))
              + adjMultiplyAdj(l.elem(2,0),r.elem(0,2));
  d.elem(1,0) = adjMultiplyAdj(l.elem(0,1),r.elem(0,0))
              + adjMultiplyAdj(l.elem(1,1),r.elem(0,1))
              + adjMultiplyAdj(l.elem(2,1),r.elem(0,2));
  d.elem(2,0) = adjMultiplyAdj(l.elem(0,2),r.elem(0,0))
              + adjMultiplyAdj(l.elem(1,2),r.elem(0,1))
              + adjMultiplyAdj(l.elem(2,2),r.elem(0,2));
  
  d.elem(0,1) = adjMultiplyAdj(l.elem(0,0),r.elem(1,0))
              + adjMultiplyAdj(l.elem(1,0),r.elem(1,1))
              + adjMultiplyAdj(l.elem(2,0),r.elem(1,2));
  d.elem(1,1) = adjMultiplyAdj(l.elem(0,1),r.elem(1,0))
              + adjMultiplyAdj(l.elem(1,1),r.elem(1,1))
              + adjMultiplyAdj(l.elem(2,1),r.elem(1,2));
  d.elem(2,1) = adjMultiplyAdj(l.elem(0,2),r.elem(1,0))
              + adjMultiplyAdj(l.elem(1,2),r.elem(1,1))
              + adjMultiplyAdj(l.elem(2,2),r.elem(1,2));
  
  d.elem(0,2) = adjMultiplyAdj(l.elem(0,0),r.elem(2,0))
              + adjMultiplyAdj(l.elem(1,0),r.elem(2,1))
              + adjMultiplyAdj(l.elem(2,0),r.elem(2,2));
  d.elem(1,2) = adjMultiplyAdj(l.elem(0,1),r.elem(2,0))
              + adjMultiplyAdj(l.elem(1,1),r.elem(2,1))
              + adjMultiplyAdj(l.elem(2,1),r.elem(2,2));
  d.elem(2,2) = adjMultiplyAdj(l.elem(0,2),r.elem(2,0))
              + adjMultiplyAdj(l.elem(1,2),r.elem(2,1))
              + adjMultiplyAdj(l.elem(2,2),r.elem(2,2));

  return d;
}
#endif


#if 0
typedef struct
{
   unsigned int c1,c2,c3,c4;
} sse_mask __attribute__ ((aligned (16)));

static sse_mask _sse_sgn13 __attribute__ ((unused)) ={0x80000000, 0x00000000, 0x80000000, 0x00000000};
static sse_mask _sse_sgn24 __attribute__ ((unused)) ={0x00000000, 0x80000000, 0x00000000, 0x80000000};
static sse_mask _sse_sgn3 __attribute__  ((unused)) ={0x00000000, 0x00000000, 0x80000000, 0x00000000};
static sse_mask _sse_sgn4 __attribute__  ((unused)) ={0x00000000, 0x00000000, 0x00000000, 0x80000000};


template<>
inline BinaryReturn<PMatrix<RComplex<float>,3,PColorMatrix>, 
  PMatrix<RComplex<float>,3,PColorMatrix>, OpMultiply>::Type_t
operator*<>(const PMatrix<RComplex<float>,3,PColorMatrix>& l, 
	    const PMatrix<RComplex<float>,3,PColorMatrix>& r)
{
  BinaryReturn<PMatrix<RComplex<float>,3,PColorMatrix>, 
    PMatrix<RComplex<float>,3,PColorMatrix>, OpMultiply>::Type_t  d;

  __asm__ __volatile__ ("movlps %0, %%xmm0 \n\t"
			"movlps %1, %%xmm1 \n\t"
			"movlps %2, %%xmm2 \n\t"
			"movhps %3, %%xmm0 \n\t"
			"movhps %4, %%xmm1 \n\t"
			"movhps %5, %%xmm2"
			:
			:
			"m" ((r).elem(0,0)),
			"m" ((r).elem(1,0)),
			"m" ((r).elem(2,0)),
			"m" ((r).elem(0,1)),
			"m" ((r).elem(1,1)),
			"m" ((r).elem(2,1)));
  __asm__ __volatile__ ("movss %0, %%xmm3 \n\t"
			"movss %1, %%xmm6 \n\t"
			"movss %2, %%xmm4 \n\t"
			"movss %3, %%xmm7 \n\t"
			"movss %4, %%xmm5 \n\t"
			"shufps $0x00, %%xmm3, %%xmm3 \n\t"
			"shufps $0x00, %%xmm6, %%xmm6 \n\t"
			"shufps $0x00, %%xmm4, %%xmm4 \n\t"
			"mulps %%xmm0, %%xmm3 \n\t"
			"shufps $0x00, %%xmm7, %%xmm7 \n\t"
			"mulps %%xmm1, %%xmm6 \n\t"
			"shufps $0x00, %%xmm5, %%xmm5 \n\t"
			"mulps %%xmm0, %%xmm4 \n\t"
			"addps %%xmm6, %%xmm3 \n\t"
			"mulps %%xmm2, %%xmm7 \n\t"
			"mulps %%xmm0, %%xmm5 \n\t"
			"addps %%xmm7, %%xmm4 \n\t"
			"movss %5, %%xmm6"
			:
			:
			"m" ((l).elem(0,0).real()),
			"m" ((l).elem(0,1).real()),
			"m" ((l).elem(1,0).real()),
			"m" ((l).elem(1,2).real()),
			"m" ((l).elem(2,0).real()),
			"m" ((l).elem(2,1).real()));
  __asm__ __volatile__ ("movss %0, %%xmm7 \n\t"
			"shufps $0x00, %%xmm6, %%xmm6 \n\t"
			"shufps $0x00, %%xmm7, %%xmm7 \n\t"
			"mulps %%xmm1, %%xmm6 \n\t"
			"mulps %%xmm2, %%xmm7 \n\t"
			"addps %%xmm6, %%xmm5 \n\t"
			"addps %%xmm7, %%xmm3 \n\t"
			"movss %1, %%xmm6 \n\t"
			"movss %2, %%xmm7 \n\t"
			"shufps $0x00, %%xmm6, %%xmm6 \n\t"
			"shufps $0x00, %%xmm7, %%xmm7 \n\t"
			"mulps %%xmm1, %%xmm6 \n\t"
			"mulps %%xmm2, %%xmm7 \n\t"
			"addps %%xmm6, %%xmm4 \n\t"
			"addps %%xmm7, %%xmm5 \n\t"
			"movss %3, %%xmm6 \n\t"
			"movss %4, %%xmm7 \n\t"
			"shufps $0xb1, %%xmm0, %%xmm0 \n\t"
			"shufps $0xb1, %%xmm1, %%xmm1 \n\t"
			"shufps $0xb1, %%xmm2, %%xmm2 \n\t"
			"shufps $0x00, %%xmm6, %%xmm6 \n\t"
			"shufps $0x00, %%xmm7, %%xmm7 \n\t"
			"xorps %5, %%xmm0"
			:
			:
			"m" ((l).elem(0,2).real()),
			"m" ((l).elem(1,1).real()),
			"m" ((l).elem(2,2).real()),
			"m" ((l).elem(0,0).imag()),
			"m" ((l).elem(1,1).imag()),
			"m" (_sse_sgn13));
  __asm__ __volatile__ ("xorps %0, %%xmm1 \n\t"
			"xorps %1, %%xmm2 \n\t"
			"mulps %%xmm0, %%xmm6 \n\t"
			"mulps %%xmm1, %%xmm7 \n\t"
			"addps %%xmm6, %%xmm3 \n\t"
			"addps %%xmm7, %%xmm4 \n\t"
			"movss %2, %%xmm6 \n\t"
			"movss %3, %%xmm7 \n\t"
			"shufps $0x00, %%xmm6, %%xmm6 \n\t"
			"shufps $0x00, %%xmm7, %%xmm7 \n\t"
			"mulps %%xmm2, %%xmm6 \n\t"
			"mulps %%xmm0, %%xmm7 \n\t"
			"addps %%xmm6, %%xmm5 \n\t"
			"addps %%xmm7, %%xmm4 \n\t"
			"movss %4, %%xmm6 \n\t"
			"movss %5, %%xmm7"
			:
			:
			"m" (_sse_sgn13),
			"m" (_sse_sgn13),
			"m" ((l).elem(2,2).imag()),
			"m" ((l).elem(1,0).imag()),
			"m" ((l).elem(0,1).imag()),
			"m" ((l).elem(2,0).imag()));
  __asm__ __volatile__ ("shufps $0x00, %%xmm6, %%xmm6 \n\t"
			"shufps $0x00, %%xmm7, %%xmm7 \n\t"
			"mulps %%xmm1, %%xmm6 \n\t"
			"mulps %%xmm0, %%xmm7 \n\t"
			"addps %%xmm6, %%xmm3 \n\t"
			"addps %%xmm7, %%xmm5 \n\t"
			"movss %0, %%xmm0 \n\t"
			"movss %1, %%xmm6 \n\t"
			"movss %2, %%xmm7 \n\t"
			"shufps $0x00, %%xmm0, %%xmm0 \n\t"
			"shufps $0x00, %%xmm6, %%xmm6 \n\t"
			"shufps $0x00, %%xmm7, %%xmm7 \n\t"
			"mulps %%xmm2, %%xmm0 \n\t"
			"mulps %%xmm1, %%xmm6 \n\t"
			"mulps %%xmm2, %%xmm7 \n\t"
			"addps %%xmm0, %%xmm3 \n\t"
			"addps %%xmm6, %%xmm5 \n\t"
			"addps %%xmm7, %%xmm4 \n\t"
			:
			:
			"m" ((l).elem(0,2).imag()),
			"m" ((l).elem(2,1).imag()),
			"m" ((l).elem(1,2).imag()));
  __asm__ __volatile__ ("movlps %%xmm3, %0 \n\t"
			"movlps %%xmm4, %1 \n\t"
			"movlps %%xmm5, %2 \n\t"
			"movhps %%xmm3, %3 \n\t"
			"movhps %%xmm4, %4 \n\t"
			"movhps %%xmm5, %5"
			:
			"=m" ((d).elem(0,0)),
			"=m" ((d).elem(1,0)),
			"=m" ((d).elem(2,0)),
			"=m" ((d).elem(0,1)),
			"=m" ((d).elem(1,1)),
			"=m" ((d).elem(2,1)));
  __asm__ __volatile__ ("movlps %0, %%xmm0 \n\t"
			"movlps %1, %%xmm1 \n\t"
			"movlps %2, %%xmm2 \n\t"
			"shufps $0x44, %%xmm0, %%xmm0 \n\t"
			"shufps $0x44, %%xmm1, %%xmm1 \n\t"
			"shufps $0x44, %%xmm2, %%xmm2 \n\t"
			"movss %3, %%xmm3 \n\t"
			"movss %4, %%xmm7 \n\t"
			"shufps $0x00, %%xmm7, %%xmm3 \n\t"
			"movss %5, %%xmm4"
			:
			:
			"m" ((r).elem(0,2)),
			"m" ((r).elem(1,2)),
			"m" ((r).elem(2,2)),
			"m" ((l).elem(0,0).real()),
			"m" ((l).elem(1,0).real()),
			"m" ((l).elem(0,1).real()));
  __asm__ __volatile__ ("movss %0, %%xmm7 \n\t"
			"shufps $0x00, %%xmm7, %%xmm4 \n\t"
			"mulps %%xmm0, %%xmm3 \n\t"
			"mulps %%xmm1, %%xmm4 \n\t"
			"addps %%xmm4, %%xmm3 \n\t"
			"movss %1, %%xmm5 \n\t"
			"movss %2, %%xmm7 \n\t"
			"shufps $0x00, %%xmm7, %%xmm5 \n\t"
			"mulps %%xmm2, %%xmm5 \n\t"
			"addps %%xmm5, %%xmm3 \n\t"
			"shufps $0x44, %%xmm0, %%xmm1 \n\t"
			"movss %3, %%xmm7 \n\t"
			"movss %4, %%xmm6 \n\t"
			"shufps $0x00, %%xmm7, %%xmm6 \n\t"
			"mulps %%xmm1, %%xmm6 \n\t"
			"shufps $0xB1, %%xmm0, %%xmm0 \n\t"
			"xorps %5, %%xmm0"
			:
			:
			"m" ((l).elem(1,1).real()),
			"m" ((l).elem(0,2).real()),
			"m" ((l).elem(1,2).real()),
			"m" ((l).elem(2,0).real()),
			"m" ((l).elem(2,1).real()),
			"m" (_sse_sgn13));
  __asm__ __volatile__ ("shufps $0x11, %%xmm1, %%xmm1 \n\t"
			"xorps %0, %%xmm1 \n\t"
			"shufps $0xB1, %%xmm2, %%xmm2 \n\t"
			"xorps %1, %%xmm2 \n\t"
			"movss %2, %%xmm4 \n\t"
			"movss %3, %%xmm7 \n\t"
			"shufps $0x00, %%xmm7, %%xmm4 \n\t"
			"mulps %%xmm0, %%xmm4 \n\t"
			"addps %%xmm4, %%xmm3 \n\t"
			"movss %4, %%xmm5 \n\t"
			"movss %5, %%xmm7"
			:
			:
			"m" (_sse_sgn13),
			"m" (_sse_sgn13),
			"m" ((l).elem(0,0).imag()),
			"m" ((l).elem(1,0).imag()),
			"m" ((l).elem(0,1).imag()),
			"m" ((l).elem(1,1).imag()));
  __asm__ __volatile__ ("shufps $0x00, %%xmm7, %%xmm5 \n\t"
			"mulps %%xmm1, %%xmm5 \n\t"
			"addps %%xmm5, %%xmm3 \n\t"
			"movss %0, %%xmm5 \n\t"
			"movss %1, %%xmm7 \n\t"
			"shufps $0x00, %%xmm7, %%xmm5 \n\t"
			"mulps %%xmm2, %%xmm5 \n\t"
			"addps %%xmm5, %%xmm3 \n\t"
			:
			:
			"m" ((l).elem(0,2).imag()),
			"m" ((l).elem(1,2).imag()));
  __asm__ __volatile__ ("movlps %%xmm3, %0 \n\t"
			"movhps %%xmm3, %1 \n\t"
			"shufps $0x44, %%xmm0, %%xmm1 \n\t"
			:
			"=m" ((d).elem(0,2)),
			"=m" ((d).elem(1,2)));
  __asm__ __volatile__ ("movss %0, %%xmm7 \n\t"
			"movss %1, %%xmm5 \n\t"
			"shufps $0x00, %%xmm7, %%xmm5 \n\t"
			"mulps %%xmm1, %%xmm5 \n\t"
			"addps %%xmm5, %%xmm6 \n\t"
			"shufps $0xB4, %%xmm2, %%xmm2 \n\t"
			"xorps %2, %%xmm2 \n\t"
			"movlps %3, %%xmm7 \n\t"
			"shufps $0x05, %%xmm7, %%xmm7 \n\t"
			"mulps %%xmm2, %%xmm7 \n\t"
			"addps %%xmm7, %%xmm6 \n\t"
			"movaps %%xmm6, %%xmm7 \n\t"
			"shufps $0xEE, %%xmm7, %%xmm7 \n\t"
			"addps %%xmm7, %%xmm6 \n\t"
			:
			:
			"m" ((l).elem(2,0).imag()),
			"m" ((l).elem(2,1).imag()),
			"m" (_sse_sgn4),
			"m" ((l).elem(2,2)));
  __asm__ __volatile__ ("movlps %%xmm6, %0 \n\t"
			:
			"=m" ((d).elem(2,2)));

  return d;
}
#endif


#if 0
typedef struct
{
   unsigned int c1,c2,c3,c4;
} sse_mask __attribute__ ((aligned (16)));

static sse_mask _sse_sgn13 __attribute__ ((unused)) ={0x80000000, 0x00000000, 0x80000000, 0x00000000};
static sse_mask _sse_sgn24 __attribute__ ((unused)) ={0x00000000, 0x80000000, 0x00000000, 0x80000000};
static sse_mask _sse_sgn3 __attribute__  ((unused)) ={0x00000000, 0x00000000, 0x80000000, 0x00000000};
static sse_mask _sse_sgn4 __attribute__  ((unused)) ={0x00000000, 0x00000000, 0x00000000, 0x80000000};

// Specialization to optimize the case   
//    LatticeColorMatrix = LatticeColorMatrix * LatticeColorMatrix
template<>
void evaluate(OLattice<PScalar<PColorMatrix<RComplex<float>, 3> > >& d, 
	      const OpAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	      Reference<QDPType<PScalar<PColorMatrix<RComplex<float>, 3> >, 
	      OLattice<PScalar<PColorMatrix<RComplex<float>, 3> > > > >, 
	      Reference<QDPType<PScalar<PColorMatrix<RComplex<float>, 3> >, 
	      OLattice<PScalar<PColorMatrix<RComplex<float>, 3> > > > > >,
	      OLattice<PScalar<PColorMatrix<RComplex<float>, 3> > > >& rhs)
{
//  cout << "call QDP_M_eq_M_times_M" << endl;

  const LatticeColorMatrix& l = static_cast<const LatticeColorMatrix&>(rhs.expression().left());
  const LatticeColorMatrix& r = static_cast<const LatticeColorMatrix&>(rhs.expression().right());

  const int vvol = Layout::vol();
  for(int i=0; i < vvol; ++i) 
  {
    __asm__ __volatile__ ("movlps %0, %%xmm0 \n\t"
			  "movlps %1, %%xmm1 \n\t"
			  "movlps %2, %%xmm2 \n\t"
			  "movhps %3, %%xmm0 \n\t"
			  "movhps %4, %%xmm1 \n\t"
			  "movhps %5, %%xmm2"
			  :
			  :
			  "m" ((r).elem(i).elem().elem(0,0)),
			  "m" ((r).elem(i).elem().elem(1,0)),
			  "m" ((r).elem(i).elem().elem(2,0)),
			  "m" ((r).elem(i).elem().elem(0,1)),
			  "m" ((r).elem(i).elem().elem(1,1)),
			  "m" ((r).elem(i).elem().elem(2,1)));
    __asm__ __volatile__ ("movss %0, %%xmm3 \n\t"
			  "movss %1, %%xmm6 \n\t"
			  "movss %2, %%xmm4 \n\t"
			  "movss %3, %%xmm7 \n\t"
			  "movss %4, %%xmm5 \n\t"
			  "shufps $0x00, %%xmm3, %%xmm3 \n\t"
			  "shufps $0x00, %%xmm6, %%xmm6 \n\t"
			  "shufps $0x00, %%xmm4, %%xmm4 \n\t"
			  "mulps %%xmm0, %%xmm3 \n\t"
			  "shufps $0x00, %%xmm7, %%xmm7 \n\t"
			  "mulps %%xmm1, %%xmm6 \n\t"
			  "shufps $0x00, %%xmm5, %%xmm5 \n\t"
			  "mulps %%xmm0, %%xmm4 \n\t"
			  "addps %%xmm6, %%xmm3 \n\t"
			  "mulps %%xmm2, %%xmm7 \n\t"
			  "mulps %%xmm0, %%xmm5 \n\t"
			  "addps %%xmm7, %%xmm4 \n\t"
			  "movss %5, %%xmm6"
			  :
			  :
			  "m" ((l).elem(i).elem().elem(0,0).real()),
			  "m" ((l).elem(i).elem().elem(0,1).real()),
			  "m" ((l).elem(i).elem().elem(1,0).real()),
			  "m" ((l).elem(i).elem().elem(1,2).real()),
			  "m" ((l).elem(i).elem().elem(2,0).real()),
			  "m" ((l).elem(i).elem().elem(2,1).real()));
    __asm__ __volatile__ ("movss %0, %%xmm7 \n\t"
			  "shufps $0x00, %%xmm6, %%xmm6 \n\t"
			  "shufps $0x00, %%xmm7, %%xmm7 \n\t"
			  "mulps %%xmm1, %%xmm6 \n\t"
			  "mulps %%xmm2, %%xmm7 \n\t"
			  "addps %%xmm6, %%xmm5 \n\t"
			  "addps %%xmm7, %%xmm3 \n\t"
			  "movss %1, %%xmm6 \n\t"
			  "movss %2, %%xmm7 \n\t"
			  "shufps $0x00, %%xmm6, %%xmm6 \n\t"
			  "shufps $0x00, %%xmm7, %%xmm7 \n\t"
			  "mulps %%xmm1, %%xmm6 \n\t"
			  "mulps %%xmm2, %%xmm7 \n\t"
			  "addps %%xmm6, %%xmm4 \n\t"
			  "addps %%xmm7, %%xmm5 \n\t"
			  "movss %3, %%xmm6 \n\t"
			  "movss %4, %%xmm7 \n\t"
			  "shufps $0xb1, %%xmm0, %%xmm0 \n\t"
			  "shufps $0xb1, %%xmm1, %%xmm1 \n\t"
			  "shufps $0xb1, %%xmm2, %%xmm2 \n\t"
			  "shufps $0x00, %%xmm6, %%xmm6 \n\t"
			  "shufps $0x00, %%xmm7, %%xmm7 \n\t"
			  "xorps %5, %%xmm0"
			  :
			  :
			  "m" ((l).elem(i).elem().elem(0,2).real()),
			  "m" ((l).elem(i).elem().elem(1,1).real()),
			  "m" ((l).elem(i).elem().elem(2,2).real()),
			  "m" ((l).elem(i).elem().elem(0,0).imag()),
			  "m" ((l).elem(i).elem().elem(1,1).imag()),
			  "m" (_sse_sgn13));
    __asm__ __volatile__ ("xorps %0, %%xmm1 \n\t"
			  "xorps %1, %%xmm2 \n\t"
			  "mulps %%xmm0, %%xmm6 \n\t"
			  "mulps %%xmm1, %%xmm7 \n\t"
			  "addps %%xmm6, %%xmm3 \n\t"
			  "addps %%xmm7, %%xmm4 \n\t"
			  "movss %2, %%xmm6 \n\t"
			  "movss %3, %%xmm7 \n\t"
			  "shufps $0x00, %%xmm6, %%xmm6 \n\t"
			  "shufps $0x00, %%xmm7, %%xmm7 \n\t"
			  "mulps %%xmm2, %%xmm6 \n\t"
			  "mulps %%xmm0, %%xmm7 \n\t"
			  "addps %%xmm6, %%xmm5 \n\t"
			  "addps %%xmm7, %%xmm4 \n\t"
			  "movss %4, %%xmm6 \n\t"
			  "movss %5, %%xmm7"
			  :
			  :
			  "m" (_sse_sgn13),
			  "m" (_sse_sgn13),
			  "m" ((l).elem(i).elem().elem(2,2).imag()),
			  "m" ((l).elem(i).elem().elem(1,0).imag()),
			  "m" ((l).elem(i).elem().elem(0,1).imag()),
			  "m" ((l).elem(i).elem().elem(2,0).imag()));
    __asm__ __volatile__ ("shufps $0x00, %%xmm6, %%xmm6 \n\t"
			  "shufps $0x00, %%xmm7, %%xmm7 \n\t"
			  "mulps %%xmm1, %%xmm6 \n\t"
			  "mulps %%xmm0, %%xmm7 \n\t"
			  "addps %%xmm6, %%xmm3 \n\t"
			  "addps %%xmm7, %%xmm5 \n\t"
			  "movss %0, %%xmm0 \n\t"
			  "movss %1, %%xmm6 \n\t"
			  "movss %2, %%xmm7 \n\t"
			  "shufps $0x00, %%xmm0, %%xmm0 \n\t"
			  "shufps $0x00, %%xmm6, %%xmm6 \n\t"
			  "shufps $0x00, %%xmm7, %%xmm7 \n\t"
			  "mulps %%xmm2, %%xmm0 \n\t"
			  "mulps %%xmm1, %%xmm6 \n\t"
			  "mulps %%xmm2, %%xmm7 \n\t"
			  "addps %%xmm0, %%xmm3 \n\t"
			  "addps %%xmm6, %%xmm5 \n\t"
			  "addps %%xmm7, %%xmm4 \n\t"
			  :
			  :
			  "m" ((l).elem(i).elem().elem(0,2).imag()),
			  "m" ((l).elem(i).elem().elem(2,1).imag()),
			  "m" ((l).elem(i).elem().elem(1,2).imag()));
    __asm__ __volatile__ ("movlps %%xmm3, %0 \n\t"
			  "movlps %%xmm4, %1 \n\t"
			  "movlps %%xmm5, %2 \n\t"
			  "movhps %%xmm3, %3 \n\t"
			  "movhps %%xmm4, %4 \n\t"
			  "movhps %%xmm5, %5"
			  :
			  "=m" ((d).elem(i).elem().elem(0,0)),
			  "=m" ((d).elem(i).elem().elem(1,0)),
			  "=m" ((d).elem(i).elem().elem(2,0)),
			  "=m" ((d).elem(i).elem().elem(0,1)),
			  "=m" ((d).elem(i).elem().elem(1,1)),
			  "=m" ((d).elem(i).elem().elem(2,1)));
    __asm__ __volatile__ ("movlps %0, %%xmm0 \n\t"
			  "movlps %1, %%xmm1 \n\t"
			  "movlps %2, %%xmm2 \n\t"
			  "shufps $0x44, %%xmm0, %%xmm0 \n\t"
			  "shufps $0x44, %%xmm1, %%xmm1 \n\t"
			  "shufps $0x44, %%xmm2, %%xmm2 \n\t"
			  "movss %3, %%xmm3 \n\t"
			  "movss %4, %%xmm7 \n\t"
			  "shufps $0x00, %%xmm7, %%xmm3 \n\t"
			  "movss %5, %%xmm4"
			  :
			  :
			  "m" ((r).elem(i).elem().elem(0,2)),
			  "m" ((r).elem(i).elem().elem(1,2)),
			  "m" ((r).elem(i).elem().elem(2,2)),
			  "m" ((l).elem(i).elem().elem(0,0).real()),
			  "m" ((l).elem(i).elem().elem(1,0).real()),
			  "m" ((l).elem(i).elem().elem(0,1).real()));
    __asm__ __volatile__ ("movss %0, %%xmm7 \n\t"
			  "shufps $0x00, %%xmm7, %%xmm4 \n\t"
			  "mulps %%xmm0, %%xmm3 \n\t"
			  "mulps %%xmm1, %%xmm4 \n\t"
			  "addps %%xmm4, %%xmm3 \n\t"
			  "movss %1, %%xmm5 \n\t"
			  "movss %2, %%xmm7 \n\t"
			  "shufps $0x00, %%xmm7, %%xmm5 \n\t"
			  "mulps %%xmm2, %%xmm5 \n\t"
			  "addps %%xmm5, %%xmm3 \n\t"
			  "shufps $0x44, %%xmm0, %%xmm1 \n\t"
			  "movss %3, %%xmm7 \n\t"
			  "movss %4, %%xmm6 \n\t"
			  "shufps $0x00, %%xmm7, %%xmm6 \n\t"
			  "mulps %%xmm1, %%xmm6 \n\t"
			  "shufps $0xB1, %%xmm0, %%xmm0 \n\t"
			  "xorps %5, %%xmm0"
			  :
			  :
			  "m" ((l).elem(i).elem().elem(1,1).real()),
			  "m" ((l).elem(i).elem().elem(0,2).real()),
			  "m" ((l).elem(i).elem().elem(1,2).real()),
			  "m" ((l).elem(i).elem().elem(2,0).real()),
			  "m" ((l).elem(i).elem().elem(2,1).real()),
			  "m" (_sse_sgn13));
    __asm__ __volatile__ ("shufps $0x11, %%xmm1, %%xmm1 \n\t"
			  "xorps %0, %%xmm1 \n\t"
			  "shufps $0xB1, %%xmm2, %%xmm2 \n\t"
			  "xorps %1, %%xmm2 \n\t"
			  "movss %2, %%xmm4 \n\t"
			  "movss %3, %%xmm7 \n\t"
			  "shufps $0x00, %%xmm7, %%xmm4 \n\t"
			  "mulps %%xmm0, %%xmm4 \n\t"
			  "addps %%xmm4, %%xmm3 \n\t"
			  "movss %4, %%xmm5 \n\t"
			  "movss %5, %%xmm7"
			  :
			  :
			  "m" (_sse_sgn13),
			  "m" (_sse_sgn13),
			  "m" ((l).elem(i).elem().elem(0,0).imag()),
			  "m" ((l).elem(i).elem().elem(1,0).imag()),
			  "m" ((l).elem(i).elem().elem(0,1).imag()),
			  "m" ((l).elem(i).elem().elem(1,1).imag()));
    __asm__ __volatile__ ("shufps $0x00, %%xmm7, %%xmm5 \n\t"
			  "mulps %%xmm1, %%xmm5 \n\t"
			  "addps %%xmm5, %%xmm3 \n\t"
			  "movss %0, %%xmm5 \n\t"
			  "movss %1, %%xmm7 \n\t"
			  "shufps $0x00, %%xmm7, %%xmm5 \n\t"
			  "mulps %%xmm2, %%xmm5 \n\t"
			  "addps %%xmm5, %%xmm3 \n\t"
			  :
			  :
			  "m" ((l).elem(i).elem().elem(0,2).imag()),
			  "m" ((l).elem(i).elem().elem(1,2).imag()));
    __asm__ __volatile__ ("movlps %%xmm3, %0 \n\t"
			  "movhps %%xmm3, %1 \n\t"
			  "shufps $0x44, %%xmm0, %%xmm1 \n\t"
			  :
			  "=m" ((d).elem(i).elem().elem(0,2)),
			  "=m" ((d).elem(i).elem().elem(1,2)));
    __asm__ __volatile__ ("movss %0, %%xmm7 \n\t"
			  "movss %1, %%xmm5 \n\t"
			  "shufps $0x00, %%xmm7, %%xmm5 \n\t"
			  "mulps %%xmm1, %%xmm5 \n\t"
			  "addps %%xmm5, %%xmm6 \n\t"
			  "shufps $0xB4, %%xmm2, %%xmm2 \n\t"
			  "xorps %2, %%xmm2 \n\t"
			  "movlps %3, %%xmm7 \n\t"
			  "shufps $0x05, %%xmm7, %%xmm7 \n\t"
			  "mulps %%xmm2, %%xmm7 \n\t"
			  "addps %%xmm7, %%xmm6 \n\t"
			  "movaps %%xmm6, %%xmm7 \n\t"
			  "shufps $0xEE, %%xmm7, %%xmm7 \n\t"
			  "addps %%xmm7, %%xmm6 \n\t"
			  :
			  :
			  "m" ((l).elem(i).elem().elem(2,0).imag()),
			  "m" ((l).elem(i).elem().elem(2,1).imag()),
			  "m" (_sse_sgn4),
			  "m" ((l).elem(i).elem().elem(2,2)));
    __asm__ __volatile__ ("movlps %%xmm6, %0 \n\t"
			  :
			  "=m" ((d).elem(i).elem().elem(2,2)));

  }
}
#endif




#if 0
#include "jlab_sse.h"


// Specialization to optimize the case   
//    LatticeColorMatrix = LatticeColorMatrix * LatticeColorMatrix
template<>
void evaluate(OLattice<PScalar<PColorMatrix<RComplex<float>, 3> > >& d, 
	      const OpAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	      Reference<QDPType<PScalar<PColorMatrix<RComplex<float>, 3> >, 
	      OLattice<PScalar<PColorMatrix<RComplex<float>, 3> > > > >, 
	      Reference<QDPType<PScalar<PColorMatrix<RComplex<float>, 3> >, 
	      OLattice<PScalar<PColorMatrix<RComplex<float>, 3> > > > > >,
	      OLattice<PScalar<PColorMatrix<RComplex<float>, 3> > > >& rhs)
{
//  cout << "call QDP_M_eq_M_times_M" << endl;

  const LatticeColorMatrix& l = static_cast<const LatticeColorMatrix&>(rhs.expression().left());
  const LatticeColorMatrix& r = static_cast<const LatticeColorMatrix&>(rhs.expression().right());
  su3* u3_base = (su3*)((void *)(&(d.elem(0))));
  su3* u1_base = (su3*)((void *)(&(l.elem(0))));
  su3* u2_base = (su3*)((void *)(&(r.elem(0))));

#define PREFDIST 2  
#define __REP_ADD_SUB_NEG__0_rep(ttt) _sse_pair_store_up_c1_c2((*(u5)));	
#define __REP_ADD_SUB_NEG__1_rep(ttt) _sse_pair_store_up_c3_c1((*(u5)),(*(u8))); 
#define __REP_ADD_SUB_NEG__2_rep(ttt) _sse_pair_store_up_c2_c3((*(u8)));

  su3 *u0;
  su3 *u1;
  su3 *u2;
  su3 *u3;
  su3 *u4;
  su3 *u5;
  su3 *u6;
  su3 *u7;
  su3 *u8;
  sse_float _sse_sgn12 ALIGN ={-1.0f,-1.0f,1.0f,1.0f};
  sse_float _sse_sgn13 ALIGN ={-1.0f,1.0f,-1.0f,1.0f};
  sse_float _sse_sgn14 ALIGN ={-1.0f,1.0f,1.0f,-1.0f};
  sse_float _sse_sgn23 ALIGN ={1.0f,-1.0f,-1.0f,1.0f};
  sse_float _sse_sgn24 ALIGN ={1.0f,-1.0f,1.0f,-1.0f};
  sse_float _sse_sgn34 ALIGN ={1.0f,1.0f,-1.0f,-1.0f};
  sse_float _sse_sgn1234 ALIGN ={-1.0f,-1.0f,-1.0f,-1.0f};

  const int vvol = Layout::vol();
  for(int ix=0; ix < vvol; ix+=2) 
  {
    {
      _prefetch_single(u1_base + (ix+PREFDIST));
      _prefetch_single(u2_base + (ix+PREFDIST));
      _prefetch_single(u3_base + (ix+PREFDIST));
    }
    u4 = u2_base + (ix);
    _sse_pair_load_c1_c2((*(u4)));   /*load 3 colors, first two rows*/
    u3 = u1_base + (ix);
    u5 = u3_base + (ix);

    _sse_su3_multiply(*(u3));
    __REP_ADD_SUB_NEG__0_rep();
    {
      _prefetch_single(u1_base + (ix+1+PREFDIST));
      _prefetch_single(u2_base + (ix+1+PREFDIST));
      _prefetch_single(u3_base + (ix+1+PREFDIST));
    }
    u7 = u2_base + (ix+1);
    _sse_pair_load_c3_c1((*(u4)),(*(u7)));
    u6 = u1_base + (ix+1);
    u8 = u3_base + (ix+1);

    _sse_su3_multiply_3x1_2sites(*(u3),*(u6));
    __REP_ADD_SUB_NEG__1_rep();
    _sse_pair_load_c2_c3((*(u7)));
    
    _sse_su3_multiply(*(u6));
    __REP_ADD_SUB_NEG__2_rep();
  }
}
#endif


#if 0
#include "jlab_sse.h"

extern "C" {
void QDP_M_eq_M_times_M_jlab(su3* u3_base, su3* u1_base, su3* u2_base, int size);
}

// Specialization to optimize the case   
//    LatticeColorMatrix = LatticeColorMatrix * LatticeColorMatrix
template<>
void evaluate(OLattice<PScalar<PColorMatrix<RComplex<float>, 3> > >& d, 
	      const OpAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	      Reference<QDPType<PScalar<PColorMatrix<RComplex<float>, 3> >, 
	      OLattice<PScalar<PColorMatrix<RComplex<float>, 3> > > > >, 
	      Reference<QDPType<PScalar<PColorMatrix<RComplex<float>, 3> >, 
	      OLattice<PScalar<PColorMatrix<RComplex<float>, 3> > > > > >,
	      OLattice<PScalar<PColorMatrix<RComplex<float>, 3> > > >& rhs)
{
//  cout << "call QDP_M_eq_M_times_M-jlab" << endl;

  const LatticeColorMatrix& l = static_cast<const LatticeColorMatrix&>(rhs.expression().left());
  const LatticeColorMatrix& r = static_cast<const LatticeColorMatrix&>(rhs.expression().right());
  su3* u3_base = (su3*)((void *)(&(d.elem(0))));
  su3* u1_base = (su3*)((void *)(&(l.elem(0))));
  su3* u2_base = (su3*)((void *)(&(r.elem(0))));

  QDP_M_eq_M_times_M_jlab(u3_base, u1_base, u2_base, Layout::vol());
}
#endif

} // end namespace QDP


using namespace QDP;


double QDP_M_eq_M_times_M(LatticeColorMatrix& dest, 
			  const LatticeColorMatrix& s1, 
			  const LatticeColorMatrix& s2,
			  int cnt)
{
  clock_t t1 = clock();
  for (; cnt-- > 0; )
    dest = s1 * s2;
  clock_t t2 = clock();

  return double(t2-t1)/CLOCKS_PER_SEC;
//    return 2.0;
}

double QDP_M_eq_Ma_times_M(LatticeColorMatrix& dest, 
			  const LatticeColorMatrix& s1, 
			  const LatticeColorMatrix& s2,
			  int cnt)
{
  clock_t t1 = clock();
  for (; cnt-- > 0; )
    dest = adj(s1) * s2;
  clock_t t2 = clock();

  return double(t2-t1)/CLOCKS_PER_SEC;
//    return 2.0;
}

double QDP_M_eq_M_times_Ma(LatticeColorMatrix& dest, 
			   const LatticeColorMatrix& s1, 
			   const LatticeColorMatrix& s2,
			   int cnt)
{
  clock_t t1 = clock();
  for (; cnt-- > 0; )
    dest = s1 * adj(s2);
  clock_t t2 = clock();

  return double(t2-t1)/CLOCKS_PER_SEC;
//  return 2.0;
}

double QDP_M_eq_Ma_times_Ma(LatticeColorMatrix& dest, 
			    const LatticeColorMatrix& s1, 
			    const LatticeColorMatrix& s2,
			    int cnt)
{
  clock_t t1 = clock();
  for (; cnt-- > 0; )
    dest = adj(s1) * adj(s2);
  clock_t t2 = clock();

  return double(t2-t1)/CLOCKS_PER_SEC;
//    return 2.0;
}

double QDP_M_peq_M_times_M(LatticeColorMatrix& dest, 
			  const LatticeColorMatrix& s1, 
			  const LatticeColorMatrix& s2,
			  int cnt)
{
  clock_t t1 = clock();
  for (; cnt-- > 0; )
    dest += s1 * s2;
  clock_t t2 = clock();

  return double(t2-t1)/CLOCKS_PER_SEC;
//    return 2;
}


