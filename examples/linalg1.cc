// $Id: linalg1.cc,v 1.1 2003-07-30 18:40:19 edwards Exp $

#include <time.h>

#include "qdp.h"
#include "linalg.h"

using namespace QDP;

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


#if 1
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



double QDP_M_eq_M_times_M(LatticeColorMatrix& dest, 
			  const LatticeColorMatrix& s1, 
			  const LatticeColorMatrix& s2,
			  int cnt)
{
  clock_t t1 = clock();
  for(int i=0; i < cnt; ++i)
    dest = s1 * s2;
  clock_t t2 = clock();

  return double(t2-t1)/CLOCKS_PER_SEC;
}

double QDP_M_eq_Ma_times_M(LatticeColorMatrix& dest, 
			  const LatticeColorMatrix& s1, 
			  const LatticeColorMatrix& s2,
			  int cnt)
{
  clock_t t1 = clock();
  for(int i=0; i < cnt; ++i)
    dest = adj(s1) * s2;
  clock_t t2 = clock();

  return double(t2-t1)/CLOCKS_PER_SEC;
}

double QDP_M_peq_M_times_M(LatticeColorMatrix& dest, 
			  const LatticeColorMatrix& s1, 
			  const LatticeColorMatrix& s2,
			  int cnt)
{
  clock_t t1 = clock();
  for(int i=0; i < cnt; ++i)
    dest += s1 * s2;
  clock_t t2 = clock();

  return double(t2-t1)/CLOCKS_PER_SEC;
}


