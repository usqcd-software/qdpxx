// $Id: linalg1.cc,v 1.14 2003-08-15 18:26:39 edwards Exp $

#include <stdlib.h>
#include <sys/time.h>

#include "qdp.h"
#include "linalg.h"


using namespace QDP;


double QDP_M_eq_M_times_M(LatticeColorMatrix& dest, 
			  const LatticeColorMatrix& s1, 
			  const LatticeColorMatrix& s2,
			  int cnt)
{
  struct timeval t1,t2;
  gettimeofday(&t1, NULL);
//  clock_t t1 = clock();
  for (; cnt-- > 0; )
    dest = s1 * s2;
//  clock_t t2 = clock();
  gettimeofday(&t2, NULL);

  double tt = (t2.tv_sec - t1.tv_sec) * 1000000.0 + (t2.tv_usec - t1.tv_usec);

//  fprintf(stdout,"t1= %f  t2= %f  t2-t1= %f\n", 
//      (t1.tv_sec*1000000.0+t1.tv_usec),
//      (t2.tv_sec*1000000.0+t2.tv_usec),
//      tt);
//  return (double)((int)(t2)-(int)(t1))/(double)(CLOCKS_PER_SEC);

  return tt / 1000000.0;

//  return 2;
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

  return double(t2-t1)/double(CLOCKS_PER_SEC);
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

  return double(t2-t1)/double(CLOCKS_PER_SEC);
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

  return double(t2-t1)/double(CLOCKS_PER_SEC);
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

  return double(t2-t1)/double(CLOCKS_PER_SEC);
//    return 2;
}


double QDP_V_eq_M_times_V(LatticeColorVector& dest, 
			  const LatticeColorMatrix& s1, 
			  const LatticeColorVector& s2,
			  int cnt)
{
  clock_t t1 = clock();
  for (; cnt-- > 0; )
    dest = s1 * s2;
  clock_t t2 = clock();

  return double(t2-t1)/double(CLOCKS_PER_SEC);
//    return 2.0;
}


double QDP_V_eq_Ma_times_V(LatticeColorVector& dest, 
			   const LatticeColorMatrix& s1, 
			   const LatticeColorVector& s2,
			   int cnt)
{
  clock_t t1 = clock();
  for (; cnt-- > 0; )
    dest = adj(s1) * s2;
  clock_t t2 = clock();

  return double(t2-t1)/double(CLOCKS_PER_SEC);
//    return 2.0;
}


double QDP_V_eq_V_plus_V(LatticeColorVector& dest, 
			 const LatticeColorVector& s1, 
			 const LatticeColorVector& s2,
			 int cnt)
{
  clock_t t1 = clock();
  for (; cnt-- > 0; )
    dest = s1 + s2;
  clock_t t2 = clock();

  return double(t2-t1)/double(CLOCKS_PER_SEC);
//    return 2.0;
}


double QDP_D_eq_M_times_D(LatticeDiracFermion& dest, 
			  const LatticeColorMatrix& s1, 
			  const LatticeDiracFermion& s2,
			  int cnt)
{
  clock_t t1 = clock();
  for (; cnt-- > 0; )
    dest = s1 * s2;
  clock_t t2 = clock();

  return double(t2-t1)/double(CLOCKS_PER_SEC);
//    return 2.0;
}


double QDP_H_eq_M_times_H(LatticeHalfFermion& dest, 
			  const LatticeColorMatrix& s1, 
			  const LatticeHalfFermion& s2,
			  int cnt)
{
  clock_t t1 = clock();
  for (; cnt-- > 0; )
    dest = s1 * s2;
  clock_t t2 = clock();

  return double(t2-t1)/double(CLOCKS_PER_SEC);
//    return 2.0;
}


double QDP_H_eq_Ma_times_H(LatticeHalfFermion& dest, 
		 	   const LatticeColorMatrix& s1, 
			   const LatticeHalfFermion& s2,
			   int cnt)
{
  clock_t t1 = clock();
  for (; cnt-- > 0; )
    dest = adj(s1) * s2;
  clock_t t2 = clock();

  return double(t2-t1)/double(CLOCKS_PER_SEC);
//    return 2.0;
}


