// $Id: t_linalg.cc,v 1.13 2003-10-09 19:59:39 edwards Exp $

#include <iostream>
#include <cstdio>

#include <time.h>

#include "qdp.h"
#include "linalg.h"

using namespace QDP;

int main(int argc, char *argv[])
{
  // Put the machine into a known state
  QDP_initialize(&argc, &argv);

  // Setup the layout
  const int foo[] = {4,4,4,4};
  multi1d<int> nrow(Nd);
  nrow = foo;  // Use only Nd elements
  Layout::setLattSize(nrow);
  Layout::create();

  NmlWriter nml("t_linalg.nml");

  push(nml,"lattis");
  Write(nml,Nd);
  Write(nml,Nc);
  Write(nml,nrow);
  pop(nml);

  QDPIO::cout << "CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << endl;

  LatticeColorMatrix a, b, c;
  gaussian(a);
  gaussian(b);
  gaussian(c);

  int icnt;
  double tt;

#define TIME_OPS 

  // Test M=M*M
  for(icnt=1; ; icnt <<= 1)
  {
    QDPIO::cout << "calling " << icnt << " times" << endl;
    tt = QDP_M_eq_M_times_M(c, a, b, icnt);
#if defined(TIME_OPS)
    if (tt > 1)
      break;
#else
    // turn off timings for some testing
    QDPIO::cout << "***WARNING*** : debug mode - timings are bogus" << endl;
    break;
#endif
  }

  double rescale = 1000*1000 / double(Layout::sitesOnNode()) / icnt;

  tt *= rescale;
#if defined(TIME_OPS)
  QDPIO::cout << "time(M=M*M) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << 198 / tt << " Mflops" << endl;
#else
  push(nml,"QDP_M_eq_M_times_M");
  Write(nml,c);
  pop(nml);
#endif
  
  // Test  M=adj(M)*M
  QDPIO::cout << "calling " << icnt << " times" << endl;
  tt = rescale * QDP_M_eq_Ma_times_M(c, a, b, icnt);
#if defined(TIME_OPS)
  QDPIO::cout << "time(M=adj(M)*M) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << 198 / tt << " Mflops" << endl;
#else
  push(nml,"QDP_M_eq_Ma_times_M");
  Write(nml,a);
  Write(nml,b);
  Write(nml,c);
  pop(nml);
#endif
  
  // Test  M=M*adj(M)
  QDPIO::cout << "calling " << icnt << " times" << endl;
  tt = rescale * QDP_M_eq_M_times_Ma(c, a, b, icnt);
#if defined(TIME_OPS)
  QDPIO::cout << "time(M=M*adj(M)) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << 198 / tt << " Mflops" << endl;
#else
  push(nml,"QDP_M_eq_M_times_Ma");
  Write(nml,a);
  Write(nml,b);
  Write(nml,c);
  pop(nml);
#endif

 
  // Test  M=adj(M)*adj(M)
  QDPIO::cout << "calling " << icnt << " times" << endl;
  tt = rescale * QDP_M_eq_Ma_times_Ma(c, a, b, icnt);
#if defined(TIME_OPS)
  QDPIO::cout << "time(M=adj(M)*adj(M)) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << 198 / tt << " Mflops" << endl;
#else
  push(nml,"QDP_M_eq_Ma_times_Ma");
  Write(nml,a);
  Write(nml,b);
  Write(nml,c);
  pop(nml);
#endif

  
  // Test  M+= M*M
  QDPIO::cout << "calling " << icnt << " times" << endl;
  tt = rescale * QDP_M_peq_M_times_M(c, a, b, icnt);
#if defined(TIME_OPS)
  QDPIO::cout << "time(M+=M*M) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << 216 / tt << " Mflops" << endl;
#else
  push(nml,"QDP_M_peq_M_times_M");
  Write(nml,c);
  pop(nml);
#endif

  nml.flush();

//----------------------------------------------------------------------------
  LatticeColorVector lv1,lv2,lv3;
  gaussian(lv1);
  gaussian(lv2);
  gaussian(lv3);

  // Test LatticeColorVector = LatticeColorMatrix * LatticeColorVector
  for(icnt=1; ; icnt <<= 1)
  {
    QDPIO::cout << "calling " << icnt << " times" << endl;
    tt = QDP_V_eq_M_times_V(lv2, a, lv1, icnt);
#if defined(TIME_OPS)
    if (tt > 1)
      break;
#else
    // turn off timings for some testing
    break;
#endif
  }

  rescale = 1000*1000 / double(Layout::sitesOnNode()) / icnt;

  tt *= rescale;
#if defined(TIME_OPS)
  QDPIO::cout << "time(V=M*V) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << 66 / tt << " Mflops" << endl;   // check the flop count
#else
  push(nml,"QDP_V_eq_M_times_V");
  Write(nml,lv2);
  pop(nml);
#endif


  // Test LatticeColorVector = LatticeColorMatrix * LatticeColorVector
  QDPIO::cout << "calling " << icnt << " times" << endl;
  tt = rescale * QDP_V_eq_Ma_times_V(lv2, a, lv1, icnt);
#if defined(TIME_OPS)
  QDPIO::cout << "time(V=adj(M)*V) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << 66 / tt << " Mflops" << endl;   // check the flop count
#else
  push(nml,"QDP_V_eq_Ma_times_V");
  Write(nml,lv2);
  pop(nml);
#endif


//----------------------------------------------------------------------------
  // Test LatticeColorVector = LatticeColorVector + LatticeColorVector
  for(icnt=1; ; icnt <<= 1)
  {
    QDPIO::cout << "calling " << icnt << " times" << endl;
    tt = QDP_V_eq_V_plus_V(lv3, lv1, lv2, icnt);
#if defined(TIME_OPS)
    if (tt > 1)
      break;
#else
    // turn off timings for some testing
    break;
#endif
  }

  rescale = 1000*1000 / double(Layout::sitesOnNode()) / icnt;

  tt *= rescale;
#if defined(TIME_OPS)
  QDPIO::cout << "time(V=V+V) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << 6 / tt << " Mflops" << endl;   // check the flop count
#else
  push(nml,"QDP_V_eq_V_plus_V");
  Write(nml,lv3);
  pop(nml);
#endif

  nml.flush();


//----------------------------------------------------------------------------
  LatticeDiracFermion lf1,lf2,lf3;
  gaussian(lf1);
  gaussian(lf2);

  // Test LatticeDiracFermion = LatticeColorMatrix * LatticeDiracFermion
  for(icnt=1; ; icnt <<= 1)
  {
    QDPIO::cout << "calling " << icnt << " times" << endl;
    tt = QDP_D_eq_M_times_D(lf2, a, lf1, icnt);
#if defined(TIME_OPS)
    if (tt > 1)
      break;
#else
    // turn off timings for some testing
    break;
#endif
  }

  rescale = 1000*1000 / double(Layout::sitesOnNode()) / icnt;

  tt *= rescale;
#if defined(TIME_OPS)
  QDPIO::cout << "time(D=M*D) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << 264 / tt << " Mflops" << endl;   // check the flop count
#else
  push(nml,"QDP_D_eq_M_times_D");
  Write(nml,lf2);
  pop(nml);
#endif

  // Test LatticeDiracFermion = adj(LatticeColorMatrix) * LatticeDiracFermion
  QDPIO::cout << "calling " << icnt << " times" << endl;
  tt = rescale * QDP_D_eq_Ma_times_D(lf2, a, lf1, icnt);
#if defined(TIME_OPS)
  QDPIO::cout << "time(D=adj(M)*D) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << 264 / tt << " Mflops" << endl;   // check the flop count
#else
  push(nml,"QDP_D_eq_Ma_times_D");
  Write(nml,lf2);
  pop(nml);
#endif

  nml.flush();

//----------------------------------------------------------------------------
  LatticeHalfFermion lh1,lh2,lh3;
  gaussian(lh1);
  gaussian(lh2);

  // Test LatticeHalfFermion = LatticeColorMatrix * LatticeHalfFermion
  for(icnt=1; ; icnt <<= 1)
  {
    QDPIO::cout << "calling " << icnt << " times" << endl;
    tt = QDP_H_eq_M_times_H(lh2, a, lh1, icnt);
#if defined(TIME_OPS)
    if (tt > 1)
      break;
#else
    // turn off timings for some testing
    break;
#endif
  }

  rescale = 1000*1000 / double(Layout::sitesOnNode()) / icnt;

  tt *= rescale;
#if defined(TIME_OPS)
  QDPIO::cout << "time(H=M*H) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << 132 / tt << " Mflops" << endl;   // check the flop count
#else
  push(nml,"QDP_H_eq_M_times_H");
  Write(nml,lh2);
  pop(nml);
#endif


  // Test LatticeHalfFermion = adj(LatticeColorMatrix) * LatticeHalfFermion
  QDPIO::cout << "calling " << icnt << " times" << endl;
  tt = rescale * QDP_H_eq_Ma_times_H(lh2, a, lh1, icnt);
#if defined(TIME_OPS)
  QDPIO::cout << "time(H=adj(M)*H) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << 132 / tt << " Mflops" << endl;   // check the flop count
#else
  push(nml,"QDP_H_eq_Ma_times_H");
  Write(nml,lh2);
  pop(nml);
#endif

  nml.flush();

  // Time to bolt
  QDP_finalize();

  exit(0);
}
