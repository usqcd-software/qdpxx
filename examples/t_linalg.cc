// $Id: t_linalg.cc,v 1.20 2006-09-27 17:26:43 bjoo Exp $

#include <iostream>
#include <cstdio>

#include <time.h>

#include "qdp.h"
#include "linalg.h"

using namespace QDP;
#define TIME_OPS

int main(int argc, char *argv[])
{
  // Put the machine into a known state
  QDP_initialize(&argc, &argv);

  multi1d<int> nrow(Nd);

  if (argc >= 5) {
    nrow[0] = atoi(argv[1]);
    nrow[1] = atoi(argv[2]);
    nrow[2] = atoi(argv[3]);
    nrow[3] = atoi(argv[4]);
    
    QDP_info ("Lattice size %dx%dx%dx%d\n", nrow[0], nrow[1], nrow[2], nrow[3]);
  }
  else {
    // Setup the layout
#if defined(ARCH_PARSCALAR) || defined (ARCH_PARSCALARVEC)
    // const int foo[] = {32,32,32,32};
    const int foo[] = {32,16,16,16};
#else
    const int foo[] = {32,32,32,32};
#endif
    nrow = foo;  // Use only Nd elements
  }
  Layout::setLattSize(nrow);
  Layout::create();

#if 1
#ifndef QDP_NO_LIBXML2
  XMLFileWriter xml("t_linalg.xml");
  push(xml, "linalgTest");

  push(xml,"lattis");
  write(xml,"Nd", Nd);
  write(xml,"Nc", Nc);
  write(xml,"nrow", nrow);
  pop(xml);
#endif

#endif

  QDPIO::cout << "CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << endl;

  LatticeColorMatrix a, b, c;
  gaussian(a);
  gaussian(b);
  gaussian(c);

  int icnt;
  double tt;

#if 1

#define TIME_OPS 
  // Test M = M
  LatticeColorMatrix m1, m2;
  m1 = zero;
  gaussian(m2);

#if defined(ARCH_SCALARVEC) || defined(ARCH_PARSCALARVEC)
  int istart = all.start() >> INNER_LOG;
  int iend = all.end() >> INNER_LOG;
  for(int i=istart; i <= iend; i++) { 
    m1.elem(i) -= m2.elem(i);
  }
#else
  for(int i=all.start(); i <= all.end(); i++) { 
    m1.elem(i) -= m2.elem(i);
  }
#endif

  LatticeColorMatrix m3=zero;
  m3 -= m2;
  LatticeColorMatrix diff_m;
  diff_m = m3 - m1;
  QDPIO::cout << "Diff M=M = " << norm2(diff_m) << endl;
  QDP::StopWatch swatch;
  swatch.reset();
  double time = 0;
  icnt = 1;
  while(time <= 1000000) { 
    swatch.start();
    for(int j=0; j < icnt; j++) {
#if defined(ARCH_SCALARVEC)|| defined(ARCH_PARSCALARVEC)
      int istart = all.start() >> INNER_LOG;
      int iend = all.end() >> INNER_LOG;
      for(int i=istart; i <= iend; i++) { 
	m1.elem(i) -= m2.elem(i);
      }
#else
      for(int i=all.start(); i <= all.end(); i++) { 
	m1.elem(i) -= m2.elem(i);
      }
#endif
    }
    swatch.stop();
    time = swatch.getTimeInMicroseconds();
    swatch.reset();
    icnt*=2;
  }
  QDPIO::cout << "Call time (old M=M) = " << time / icnt << " us per call" << endl;
  
  swatch.reset();
  time = 0;
  icnt = 1;
  while(time <= 1000000) { 
    swatch.start();
    for(int j=0; j < icnt; j++) {
      m1-=m2;
    }
    swatch.stop();
    time = swatch.getTimeInMicroseconds();
    swatch.reset();
    icnt*=2;
  }
  
  QDPIO::cout << "Call time (New M=M)= " << time / icnt << " us per call" << endl;
#endif


  // Test M=M*M
  for(icnt=1; ; icnt <<= 1)
  {
    QDPIO::cout << "calling M=M*M " << icnt << " times" << endl;
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

  //  double rescale = 1000*1000 / double(Layout::sitesOnNode()) / icnt;
  double rescale = 1000*1000 / double(Layout::vol()) / icnt;

  tt *= rescale;
  int Nflops = Nc*Nc*(4*Nc + (4*Nc-2));
  int NBytes = sizeof(a) * 3;
#if defined(TIME_OPS)
  QDPIO::cout << "time(M=M*M) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << Nflops / tt << " Mflops" 
              << " , " << NBytes / tt << " MBytes/s"
	      << endl;
#else
#ifndef QDP_NO_LIBXML2
  push(xml,"QDP_M_eq_M_times_M");
  write(xml,"c", c);
  pop(xml);
#endif

#endif
  
#if 1
  // Test  M=adj(M)*M
  QDPIO::cout << "calling M=adj(M)*M " << icnt << " times" << endl;
  tt = rescale * QDP_M_eq_Ma_times_M(c, a, b, icnt);
#if defined(TIME_OPS)
  QDPIO::cout << "time(M=adj(M)*M) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << Nflops / tt << " Mflops" << endl;
#else
#ifndef QDP_NO_LIBXML2
  push(xml,"QDP_M_eq_Ma_times_M");
  write(xml,"a",a);
  write(xml,"b",b);
  write(xml,"c",c);
  pop(xml);
#endif
#endif
  
  // Test  M=M*adj(M)
  QDPIO::cout << "calling M=M*adj(M) " << icnt << " times" << endl;
  tt = rescale * QDP_M_eq_M_times_Ma(c, a, b, icnt);
#if defined(TIME_OPS)
  QDPIO::cout << "time(M=M*adj(M)) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << Nflops / tt << " Mflops" << endl;
#else
#ifndef QDP_NO_LIBXML2
  push(xml,"QDP_M_eq_M_times_Ma");
  write(xml,"a",a);
  write(xml,"b",b);
  write(xml,"c",c);
  pop(xml);
#endif
#endif

 
  // Test  M=adj(M)*adj(M)
  QDPIO::cout << "calling M=adj(M)*adj(M) " << icnt << " times" << endl;
  tt = rescale * QDP_M_eq_Ma_times_Ma(c, a, b, icnt);
#if defined(TIME_OPS)
  QDPIO::cout << "time(M=adj(M)*adj(M)) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << Nflops / tt << " Mflops" << endl;
#else
#ifndef QDP_NO_LIBXML2
  push(xml,"QDP_M_eq_Ma_times_Ma");
  write(xml,"a",a);
  write(xml,"b",b);
  write(xml,"c",c);
  pop(xml);
#endif
#endif

  
  // Test  M+= M*M
  QDPIO::cout << "calling M+=M*M " << icnt << " times" << endl;
  tt = rescale * QDP_M_peq_M_times_M(c, a, b, icnt);
  Nflops += Nc*Nc * 2;
#if defined(TIME_OPS)
  QDPIO::cout << "time(M+=M*M) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << Nflops / tt << " Mflops" << endl;
#else
#ifndef QDP_NO_LIBXML2
  push(xml,"QDP_M_peq_M_times_M");
  write(xml,"c",c);
  pop(xml);
#endif
#endif


  // Test  M+= adj(M)*M
  QDPIO::cout << "calling M+=adj(M)*M " << icnt << " times" << endl;
  tt = rescale * QDP_M_peq_Ma_times_M(c, a, b, icnt);
#if defined(TIME_OPS)
  QDPIO::cout << "time(M+=adj(M)*M) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << Nflops / tt << " Mflops" << endl;
#else
#ifndef QDP_NO_LIBXML2
  push(xml,"QDP_M_peq_Ma_times_M");
  write(xml,"c",c);
  pop(xml);
#endif
#endif


  // Test  M+= M*adj(M)
  QDPIO::cout << "calling M+=M*adj(M) " << icnt << " times" << endl;
  tt = rescale * QDP_M_peq_M_times_Ma(c, a, b, icnt);
#if defined(TIME_OPS)
  QDPIO::cout << "time(M+=M*adj(M)) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << Nflops / tt << " Mflops" << endl;
#else
#ifndef QDP_NO_LIBXML2
  push(xml,"QDP_M_peq_M_times_Ma");
  write(xml,"c",c);
  pop(xml);
#endif
#endif


  // Test  M+= adj(M)*adj(M)
  QDPIO::cout << "calling M+=adj(M)*adj(M) " << icnt << " times" << endl;
  tt = rescale * QDP_M_peq_Ma_times_Ma(c, a, b, icnt);
#if defined(TIME_OPS)
  QDPIO::cout << "time(M+=adj(M)*adj(M)) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << Nflops / tt << " Mflops" << endl;
#else
#ifndef QDP_NO_LIBXML2
  push(xml,"QDP_M_peq_Ma_times_Ma");
  write(xml,"c",c);
  pop(xml);
#endif
#endif


  // Test  M-= M*M
  QDPIO::cout << "calling M-=M*M " << icnt << " times" << endl;
  tt = rescale * QDP_M_meq_M_times_M(c, a, b, icnt);
#if defined(TIME_OPS)
  QDPIO::cout << "time(M-=M*M) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << Nflops / tt << " Mflops" << endl;
#else
#ifndef QDP_NO_LIBXML2
  push(xml,"QDP_M_meq_M_times_M");
  write(xml,"c",c);
  pop(xml);
#endif
#endif


  // Test  M-= adj(M)*M
  QDPIO::cout << "calling M-=adj(M)*M " << icnt << " times" << endl;
  tt = rescale * QDP_M_meq_Ma_times_M(c, a, b, icnt);
#if defined(TIME_OPS)
  QDPIO::cout << "time(M-=adj(M)*M) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << Nflops / tt << " Mflops" << endl;
#else
#ifndef QDP_NO_LIBXML2
  push(xml,"QDP_M_meq_Ma_times_M");
  write(xml,"c",c);
  pop(xml);
#endif
#endif


  // Test  M-= M*adj(M)
  QDPIO::cout << "calling M-=M*adj(M) " << icnt << " times" << endl;
  tt = rescale * QDP_M_meq_M_times_Ma(c, a, b, icnt);
#if defined(TIME_OPS)
  QDPIO::cout << "time(M-=M*adj(M)) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << Nflops / tt << " Mflops" << endl;
#else
#ifndef QDP_NO_LIBXML2
  push(xml,"QDP_M_meq_M_times_Ma");
  write(xml,"c",c);
  pop(xml)
#endif
#endif


  // Test  M-= adj(M)*adj(M)
  QDPIO::cout << "calling M-=adj(M)*adj(M) " << icnt << " times" << endl;
  tt = rescale * QDP_M_meq_Ma_times_Ma(c, a, b, icnt);
#if defined(TIME_OPS)
  QDPIO::cout << "time(M-=adj(M)*adj(M)) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << Nflops / tt << " Mflops" << endl;
#else
#ifndef QDP_NO_LIBXML2
  push(xml,"QDP_M_meq_Ma_times_Ma");
  write(xml,"c",c);
  pop(xml);
#endif
#endif


//----------------------------------------------------------------------------
  LatticeColorVector lv1,lv2,lv3;
  gaussian(lv1);
  gaussian(lv2);
  gaussian(lv3);

  // Test LatticeColorVector = LatticeColorMatrix * LatticeColorVector
  for(icnt=1; ; icnt <<= 1)
  {
    QDPIO::cout << "calling V=M*V " << icnt << " times" << endl;
    tt = QDP_V_eq_M_times_V(lv2, a, lv1, icnt);
#if defined(TIME_OPS)
    if (tt > 1)
      break;
#else
    // turn off timings for some testing
    break;
#endif
  }


  //  rescale = 1000*1000 / double(Layout::sitesOnNode()) / icnt;
  rescale = 1000*1000 / double(Layout::vol()) / icnt;

  tt *= rescale;
#if defined(TIME_OPS)
  QDPIO::cout << "time(V=M*V) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << 66 / tt << " Mflops" << endl;   // check the flop count
#else
#ifndef QDP_NO_LIBXML2
  push(xml,"QDP_V_eq_M_times_V");
  write(xml,"lv2",lv2);
  pop(xml)
#endif
#endif


  // Test LatticeColorVector = LatticeColorMatrix * LatticeColorVector
  QDPIO::cout << "calling V=adj(M)*V " << icnt << " times" << endl;
  tt = rescale * QDP_V_eq_Ma_times_V(lv2, a, lv1, icnt);
#if defined(TIME_OPS)
  QDPIO::cout << "time(V=adj(M)*V) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << 66 / tt << " Mflops" << endl;   // check the flop count
#else
#ifndef QDP_NO_LIBXML2
  push(xml,"QDP_V_eq_Ma_times_V");
  write(xml,"lv2",lv2);
  pop(xml);
#endif
#endif


//----------------------------------------------------------------------------
  // Test LatticeColorVector = LatticeColorVector + LatticeColorVector
  for(icnt=1; ; icnt <<= 1)
  {
    QDPIO::cout << "calling V=V+V " << icnt << " times" << endl;
    tt = QDP_V_eq_V_plus_V(lv3, lv1, lv2, icnt);
#if defined(TIME_OPS)
    if (tt > 1)
      break;
#else
    // turn off timings for some testing
    break;
#endif
  }

  //  rescale = 1000*1000 / double(Layout::sitesOnNode()) / icnt;
  rescale = 1000*1000 / double(Layout::vol()) / icnt;

  tt *= rescale;
#if defined(TIME_OPS)
  QDPIO::cout << "time(V=V+V) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << 6 / tt << " Mflops" << endl;   // check the flop count
#else
#ifndef QDP_NO_LIBXML2
  push(xml,"QDP_V_eq_V_plus_V");
  write(xml,"lv3",lv3);
  pop(xml);
#endif
#endif



//----------------------------------------------------------------------------
  LatticeDiracFermion lf1,lf2,lf3;
  gaussian(lf1);
  gaussian(lf2);

  // Test LatticeDiracFermion = LatticeColorMatrix * LatticeDiracFermion
  for(icnt=1; ; icnt <<= 1)
  {
    QDPIO::cout << "calling D=M*D " << icnt << " times" << endl;
    tt = QDP_D_eq_M_times_D(lf2, a, lf1, icnt);
#if defined(TIME_OPS)
    if (tt > 1)
      break;
#else
    // turn off timings for some testing
    break;
#endif
  }

  //  rescale = 1000*1000 / double(Layout::sitesOnNode()) / icnt;
  rescale = 1000*1000 / double(Layout::vol()) / icnt;

  tt *= rescale;
#if defined(TIME_OPS)
  QDPIO::cout << "time(D=M*D) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << 264 / tt << " Mflops" << endl;   // check the flop count
#else
#ifndef QDP_NO_LIBXML2
  push(xml,"QDP_D_eq_M_times_D");
  write(xml,"lf2",lf2);
  pop(xml);
#endif
#endif

  // Test LatticeDiracFermion = adj(LatticeColorMatrix) * LatticeDiracFermion
  QDPIO::cout << "calling D=adj(M)*D " << icnt << " times" << endl;
  tt = rescale * QDP_D_eq_Ma_times_D(lf2, a, lf1, icnt);
#if defined(TIME_OPS)
  QDPIO::cout << "time(D=adj(M)*D) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << 264 / tt << " Mflops" << endl;   // check the flop count
#else
#ifndef QDP_NO_LIBXML2
  push(xml,"QDP_D_eq_Ma_times_D");
  write(xml,"lf2",lf2);
  pop(xml);
#endif
#endif


//----------------------------------------------------------------------------
  LatticeHalfFermion lh1,lh2,lh3;
  gaussian(lh1);
  gaussian(lh2);

  // Test LatticeHalfFermion = LatticeColorMatrix * LatticeHalfFermion
  for(icnt=1; ; icnt <<= 1)
  {
    QDPIO::cout << "calling H=M*H " << icnt << " times" << endl;
    tt = QDP_H_eq_M_times_H(lh2, a, lh1, icnt);
#if defined(TIME_OPS)
    if (tt > 1)
      break;
#else
    // turn off timings for some testing
    break;
#endif
  }

  //  rescale = 1000*1000 / double(Layout::sitesOnNode()) / icnt;
  rescale = 1000*1000 / double(Layout::vol()) / icnt;

  tt *= rescale;
#if defined(TIME_OPS)
  QDPIO::cout << "time(H=M*H) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << 132 / tt << " Mflops" << endl;   // check the flop count
#else
#ifndef QDP_NO_LIBXML2
  push(xml,"QDP_H_eq_M_times_H");
  write(xml,"lh2", lh2);
  pop(xml);
#endif
#endif


  // Test LatticeHalfFermion = adj(LatticeColorMatrix) * LatticeHalfFermion
  QDPIO::cout << "calling H=adj(M)*H " << icnt << " times" << endl;
  tt = rescale * QDP_H_eq_Ma_times_H(lh2, a, lh1, icnt);
#if defined(TIME_OPS)
  QDPIO::cout << "time(H=adj(M)*H) = " << tt
	      << " micro-secs/site/iteration" 
	      << " , " << 132 / tt << " Mflops" << endl;   // check the flop count
#else
#ifndef QDP_NO_LIBXML2
  push(xml,"QDP_H_eq_Ma_times_H");
  write(xml,"lh2", lh2);
  pop(xml);
#endif

#endif


#ifndef QDP_NO_LIBXML2
  pop(xml);
  xml.close();
#endif

#endif

  // Time to bolt
  QDP_finalize();

  exit(0);
}
