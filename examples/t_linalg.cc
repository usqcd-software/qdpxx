// $Id: t_linalg.cc,v 1.3 2003-08-08 21:20:43 edwards Exp $

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

  cout << "CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << endl;

  LatticeColorMatrix a, b, c;
  gaussian(a);
  gaussian(b);
  gaussian(c);

  int icnt;
  double tt;

#if defined(TEST_OPS)
  // Test c=a*b
  tt = QDP_M_eq_M_times_M(c, a, b, 1);
  push(nml,"QDP_M_eq_M_times_M");
  Write(nml,c);
  pop(nml);
#endif

#if ! defined(TEST_OPS)
  // Time the c=a*b
  for(icnt=1; ; icnt <<= 1)
  {
    cout << "calling " << icnt << " times" << endl;
    tt = QDP_M_eq_M_times_M(c, a, b, icnt);
    if (tt > 1)
      break;
  }
#else
  cout << "***WARNING*** : debug mode - timings are bogus" << endl;
  icnt = 1;   // turn off timings for some testing
#endif

  double rescale = 1000*1000 / double(Layout::sitesOnNode()) / icnt;

  tt *= rescale;
  cout << "time(c=a*b) = " << tt
       << " micro-secs/site/iteration" 
       << " , " << 198 / tt << " Mflops" << endl;
  
#if defined(TEST_OPS)
  // Test c=adj(a)*b
  tt = QDP_M_eq_Ma_times_M(c, a, b, 1);
  push(nml,"QDP_M_eq_Ma_times_M");
  Write(nml,a);
  Write(nml,b);
  Write(nml,c);
  pop(nml);
#endif

  // Time the c=adj(a)*b
  cout << "calling " << icnt << " times" << endl;
  tt = rescale * QDP_M_eq_Ma_times_M(c, a, b, icnt);
  cout << "time(c=adj(a)*b) = " << tt
       << " micro-secs/site/iteration" 
       << " , " << 198 / tt << " Mflops" << endl;
  
#if defined(TEST_OPS)
  // Test c=a*adj(b)
  tt = QDP_M_eq_M_times_Ma(c, a, b, 1);
  push(nml,"QDP_M_eq_M_times_Ma");
  Write(nml,a);
  Write(nml,b);
  Write(nml,c);
  pop(nml);
#endif

  // Time the c=a*adj(b)
  cout << "calling " << icnt << " times" << endl;
  tt = rescale * QDP_M_eq_M_times_Ma(c, a, b, icnt);
  cout << "time(c=a*adj(b)) = " << tt
       << " micro-secs/site/iteration" 
       << " , " << 198 / tt << " Mflops" << endl;
  
#if defined(TEST_OPS)
  // Test c=adj(a)*adj(b)
  tt = QDP_M_eq_Ma_times_Ma(c, a, b, 1);
  push(nml,"QDP_M_eq_Ma_times_Ma");
  Write(nml,a);
  Write(nml,b);
  Write(nml,c);
  pop(nml);
#endif

  // Time the c=adj(a)*adj(b)
  cout << "calling " << icnt << " times" << endl;
  tt = rescale * QDP_M_eq_Ma_times_Ma(c, a, b, icnt);
  cout << "time(c=adj(a)*adj(b)) = " << tt
       << " micro-secs/site/iteration" 
       << " , " << 198 / tt << " Mflops" << endl;
  
#if defined(TEST_OPS)
  // Test c+=a*b
  tt = QDP_M_peq_M_times_M(c, a, b, 1);
  push(nml,"QDP_M_peq_M_times_M");
  Write(nml,c);
  pop(nml);
#endif

  // Time the c+=a*b
  cout << "calling " << icnt << " times" << endl;
  tt = rescale * QDP_M_peq_M_times_M(c, a, b, icnt);
  cout << "time(c+=a*b) = " << tt
       << " micro-secs/site/iteration" 
       << " , " << 216 / tt << " Mflops" << endl;

  nml.flush();

//----------------------------------------------------------------------------
  LatticeColorVector lv1,lv2;
  gaussian(lv1);
  gaussian(lv2);

#if ! defined(TEST_OPS)
  // Time the c=a*b
  for(icnt=1; ; icnt <<= 1)
  {
    cout << "calling " << icnt << " times" << endl;
    tt = QDP_V_eq_M_times_V(lv2, a, lv1, icnt);
    if (tt > 1)
      break;
  }
#else
  icnt = 1;   // turn off timings for some testing
#endif

#if defined(TEST_OPS)
  // Test LatticeColorVector = LatticeColorMatrix * LatticeColorVector
  tt = QDP_V_eq_M_times_V(lv2, a, lv1, 1);
  push(nml,"QDP_V_eq_M_times_V");
  Write(nml,lv2);
  pop(nml);
#endif

  // Test LatticeColorVector = LatticeColorMatrix * LatticeColorVector
  cout << "calling " << icnt << " times" << endl;
  tt = rescale * QDP_V_eq_M_times_V(lv2, a, lv1, icnt);
  cout << "time(lv=lm*lv) = " << tt
       << " micro-secs/site/iteration" 
       << " , " << 108 / tt << " Mflops" << endl;   // check the flop count

#if defined(TEST_OPS)
  // Test LatticeColorVector = LatticeColorMatrix * LatticeColorVector
  tt = QDP_V_eq_Ma_times_V(lv2, a, lv1, 1);
  push(nml,"QDP_V_eq_Ma_times_V");
  Write(nml,lv2);
  pop(nml);
#endif

  // Test LatticeColorVector = LatticeColorMatrix * LatticeColorVector
  cout << "calling " << icnt << " times" << endl;
  tt = rescale * QDP_V_eq_Ma_times_V(lv2, a, lv1, icnt);
  cout << "time(lv=adj(lm)*lv) = " << tt
       << " micro-secs/site/iteration" 
       << " , " << 108 / tt << " Mflops" << endl;   // check the flop count

  nml.flush();

  // Time to bolt
  QDP_finalize();

  exit(0);
}
