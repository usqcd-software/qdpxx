// $Id: t_linalg.cc,v 1.1 2003-07-30 18:40:19 edwards Exp $

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

  double rescale = 1000*1000 / double(Layout::sitesOnNode());

  // Time the c=a*b
  int icnt;
  double tt;
  for(icnt=1; ; icnt <<= 1)
  {
    cout << "calling " << icnt << " times" << endl;
    tt = QDP_M_eq_M_times_M(c, a, b, icnt);
    if (tt > 1)
      break;
  }
  double t_mltmm = rescale*tt/icnt;
  cout << "time(c=a*b) = " << t_mltmm
       << " micro-secs/site/iteration" 
       << " , " << 198 / t_mltmm << " Mflops" << endl;

  
  // Time the c=adj(a)*b
  cout << "calling " << icnt << " times" << endl;
  tt = QDP_M_eq_Ma_times_M(c, a, b, icnt);
  double t_mltcm = rescale*tt/icnt;
  cout << "time(c=adj(a)*b) = " << t_mltcm
       << " micro-secs/site/iteration" 
       << " , " << 198 / t_mltcm << " Mflops" << endl;
  
  // Time the c+=a*b
  cout << "calling " << icnt << " times" << endl;
  tt = QDP_M_peq_M_times_M(c, a, b, icnt);
  double t_peq_mltmm = rescale*tt/icnt;
  cout << "time(c+=a*b) = " << t_peq_mltmm
       << " micro-secs/site/iteration" 
       << " , " << 216 / t_peq_mltmm << " Mflops" << endl;

  nml.flush();

  // Time to bolt
  QDP_finalize();

  exit(0);
}
