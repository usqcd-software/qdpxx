// $Id: t_blas_g5.cc,v 1.1 2005-03-16 16:29:39 bjoo Exp $

#include <iostream>
#include <cstdio>

#include "qdp.h"
#include "scalarsite_generic/generic_blas_vaxpy3_g5.h"

 
using namespace QDP;

int main(int argc, char *argv[])
{
  // Put the machine into a known state
  QDP_initialize(&argc, &argv);

  // Setup the layout
  const int foo[] = {8,8,8,8};
  multi1d<int> nrow(Nd);
  nrow = foo;  // Use only Nd elements
  Layout::setLattSize(nrow);
  Layout::create();

  
  Real a=Real(1.5);
  LatticeFermion qx;
  LatticeFermion qy;
  LatticeFermion qz;
  LatticeFermion qz2;
 
  gaussian(qx);
  qy=qx;

  qz  = a*qx + chiralProjectPlus(qy);
  REAL *Out = (REAL *)&(qz2.elem(0).elem(0).elem(0).real());
  REAL *scalep = (REAL *)&(a.elem().elem().elem().elem());
  REAL *InScale = (REAL *)&(qx.elem(0).elem(0).elem(0).real());
  REAL *Add = (REAL *)&(qy.elem(0).elem(0).elem(0).real());
  int n_4vec = all.end()-all.start()+1;

  axpyz_g5ProjPlus(Out, scalep, InScale, Add, n_4vec);
 
 
  Double norm_diff=norm2(qz-qz2);
  {
    QDPIO::cout << "ax + P_{+}y diff=" << norm_diff << endl;
    StopWatch swatch;
    double time=0;
    int iter=1;
    while( time < 1.0 ) { 
      iter *=2;
      QDPIO::cout << "Calling " << iter << " times " << endl;
      swatch.reset();
      swatch.start();
      for(int i=0; i < iter; i++) {
	qz  = a*qx + chiralProjectPlus(qy);
      }
      swatch.stop();
      time = swatch.getTimeInSeconds();
      Internal::broadcast(time);

    }
    
    QDPIO::cout << "Timing with " << iter << " iters" << endl;
    swatch.reset();
    swatch.start();
    for(int i=0; i < iter; i++) {
      qz  = a*qx + chiralProjectPlus(qy);
    }
    swatch.stop();
    time = swatch.getTimeInMicroseconds();
    Internal::broadcast(time);
    
    double Nflops = (double)(2*Nc*Ns*Layout::sitesOnNode()*iter);
    QDPIO::cout << "Time taken: " << time << "(us) Perf: " << Nflops/time << " Mflop/s per node" << endl;
  }
  {
    StopWatch swatch;
    double time=0;
    int iter=1;
    while( time < 1.0 ) { 
      iter *=2;
      QDPIO::cout << "Calling " << iter << " times " << endl;
      swatch.reset();
      swatch.start();
      for(int i=0; i < iter; i++) {
	axpyz_g5ProjPlus(Out, scalep, InScale, Add, n_4vec);
      }
      swatch.stop();
      time = swatch.getTimeInSeconds();
      Internal::broadcast(time);
    }
    
    QDPIO::cout << "Timing with " << iter << " iters" << endl;
    swatch.reset();
    swatch.start();
    for(int i=0; i < iter; i++) {
      axpyz_g5ProjPlus(Out, scalep, InScale, Add,n_4vec);
    }
    swatch.stop();
    time = swatch.getTimeInMicroseconds();
    Internal::broadcast(time);
    
    double Nflops = (double)(2*Nc*Ns*Layout::sitesOnNode()*iter);
    QDPIO::cout << "Time taken: " << time << "(us) Perf: " << Nflops/time << " MFlops/node" << endl;
  }
    
    
  // Time to bolt
  QDP_finalize();

  exit(0);
}
  
