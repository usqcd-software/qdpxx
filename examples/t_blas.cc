// $Id: t_blas.cc,v 1.3 2004-03-23 12:56:10 bjoo Exp $

#include <iostream>
#include <cstdio>

#include <time.h>

#include "qdp.h"
#include <blas1.h>

 
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

  
  Real a=1.5;
  LatticeFermion qx;
  LatticeFermion qy;
  LatticeFermion qz;
  LatticeFermion qtmp;
  LatticeFermion d;
  Double dnorm;

  // Test y += a*x
  gaussian(qx);
  gaussian(qy);
  qz = qy;
  qtmp = qx;
  qtmp *= a;

  // qtmp is now qy + a*qx
  qz += qtmp;
  
  // Now doit in a onner
  qy += a*qx;
  d = qy - qz;
  dnorm = norm2(d);
  QDPIO::cout << "y+=a*x: diff = " << dnorm << endl;

  // Test y -= a*x
  gaussian(qx);
  gaussian(qy);

  qz = qy;
  qtmp = qx;
  qtmp *= a;
  qz -= qtmp; // qz = qy - a*qx

  // Now do it in a onner
  qy -= a*qx; // qy -= a*qx = qy - a*qx
  d = qy - qz;
  dnorm = norm2(d);
  QDPIO::cout << "y-=a*x: diff = " << dnorm << endl;

  // Test z = ax + y 
  gaussian(qx);
  gaussian(qy);

  qz = a*qx;
  qz += qy;

  // Now do it in one
  qtmp = a*qx + qy;
  d = qtmp - qz;
  dnorm = norm2(d);
  QDPIO::cout << "z=a*x + y: diff = " << dnorm << endl;
  
  // Test z = y + ax
  gaussian(qx);
  gaussian(qy);

  qz = qy ;
  qtmp = a*qx;
  qz += qtmp;

  // Now do in a onner.
  qtmp = qy + a*qx;
  d = qtmp - qz;
  dnorm = norm2(d);
  QDPIO::cout << "z=y + a*x: diff = " << dnorm << endl;

  // Test z = ax - y 
  gaussian(qx);
  gaussian(qy);

  qz = a*qx ;
  qz -= qy;

  // Now in a onner
  qtmp = a*qx-qy;
  d = qtmp - qz;
  dnorm = norm2(d);
  QDPIO::cout << "z=a*x - y: diff = " << dnorm << endl;
  

  // Test z = y - ax
  gaussian(qx);
  gaussian(qy);

  qz = qy;
  qtmp = a*qx ;
  qz -= qtmp;

  // Now in a onner
  qtmp = qy - a*qx;
  d = qtmp - qz;
  dnorm = norm2(d);
  QDPIO::cout << "z=y - a*x: diff = " << dnorm << endl;
  
  // If all the above work I can now do others

  // Test y = ax + y
  gaussian(qx);
  gaussian(qy);

  qz = a*qx;
  qz += qy;
  qy = a*qx + qy;  
  d = qz - qy;
  dnorm = norm2(d);
  QDPIO::cout << "y = ax + y: diff = " << dnorm << endl;

  // Test x = ax + y
  gaussian(qx);
  gaussian(qy);
  qz = a*qx;
  qz +=  qy;
  qx = a*qx + qy;
  d = qz - qx;
  dnorm = norm2(d);
  QDPIO::cout << "x = ax + y: diff = " << dnorm << endl;

  // Test y = ax - y
  gaussian(qx);
  gaussian(qy);
  qz = a*qx; 
  qz -= qy;
  qy = a*qx - qy;
  d = qz - qy;
  dnorm = norm2(d);
  QDPIO::cout << "y = ax - y: diff = " << dnorm << endl;

  // Test y = y - ax
  gaussian(qx);
  gaussian(qy);
  qz = - a*qx;
  qz += qy;
  qy = qy - a*qx;
  d = qz - qy;
  dnorm = norm2(d);
  QDPIO::cout << "y = y - ax: diff = " << dnorm << endl;

  // Test x = ax - y
  gaussian(qx);
  gaussian(qy);
  qz = a*qx;
  qz -= qy;
  qx = a*qx - qy;
  d = qz - qx;
  dnorm = norm2(d);
  QDPIO::cout << "x = ax - y: diff = " << dnorm << endl;

  // Test x = y - ax
  gaussian(qx);
  gaussian(qy);
  qz = - a*qx;
  qz += qy;

  qx = qy - a*qx;
  d = qz - qx;
  dnorm = norm2(d);
  QDPIO::cout << "x = y - ax: diff = " << dnorm << endl;

  // Test y += a*x
  gaussian(qx);
  gaussian(qy);
  qz = qy;
  qtmp = qx;
  qtmp *= a;

  // qtmp is now qy + a*qx
  qz += qtmp;
  
  // Now doit in a onner
  qy += qx*a;
  d = qy - qz;
  dnorm = norm2(d);
  QDPIO::cout << "y+=x*a: diff = " << dnorm << endl;

  // Test y -= a*x
  gaussian(qx);
  gaussian(qy);

  qz = qy;
  qtmp = qx;
  qtmp *= a;
  qz -= qtmp; // qz = qy - a*qx

  // Now do it in a onner
  qy -= qx*a; // qy -= a*qx = qy - a*qx
  d = qy - qz;
  dnorm = norm2(d);
  QDPIO::cout << "y-=x*a: diff = " << dnorm << endl;


  // Test z = ax + y 
  gaussian(qx);
  gaussian(qy);

  qz = a*qx;
  qz += qy;

  // Now do it in one
  qtmp = qx*a + qy;
  d = qtmp - qz;
  dnorm = norm2(d);
  QDPIO::cout << "z=x*a + y: diff = " << dnorm << endl;
  
  // Test z = y + ax
  gaussian(qx);
  gaussian(qy);

  qz = qy ;
  qtmp = a*qx;
  qz += qtmp;

  // Now do in a onner.
  qtmp = qy + qx*a;
  d = qtmp - qz;
  dnorm = norm2(d);
  QDPIO::cout << "z=y + x*a: diff = " << dnorm << endl;

  // Test z = ax - y 
  gaussian(qx);
  gaussian(qy);

  qz = a*qx ;
  qz -= qy;

  // Now in a onner
  qtmp = qx*a-qy;
  d = qtmp - qz;
  dnorm = norm2(d);
  QDPIO::cout << "z=xa - y: diff = " << dnorm << endl;
  

  // Test z = y - ax
  gaussian(qx);
  gaussian(qy);

  qz = qy;
  qtmp = a*qx ;
  qz -= qtmp;

  // Now in a onner
  qtmp = qy - qx*a;
  d = qtmp - qz;
  dnorm = norm2(d);
  QDPIO::cout << "z=y - xa: diff = " << dnorm << endl;

  // Test z = x + y
  gaussian(qx);
  gaussian(qy);

  qz = qy;
  qtmp = qx ;
  qz += qtmp;

  // Now in a onner
  qtmp = qx+qy;
  d = qtmp - qz;
  dnorm = norm2(d);
  QDPIO::cout << "z=x + y: diff = " << dnorm << endl;

  // Test z = x - y
  gaussian(qx);
  gaussian(qy);

  qz = qx;
  qtmp = qy ;
  qz -= qtmp;

  // Now in a onner
  qtmp = qx-qy;
  d = qtmp - qz;
  dnorm = norm2(d);
  QDPIO::cout << "z=x - y: diff = " << dnorm << endl;


  // Test norm2(x)
  for(int site=all.start(); site <= all.end(); site++) {
        for(int spin=0; spin < Ns; spin++) {
           for(int col =0; col < Nc; col++) {
             qx.elem(site).elem(spin).elem(col).real().elem()=1;
             qx.elem(site).elem(spin).elem(col).imag().elem()=1;
           }
        }
  }
 
  // sum it by hand.
  Double rc = Double(0);
  for(int site=all.start(); site <= all.end(); site++) { 
	for(int spin=0; spin < Ns; spin++) { 
	   for(int col =0; col < Nc; col++) { 
	     rc += qx.elem(site).elem(spin).elem(col).real().elem()
		 * qx.elem(site).elem(spin).elem(col).real().elem();
	     rc += qx.elem(site).elem(spin).elem(col).imag().elem()
	         * qx.elem(site).elem(spin).elem(col).imag().elem();
           }
        }
  } 
  Internal::globalSum(rc);

  
  Double bjs = norm2(qx);
  QDPIO::cout << "lattice volume = " << Layout::vol() << " Ns = " << Ns << " Nc = " << Nc << " Ncompx = 2.  Total Sum should be = " << Layout::vol()*Ns*Nc*2 << endl;

  QDPIO::cout << "Hand sumsq-ed qx = " << rc << endl;
  QDPIO::cout << "norm2(qx) = " << bjs << endl;
  QDPIO::cout << "norm2 diff = " << rc - bjs << endl;

  // Timings
   // Test VSCAL
  int icnt;
  double tt;
  gaussian(qx);

  for(icnt=1; ; icnt <<= 1)
  {
    QDPIO::cout << "calling V=a*V " << icnt << " times" << endl;
    tt = QDP_SCALE(qy, a, qx, icnt);
    if (tt > 1)
      break;
  }

  {
    double rescale = 1000*1000 / double(Layout::sitesOnNode()) / icnt;
    tt *= rescale;
    int Nflops = 2*Ns*Nc;
    QDPIO::cout << "time(V=aV) = " << tt
		<< " micro-secs/site/iteration" 
		<< " , " << Nflops / tt << " Mflops" << endl;
  }


   // Test VAXPY
  gaussian(qx);
  gaussian(qy);

  for(icnt=1; ; icnt <<= 1)
  {
    QDPIO::cout << "calling V=aV+V " << icnt << " times" << endl;
    tt = QDP_AXPY(qz, a, qx, qy, icnt);
    if (tt > 1)
      break;
  }
  {
    double rescale = 1000*1000 / double(Layout::sitesOnNode()) / icnt;
    tt *= rescale;
    int Nflops = 4*Ns*Nc;
    QDPIO::cout << "time(V=aV+V) = " << tt
		<< " micro-secs/site/iteration" 
		<< " , " << Nflops / tt << " Mflops" << endl;
  }


   // Test VAXMY
  gaussian(qx);
  gaussian(qy);

  for(icnt=1; ; icnt <<= 1)
  {
    QDPIO::cout << "calling V=aV-V " << icnt << " times" << endl;
    tt = QDP_AXMY(qz, a, qx, qy, icnt);
    if (tt > 1)
      break;
  }

  {
    double rescale = 1000*1000 / double(Layout::sitesOnNode()) / icnt;
    tt *= rescale;
    int Nflops = 4*Ns*Nc;
    QDPIO::cout << "time(V=aV-V) = " << tt
		<< " micro-secs/site/iteration" 
		<< " , " << Nflops / tt << " Mflops" << endl;
  }

   // Test VADD
  gaussian(qx);
  gaussian(qy);
  for(icnt=1; ; icnt <<= 1)
  {
    QDPIO::cout << "calling V=V+V " << icnt << " times" << endl;
    tt = QDP_VADD(qz, qx, qy, icnt);
    if (tt > 1)
      break;
  }

  {
    double rescale = 1000*1000 / double(Layout::sitesOnNode()) / icnt;
    tt *= rescale;
    int Nflops = 2*Ns*Nc;
    QDPIO::cout << "time(V=V+V) = " << tt
		<< " micro-secs/site/iteration" 
		<< " , " << Nflops / tt << " Mflops" << endl;
    
  }

   // Test VSUB
  gaussian(qx);
  gaussian(qy);

  for(icnt=1; ; icnt <<= 1)
  {
    QDPIO::cout << "calling V=V-V " << icnt << " times" << endl;
    tt = QDP_VSUB(qz, qx, qy, icnt);
    if (tt > 1)
      break;
  }
  {
    
    double rescale = 1000*1000 / double(Layout::sitesOnNode()) / icnt;
    tt *= rescale;
    int Nflops = 2*Ns*Nc;
    QDPIO::cout << "time(V=V-V) = " << tt
		<< " micro-secs/site/iteration" 
		<< " , " << Nflops / tt << " Mflops" << endl;
    
  }

   // Test SUMSQ
  gaussian(qx);

  for(icnt=1; ; icnt <<= 1)
  {
    QDPIO::cout << "calling norm2(v) " << icnt << " times" << endl;
    tt = QDP_NORM2(qx, icnt);
    if (tt > 1)
      break;
  }
  {
    
    double rescale = 1000*1000 / double(Layout::sitesOnNode()) / icnt;
    tt *= rescale;
    int Nflops = 4*Ns*Nc; // Mult an Add for each complex component
    QDPIO::cout << "time(norm2(V)) = " << tt
		<< " micro-secs/site/iteration" 
		<< " , " << Nflops / tt << " Mflops" << endl;
    
  }

  // Time to bolt
  QDP_finalize();

  exit(0);
}
  
