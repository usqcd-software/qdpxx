// $Id: t_blas.cc,v 1.11 2005-05-25 04:20:33 edwards Exp $

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

  
  Real a=Real(1.5);
  LatticeFermion qx;
  LatticeFermion qy;
  LatticeFermion qz;
  LatticeFermion qtmp;
  LatticeFermion d;
  Double dnorm;
  Double dnorm2;

  // Test norm2(x)
  gaussian(qx);

  // sum it by hand.
  Double rc = Double(0);
  for(int site=all.start(); site <= all.end(); site++) { 
	for(int spin=0; spin < Ns; spin++) { 
	   for(int col =0; col < Nc; col++) { 
	     rc += qx.elem(site).elem(spin).elem(col).real()
		 * qx.elem(site).elem(spin).elem(col).real();
	     rc += qx.elem(site).elem(spin).elem(col).imag()
	         * qx.elem(site).elem(spin).elem(col).imag();
           }
        }
  } 
  Internal::globalSum(rc);

  
  Double bjs = norm2(qx);
//  QDPIO::cout << "lattice volume = " << Layout::vol() << " Ns = " << Ns << " Nc = " << Nc << " Ncompx = 2.  Total Sum should be = " << Layout::vol()*Ns*Nc*2 << endl;

  QDPIO::cout << "Hand sumsq-ed qx = " << rc << endl;
  QDPIO::cout << "norm2(qx) = " << bjs << endl;
  QDPIO::cout << "norm2 diff = " << rc - bjs << endl;



  // Test y += a*x
  gaussian(qx);
  gaussian(qy);

  /*
  // Hand roll it
  for(int site=all.start(); site <=all.end(); site++) {
    QDPIO::cout << "site=" << site << endl;
    for(int spin=0; spin < 4; spin++) {
      for(int col=0; col < 3; col++) {
       qx.elem(site).elem(spin).elem(col).real() = 1;
       qx.elem(site).elem(spin).elem(col).imag() = 2;

       qy.elem(site).elem(spin).elem(col).real() = 3;
       qy.elem(site).elem(spin).elem(col).imag() = 4;

      }
    }
  }
 
  // Hand roll it
  for(int site=all.start(); site <=all.end(); site++) { 
    QDPIO::cout << "site=" << site << endl;
    for(int spin=0; spin < 4; spin++) { 
      for(int col=0; col < 3; col++) { 
       qz.elem(site).elem(spin).elem(col).real() = 
          qy.elem(site).elem(spin).elem(col).real() 
        + (a.elem().elem().elem().elem() * 
           qx.elem(site).elem(spin).elem(col).real());

       qz.elem(site).elem(spin).elem(col).imag() =
          qy.elem(site).elem(spin).elem(col).imag()
        + (a.elem().elem().elem().elem() *
           qx.elem(site).elem(spin).elem(col).imag());
      }
    }
  } 
  */
  qtmp = a*qx;
  qz = qy + qtmp;

  // Now doit in a onner
  qy += a*qx;

  dnorm = norm2(qy);
  dnorm2 = norm2(qz);
  QDPIO::cout << "norm(qy) = " << dnorm << endl;
  QDPIO::cout << "norm(qz) = " << dnorm2 << endl;

  LatticeFermion diff_q;
  /*
    for(int site=all.start(); site <= all.end(); site++) {
    for(int spin=0; spin < 4; spin++) { 
    for(int col=0; col < 3; col++) { 
    QDPIO::cout << "qz = ( " <<
    qz.elem(site).elem(spin).elem(col).real() <<" , "  <<
    qz.elem(site).elem(spin).elem(col).imag() <<" ) " 
    << "  qy = ( " << 
    qy.elem(site).elem(spin).elem(col).real() <<" , "  << 
    qy.elem(site).elem(spin).elem(col).imag() <<" ) " << endl;
    
    }
    }
    }
  */
  diff_q = qz - qy;

  Double dnorm3=norm2(diff_q);
  QDPIO::cout << "diff = " << dnorm3 << endl;

  // QDPIO::cout << "y+=a*x: diff = " << dnorm << endl;


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

  LatticeFermion qtmp2,qtmp3;
  // Do AX + BY
  Real b = -3.2;
  gaussian(qx);
  gaussian(qy);

  qtmp = a*qx;
  qtmp += b*qy;
  qtmp2 = a*qx + b*qy;
  d = qtmp - qtmp2;
  dnorm = norm2(d);
  QDPIO::cout << "z=ax+by: diff = " << dnorm << endl;

  qtmp2 = qx*a + b*qy;
  d = qtmp - qtmp2;
  dnorm = norm2(d);
  QDPIO::cout << "z=xa+by: diff = " << dnorm << endl;

  qtmp2 = a*qx + qy*b;
  d = qtmp - qtmp2;
  dnorm = norm2(d);
  QDPIO::cout << "z=ax+yb: diff = " << dnorm << endl;

  qtmp2 = qx*a+ qy*b;
  d = qtmp - qtmp2;
  dnorm = norm2(d);
  QDPIO::cout << "z=xa+yb: diff = " << dnorm << endl;

  
  gaussian(qx);
  gaussian(qy);

  qtmp = a*qx;
  qtmp -= b*qy;
  qtmp2 = a*qx - b*qy;
  d = qtmp - qtmp2;
  dnorm = norm2(d);
  QDPIO::cout << "z=ax-by: diff = " << dnorm << endl;

  qtmp2 = qx*a - b*qy;
  d = qtmp - qtmp2;
  dnorm = norm2(d);
  QDPIO::cout << "z=xa-by: diff = " << dnorm << endl;

  qtmp2 = a*qx - qy*b;
  d = qtmp - qtmp2;
  dnorm = norm2(d);
  QDPIO::cout << "z=ax-yb: diff = " << dnorm << endl;

  qtmp2 = qx*a - qy*b;
  d = qtmp - qtmp2;
  dnorm = norm2(d);
  QDPIO::cout << "z=xa-yb: diff = " << dnorm << endl;


  gaussian(qx);
  gaussian(qy);
  qtmp = a*qx;
  qtmp += b*qy;
  qx = a*qx + b*qy;
  d = qtmp - qx;
  dnorm = norm2(d);
  QDPIO::cout << "x=ax+by: diff = " << dnorm << endl;

  gaussian(qx);
  qtmp = a*qx;
  qtmp += b*qy;
  qx = qx*a + b*qy;
  d = qtmp - qx;
  dnorm = norm2(d);
  QDPIO::cout << "x=xa+by: diff = " << dnorm << endl;

  gaussian(qx);
  qtmp = a*qx;
  qtmp += b*qy;
  qx = a*qx + qy*b;
  d = qtmp - qx;
  dnorm = norm2(d);
  QDPIO::cout << "x=ax+yb: diff = " << dnorm << endl;

  gaussian(qx);
  qtmp = a*qx;
  qtmp += b*qy;;
  qx = qx*a+ qy*b;
  d = qtmp - qx;
  dnorm = norm2(d);
  QDPIO::cout << "x=xa+yb: diff = " << dnorm << endl;

  
  gaussian(qx);
  gaussian(qy);
  qtmp = a*qx;
  qtmp -= b*qy;
  qx = a*qx - b*qy;
  d = qtmp - qx;
  dnorm = norm2(d);
  QDPIO::cout << "x=ax-by: diff = " << dnorm << endl;

  gaussian(qx);
  qtmp = a*qx;
  qtmp -= b*qy;
  qx = qx*a - b*qy;
  d = qtmp - qx;
  dnorm = norm2(d);
  QDPIO::cout << "x=xa-by: diff = " << dnorm << endl;

  gaussian(qx);
  qtmp = a*qx;
  qtmp -= b*qy;
 
  qx = a*qx - qy*b;
  d = qtmp - qx;
  dnorm = norm2(d);
  QDPIO::cout << "x=ax-yb: diff = " << dnorm << endl;

  gaussian(qx);
  qtmp = a*qx;
  qtmp -= b*qy;
  qx = qx*a - qy*b;
  d = qtmp - qx;
  dnorm = norm2(d);
  QDPIO::cout << "x=xa-yb: diff = " << dnorm << endl;


  gaussian(qx);
  gaussian(qy);
  qtmp = a*qx;
  qtmp += b*qy;
  qy = a*qx + b*qy;
  d = qtmp - qy;
  dnorm = norm2(d);
  QDPIO::cout << "y=ax+by: diff = " << dnorm << endl;

  gaussian(qy);
  qtmp = a*qx;
  qtmp += b*qy;
  qy = qx*a + b*qy;
  d = qtmp - qy;
  dnorm = norm2(d);
  QDPIO::cout << "y=xa+by: diff = " << dnorm << endl;

  gaussian(qy);
  qtmp = a*qx;
  qtmp += b*qy;
  qy = a*qx + qy*b;
  d = qtmp - qy;
  dnorm = norm2(d);
  QDPIO::cout << "y=ax+yb: diff = " << dnorm << endl;

  gaussian(qy);
  qtmp = a*qx;
  qtmp += b*qy;;
  qy = qx*a+ qy*b;
  d = qtmp - qy;
  dnorm = norm2(d);
  QDPIO::cout << "y=xa+yb: diff = " << dnorm << endl;

  
  gaussian(qx);
  gaussian(qy);
  qtmp = a*qx;
  qtmp -= b*qy;
  qy = a*qx - b*qy;
  d = qtmp - qy;
  dnorm = norm2(d);
  QDPIO::cout << "y=ax-by: diff = " << dnorm << endl;

  gaussian(qy);
  qtmp = a*qx;
  qtmp -= b*qy;
  qy = qx*a - b*qy;
  d = qtmp - qy;
  dnorm = norm2(d);
  QDPIO::cout << "y=xa-by: diff = " << dnorm << endl;

  gaussian(qy);
  qtmp = a*qx;
  qtmp -= b*qy;
 
  qy = a*qx - qy*b;
  d = qtmp - qy;
  dnorm = norm2(d);
  QDPIO::cout << "y=ax-yb: diff = " << dnorm << endl;

  gaussian(qy);
  qtmp = a*qx;
  qtmp -= b*qy;
  qy = qx*a - qy*b;
  d = qtmp - qy;
  dnorm = norm2(d);
  QDPIO::cout << "y=xa-yb: diff = " << dnorm << endl;


  gaussian(qy);
  gaussian(qx);
  DComplex accum=cmplx(Double(0), Double(0));

  for(int site=all.start(); site <= all.end(); site++) { 
    for(int spin = 0; spin < Ns; spin++) { 
      for(int col = 0; col < Nc; col++) {
	RComplex<REAL> rca= conj(qy.elem(site).elem(spin).elem(col));
	RComplex<REAL> rcb= qx.elem(site).elem(spin).elem(col);

	DComplex ca=cmplx(Double(rca.real()), Double(rca.imag()));
	DComplex cb=cmplx(Double(rcb.real()), Double(rcb.imag()));

	accum += ca*cb;
      }
    }
  }
  Internal::globalSum(accum); 
  Complex fred = innerProduct(qy, qx);
  DComplex dfred = cmplx(Double(fred.elem().elem().elem().real()),
			 Double(fred.elem().elem().elem().imag()));

  DComplex diff = accum - dfred;
  QDPIO::cout << "Diff innerProduct = " << diff << endl;

  accum = cmplx(Double(0), Double(0));
  for(int site=(rb[1]).start(); site <= (rb[1]).end(); site++) { 
    for(int spin = 0; spin < Ns; spin++) { 
      for(int col = 0; col < Nc; col++) {
	RComplex<REAL> rca= conj(qy.elem(site).elem(spin).elem(col));
	RComplex<REAL> rcb= qx.elem(site).elem(spin).elem(col);

	DComplex ca=cmplx(Double(rca.real()), Double(rca.imag()));
	DComplex cb=cmplx(Double(rcb.real()), Double(rcb.imag()));

	accum += ca*cb;
      }
    }
  }
  Internal::globalSum(accum); 
  fred = innerProduct(qy, qx, rb[1]);
  dfred = cmplx(Double(fred.elem().elem().elem().real()),
			 Double(fred.elem().elem().elem().imag()));

  diff = accum - dfred;
  QDPIO::cout << "Diff innerProduct Subset = " << diff << endl;

  Double daccum = Double(0);
  for(int site=all.start(); site <= all.end(); site++) { 
    for(int spin = 0; spin < Ns; spin++) { 
      for(int col = 0; col < Nc; col++) {
	RComplex<REAL> rca= conj(qy.elem(site).elem(spin).elem(col));
	RComplex<REAL> rcb= qx.elem(site).elem(spin).elem(col);

	DComplex ca=cmplx(Double(rca.real()), Double(rca.imag()));
	DComplex cb=cmplx(Double(rcb.real()), Double(rcb.imag()));

	DComplex dctmp = ca*cb;
	daccum += Double(dctmp.elem().elem().elem().real());
      }
    }
  }
  Internal::globalSum(daccum); 
  Double djim = innerProductReal(qy, qx);

  Double drdiff = daccum - djim;
  QDPIO::cout << "Diff innerProductReal all = " << drdiff << endl;

  daccum = Double(0);
  for(int site=(rb[1]).start(); site <= (rb[1]).end(); site++) { 
    for(int spin = 0; spin < Ns; spin++) { 
      for(int col = 0; col < Nc; col++) {
	RComplex<REAL> rca= conj(qy.elem(site).elem(spin).elem(col));
	RComplex<REAL> rcb= qx.elem(site).elem(spin).elem(col);

	DComplex ca=cmplx(Double(rca.real()), Double(rca.imag()));
	DComplex cb=cmplx(Double(rcb.real()), Double(rcb.imag()));

	DComplex dctmp = ca*cb;
	daccum += Double(dctmp.elem().elem().elem().real());
      }
    }
  }
  Internal::globalSum(daccum); 
  djim = innerProductReal(qy, qx,rb[1]);

  drdiff = daccum - djim;
  QDPIO::cout << "Diff innerProductReal Subset = " << drdiff << endl;

  // Test array innerProduct
  int NN = 8;
  multi1d<LatticeFermion> lqx(NN);
  multi1d<LatticeFermion> lqy(NN);

  accum = zero;
  for(int i=0; i < NN; ++i)
  {
    gaussian(lqx[i]);
    gaussian(lqy[i]);

    accum += innerProduct(lqx[i],lqy[i]);
  }
  fred = innerProduct(lqx,lqy);
  QDPIO::cout << "Diff innerProduct(multi1d) all = " << Complex(fred-accum) << endl;


  Double daccr = zero;
  for(int i=0; i < NN; ++i)
    daccr += innerProductReal(lqx[i],lqy[i]);

  Double dreal = innerProductReal(lqx,lqy);
  QDPIO::cout << "Diff innerProductReal(multi1d) all = " << Real(daccr-dreal) << endl;


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

   // Test VAXPBY
  gaussian(qx);
  gaussian(qy);

  for(icnt=1; ; icnt <<= 1)
  {
    QDPIO::cout << "calling V=aV+bV " << icnt << " times" << endl;
    tt = QDP_AXPBY(qz, a, b, qx, qy, icnt);
    if (tt > 1)
      break;
  }
  {
    double rescale = 1000*1000 / double(Layout::sitesOnNode()) / icnt;
    tt *= rescale;
    int Nflops = 3*2*Ns*Nc;
    QDPIO::cout << "time(V=aV+bV) = " << tt
		<< " micro-secs/site/iteration" 
		<< " , " << Nflops / tt << " Mflops" << endl;
  }


   // Test VAXMBY
  gaussian(qx);
  gaussian(qy);

  for(icnt=1; ; icnt <<= 1)
  {
    QDPIO::cout << "calling V=aV-bV " << icnt << " times" << endl;
    tt = QDP_AXMBY(qz, a, b, qx, qy, icnt);
    if (tt > 1)
      break;
  }

  {
    double rescale = 1000*1000 / double(Layout::sitesOnNode()) / icnt;
    tt *= rescale;
    int Nflops = 3*2*Ns*Nc;
    QDPIO::cout << "time(V=aV-bV) = " << tt
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

   // Test INNER_PRODUCT
  gaussian(qx);
  gaussian(qy);

  for(icnt=1; ; icnt <<= 1)
  {
    QDPIO::cout << "calling innerProduct(v,v)) " << icnt << " times" << endl;
    tt = QDP_INNER_PROD(qx, qy, icnt);
    if (tt > 1)
      break;
  }
  {
    
    double rescale = 1000*1000 / double(Layout::sitesOnNode()) / icnt;
    tt *= rescale;
    int Nflops = 8*Ns*Nc; // 2Mults and 2Adds for each complex component
    QDPIO::cout << "time(innerProduct(V,V)) = " << tt
		<< " micro-secs/site/iteration" 
		<< " , " << Nflops / tt << " Mflops" << endl;
  }

  // Test INNER_PRODUCT_REAL
  gaussian(qx);
  gaussian(qy);
  for(icnt=1; ; icnt <<= 1)
  {
    QDPIO::cout << "calling innerProductReal(V,V) " << icnt << " times" << endl;
    tt = QDP_INNER_PROD_REAL(qx,qy, icnt);
    if (tt > 1)
      break;
  }
  {
    
    double rescale = 1000*1000 / double(Layout::sitesOnNode()) / icnt;
    tt *= rescale;
    int Nflops = 4*Ns*Nc; // Mult an Add for each complex component
    QDPIO::cout << "time(innerProductReal(V,V)) = " << tt
		<< " micro-secs/site/iteration" 
		<< " , " << Nflops / tt << " Mflops" << endl;
  }


  // Test SUMSQ array
  for(icnt=1; ; icnt <<= 1)
  {
    QDPIO::cout << "calling norm2(multi1d<V>) " << icnt << " times" << endl;
    tt = QDP_NORM2(lqx, icnt);
    if (tt > 1)
      break;
  }
  {
    
    double rescale = 1000*1000 / double(Layout::sitesOnNode()) / icnt;
    tt *= rescale;
    int Nflops = lqx.size()*4*Ns*Nc; // Mult an Add for each complex component
    QDPIO::cout << "time(norm2(multi1d<V>)) = " << tt
		<< " micro-secs/site/iteration" 
		<< " , " << Nflops / tt << " Mflops" << endl;
    
  }

   // Test INNER_PRODUCT array
  for(icnt=1; ; icnt <<= 1)
  {
    QDPIO::cout << "calling innerProduct(multi1d<V>,multi1d<V>)) " << icnt << " times" << endl;
    tt = QDP_INNER_PROD(lqx, lqy, icnt);
    if (tt > 1)
      break;
  }
  {
    
    double rescale = 1000*1000 / double(Layout::sitesOnNode()) / icnt;
    tt *= rescale;
    int Nflops = lqx.size()*8*Ns*Nc; // 2Mults and 2Adds for each complex component
    QDPIO::cout << "time(innerProduct(multi1d<V>,multi1d<V>)) = " << tt
		<< " micro-secs/site/iteration" 
		<< " , " << Nflops / tt << " Mflops" << endl;
  }

  // Test INNER_PRODUCT_REAL array
  for(icnt=1; ; icnt <<= 1)
  {
    QDPIO::cout << "calling innerProductReal(multi1d<V>,multi1d<V>) " << icnt << " times" << endl;
    tt = QDP_INNER_PROD_REAL(lqx,lqy, icnt);
    if (tt > 1)
      break;
  }
  {
    
    double rescale = 1000*1000 / double(Layout::sitesOnNode()) / icnt;
    tt *= rescale;
    int Nflops = lqx.size()*4*Ns*Nc; // Mult an Add for each complex component
    QDPIO::cout << "time(innerProductReal(multi1d<V>,multi1d<V>)) = " << tt
		<< " micro-secs/site/iteration" 
		<< " , " << Nflops / tt << " Mflops" << endl;
  }


  // Time to bolt
  QDP_finalize();

  exit(0);
}
  
