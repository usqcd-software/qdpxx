// $Id: t_cblas.cc,v 1.1 2004-05-07 15:16:46 bjoo Exp $

#include <iostream>
#include <cstdio>

#include <time.h>

#include "qdp.h"

#include "scalarsite_generic/qdp_scalarsite_generic_cblas.h"
 
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

  LatticeFermion x, y, z1, z2,r;

  Complex alpha=cmplx(Real(1), Real(-2));

  
  // Do z1 = alpha x with an axpy
  z2 = zero;
  z1 = alpha*x + z2;
  
  // Do z2 = alpha x using *=
  z2 = x;
  z2 *= alpha;

  r = z1 - z2;
  Double diff = norm2(r);
  QDPIO::cout << " z2 *= a diff = " << sqrt(diff) << endl;

  // Do z2 = alpha * x using = 
  z2 = alpha*x;
  r = z1 - z2;
  diff = norm2(r);
  QDPIO::cout << " z2 = a * x  diff = " << sqrt(diff) << endl;

  // Do z2 = x * alpha using = 
  z2 = x*alpha;
  r = z1 - z2;
  diff = norm2(r);
  QDPIO::cout << " z2 = x * a diff = " << sqrt(diff) << endl;
  
  // Do z2 = x, z2 = a*z2
  z2 = x;
  z2 = alpha*z2;
  r = z1 - z2;
  diff = norm2(r);
  QDPIO::cout << "z2=x, z2 = a * z2 diff = " << sqrt(diff) << endl;


  gaussian(x);
  gaussian(y);

  z1 = alpha*x;
  z1 += y;

  z2 = y;
  z2 += alpha*x;

  r = z1 - z2;
  diff = norm2(r);
  QDPIO::cout << "z += alpha*x  diff = " << sqrt(diff) << endl;

  z2 = y;
  z2 += x*alpha;

  r = z1 - z2;
  diff = norm2(r);
  QDPIO::cout << "z += x*alpha  diff = " << sqrt(diff) << endl;


  z1 = alpha*x;
  z1 = -z1;
  z1 += y;

  z2 = y;
  z2 -= alpha*x;
  r = z1 - z2;
  diff = norm2(r);
  QDPIO::cout << "z -= x*alpha  diff = " << sqrt(diff) << endl;

  z2 = y;
  z2 -= x*alpha;
  r = z1 - z2;
  diff = norm2(r);
  QDPIO::cout << "z -= x*alpha  diff = " << sqrt(diff) << endl;

  
  z1 = alpha*x;
  z1 += y;

  z2 = alpha*x + y;
  r = z1 - z2;
  diff = norm2(r);
  QDPIO::cout << "z = alpha * x + y  diff = " << sqrt(diff) << endl;

  z2 = x * alpha + y;
  r = z1 - z2;
  diff = norm2(r);
  QDPIO::cout << "z = x * alpha + y  diff = " << sqrt(diff) << endl;

  z2 = y + alpha*x;
  r = z1 - z2;
  diff = norm2(r);
  QDPIO::cout << "z = y + alpha * x  diff = " << sqrt(diff) << endl;

  z2 = y + x*alpha;
  r = z1 - z2;
  diff = norm2(r);
  QDPIO::cout << "z = y + x * alpha  diff = " << sqrt(diff) << endl;


  z1 = alpha*x;
  z1 -= y;
  z2 = alpha*x - y;
  r = z1 - z2;
  diff = norm2(r);
  QDPIO::cout << "z = alpha * x - y  diff = " << sqrt(diff) << endl;

  z2 = x * alpha - y;
  r = z1 - z2;
  diff = norm2(r);
  QDPIO::cout << "z = x * alpha - y  diff = " << sqrt(diff) << endl;

  z1 = y;
  z1 -= alpha*x;

  z2 = y - alpha*x;
  r = z1 - z2;
  diff = norm2(r);
  QDPIO::cout << "z = y - alpha*x  diff = " << sqrt(diff) << endl;

  z2 = y - x*alpha;
  r = z1 - z2;
  diff = norm2(r);
  QDPIO::cout << "z = y - x*alpha  diff = " << sqrt(diff) << endl;


  // Time to bolt
  QDP_finalize();

  exit(0);
}
