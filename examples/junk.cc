// $Id: junk.cc,v 1.1 2002-09-12 18:22:17 edwards Exp $

#include "tests.h"

using namespace QDP;

void junk(LatticeGauge& b3, const LatticeGauge& b1, const LatticeGauge& b2, const Subset& s)
{
  Fermion f1, f2;
  Gamma g(7);
  f2 = g * f1;

  b3 = b1 * b2;
  Double sum;

  sum = sumsq(b1);
  cerr << "Sumsq before shift = " << sum << endl;

  b3 = shift(b1,FORWARD,0);

  sum = sumsq(b3);
  cerr << "Sumsq after shift = " << sum << endl;

  sum = innerproduct_real(b3,b3);
  cerr << "Inner product = " << sum << endl;

  DComplex dcsum;
  dcsum = innerproduct(b3,b3);
  cerr << "Complex Inner product = " << dcsum << endl;

//  fprintf(stderr,"Test 3\n");
//  b3 = b1*b2;

//  fprintf(stderr,"Test 4\n");
//  b3 = conj(b1)*b2;

//  fprintf(stderr,"Test 5\n");
//  b3 = b1*b1*b2;
}
