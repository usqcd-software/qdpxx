// $Id: junk.cc,v 1.7 2004-02-11 10:33:09 bjoo Exp $

#include "tests.h"

using namespace QDP;

void junk(NmlWriter& nml, 
	  LatticeColorMatrix& b3, const LatticeColorMatrix& b1, const LatticeColorMatrix& b2, 
	  const Subset& s)
{
//  Fermion f1, f2;
//  Gamma g(7);
//  f2 = g * f1;

//  b3 = b1 * b2;
  Double sum;

  nml << "b1 before shift";
  write(nml, "b1", b1);
  
  sum = norm2(b1);
  QDPIO::cerr << "Norm2 before shift = " << sum << endl;
  nml << "Norm2 before shift";
  write(nml,"sum", sum);

  b3 = shift(b1,FORWARD,0);

  nml << "b3 after shift of b1";
  write(nml,"b3", b3);

  sum = norm2(b3);
  nml << "Norm2 after shift";
  write(nml, "sum", sum);
  QDPIO::cerr << "Norm2 after shift = " << sum << endl;

  sum = innerproductReal(b3,b3);
  QDPIO::cerr << "Inner product = " << sum << endl;
  nml << "Inner product";
  write(nml,"sum", sum);

  DComplex dcsum;
  dcsum = innerproduct(b3,b3);
  QDPIO::cerr << "Complex Inner product = " << dcsum << endl;

//  fprintf(stderr,"Test 3\n");
//  b3 = b1*b2;

//  fprintf(stderr,"Test 4\n");
//  b3 = adj(b1)*b2;

//  fprintf(stderr,"Test 5\n");
//  b3 = b1*b1*b2;
}
