// -*- C++ -*-
//
// $Id: foo.cc,v 1.5 2002-10-02 20:29:37 edwards Exp $
//
// Silly little internal test code

//#include <iostream.h>
#include <cstdio>
#include <cstdlib>

#include "qdp.h"

//using namespace QDP;

int main()
{
  // Setup the geometry
  const int foo[Nd] = {LX0,LX1};
  multi1d<int> nrow(Nd);
  nrow = foo;
  layout.Init(nrow);

#if 0
  // Initialize the random number generator
  Seed seed;
  seed = 11;
  cerr << seed << endl;
  RNG::setrn(seed);
#endif


//  typedef QDPType<int> LatticeInteger;
//  typedef OLattice<int> LatticeInteger;
//  typedef QDPType<OLattice<int> > LatticeInteger;
//  LatticeComplexInteger a, b, c;
//  LatticeReal a, b, c;
  LatticeReal a, b, c, e;
  Complex d;
  float ccc = 2.0;
  float x;
  
#if 1
  b = 3;
  c = 4;
  d = 5;

  cerr << "First set of b\n";
  cerr << b << endl;

  random(b);

  cerr << "Second set of b\n";
  cerr << b << endl;

  b(rb[0]) = -17;

  cerr << "Third set of b\n";
  cerr << b << endl;

//  b.elem(1).elem() = -1;
//  b.elem().elem(1).elem() = -1;
//  b.elem().elem(1) = -1;

  cerr << "here 0\n";
#endif
//  a = -b + c*d;
//  a = c*d - b;
//  a = b*c;
//  a = -b + ccc*c;
//  a = -b + 2*c;
  a = shift(b*c,FORWARD,0);
//  a = b*c;
//  a = ccc*c;
//  x = ccc*c;

  cerr << "here b\n";
  cerr << b << endl;
  cerr << "here a\n";
  cerr << a << endl;

  cerr << "here e\n";
  LatticeReal rr;
//  random(rr);
  rr = 0.2;
  e = where(rr < 0.5, a, c);
  cerr << e << endl;



#if 0

  a = -b + ccc*c;
//  a = ccc*c;

  cerr << "here c\n";
  cerr << a << endl;
#endif

#if 0
  std::ofstream f;
  f.open("foobar",std::ios_base::out|std::ios_base::binary);
  float aa[3] = {0.0,0.0,0.0};
  std::fwrite(&aa,sizeof(LatticeComplex),1,f.rdbuf());
  f.close();


  BinaryWriter to("fred");
  to << a;
  to.close();

  cerr << "enter some data\n";
  TextReader from("input");
  from >> x;
  from.close();

  cerr << "you entered :" << x << ":\n";
  
#endif


#if 0
  LatticeInteger d;
  cerr << "here c\n";
  const Expression<BinaryNode<OpAdd,
    Reference<LatticeInteger>, Reference<LatticeInteger> > > &expr1 = b + c;
  cerr << "here d\n";
  d = expr1;
  cerr << "here e\n";
  cerr << d << endl;
  cerr << "here f\n";
  
  int num = forEach(expr1, CountLeaf(), SumCombine());
  cerr << num << endl;

//  const Expression<BinaryNode<OpAdd, Reference<LatticeInteger>, 
//    BinaryNode<OpMultiply, Scalar<int>,
//    Reference<LatticeInteger> > > > &expr2 = b + 3 * c;
//  num = forEach(expr2, CountLeaf(), SumCombine());
//  cerr << num << endl;
  
  const Expression<BinaryNode<OpAdd, Reference<LatticeInteger>, 
    BinaryNode<OpMultiply, Reference<LatticeInteger>,
    Reference<LatticeInteger> > > > &expr3 = b + c * d;
  num = forEach(expr3, CountLeaf(), SumCombine());
  cerr << num << endl;
#endif
}
