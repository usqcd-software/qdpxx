// -*- C++ -*-
//
// $Id: foo.cc,v 1.9 2002-10-26 01:54:30 edwards Exp $
//
// Silly little internal test code

#include "qdp.h"
#include "proto.h"

//using namespace std;

//using namespace QDP;

int main()
{
  // Setup the geometry
  const int foo[Nd] = {LX0,LX1,LX2,LX3};
  multi1d<int> nrow(Nd);
  nrow = foo;
  layout.Init(nrow);

  NmlWriter nml("foo.nml");

#if 0
  // Initialize the random number generator
  Seed seed;
  seed = 11;
  nml << seed;
  RNG::setrn(seed);
#endif


//  typedef QDPType<int> LatticeInteger;
//  typedef OLattice<int> LatticeInteger;
//  typedef QDPType<OLattice<int> > LatticeInteger;
//  LatticeComplexInteger a, b, c;
//  LatticeReal a, b, c;
  LatticeReal a, b, c, e;
  Complex d;
//  float ccc = 2.0;
//  float x;
  
#if 0
  LatticeComplex  foob(zero);
  nml << "Here is foob";
  Write(nml,foob);
#endif


#if 1
  b = 3;
  c = 4;
  d = 5;

  b(rb[1]) = zero;

  nml << "First set of b";
  Write(nml,b);

  random(b);

  nml << "Second set of b";
  Write(nml,b);

  b(rb[0]) = -17;

  nml << "Third set of b";
  Write(nml,b);

//  b.elem(1).elem() = -1;
//  b.elem().elem(1).elem() = -1;
//  b.elem().elem(1) = -1;

  nml << "here 0";
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

  nml << "here b";
  Write(nml,b);
  nml << "here a";
  Write(nml,a);

  nml << "here e";
  LatticeReal rr;
//  random(rr);
  rr = 0.2;
  e = where(rr < 0.5, a, c);
  Write(nml,e);


#if 1
  nml << "test peekColor";
  LatticeGauge barf = 1.0;
  LatticeComplex fred = peekColor(barf,0,0);
  Write(nml,barf);
  Write(nml,fred);

  multi1d<int> ccrd(Nd);
  ccrd = 1;

  nml << "test peeksite";
  random(fred);
  Complex sitefred = peekSite(fred,ccrd);
  Write(nml,fred);
  Write(nml,sitefred);


  nml << "test pokeColor";
  random(barf);
  fred = 1.0;
  LatticeGauge barfagain = pokeColor(barf,fred,0,0);
  Write(nml,barfagain);

  nml << "test pokeSite";
  random(sitefred);
  LatticeComplex fredagain = pokeSite(fred,sitefred,ccrd);
  Write(nml,fredagain);
#endif


#if 0

  a = -b + ccc*c;
//  a = ccc*c;

  nml << "here a";
  Write(nml,a);
#endif

#if 0
//  std::ofstream f;
//  f.open("foobar",std::ios_base::out|std::ios_base::binary);
//  float aa[3] = {0.0,0.0,0.0};
//  std::fwrite(&aa,sizeof(LatticeComplex),1,f.rdbuf());
//  f.close();


  {
    NmlWriter to("fred.txt");
    write(to,"a",a);
  }

  {
    cerr << "open fred.bin\n";
    BinaryWriter to("fred.bin");
    write(to,a);
  }

  {
    cout << "enter some data";
    TextReader from("input");
    from >> x;
  }

  cout << "you entered :" << x << ":";
  
  // Zero out a and read it again
  a = zero;

  {
    BinaryReader from("fred.bin");
    read(from,a);
  }

  nml << "Reset and reread a";
  Write(nml,a);

#endif

#if 1
  // Read a nersc file
  multi1d<LatticeGauge> u(Nd);
  readArchiv(u, "archiv.cfg");
  nml << "Here is the nersc archive u field";
  Write(nml,u);
#endif


#if 0
  LatticeInteger d;
  nml << "here c";
  const Expression<BinaryNode<OpAdd,
    Reference<LatticeInteger>, Reference<LatticeInteger> > > &expr1 = b + c;
  nml << "here d";
  d = expr1;
  nml << "here e";
  nml << d;
  nml << "here f";
  
  int num = forEach(expr1, CountLeaf(), SumCombine());
  nml << num << endl;

//  const Expression<BinaryNode<OpAdd, Reference<LatticeInteger>, 
//    BinaryNode<OpMultiply, Scalar<int>,
//    Reference<LatticeInteger> > > > &expr2 = b + 3 * c;
//  num = forEach(expr2, CountLeaf(), SumCombine());
//  nml << num << endl;
  
  const Expression<BinaryNode<OpAdd, Reference<LatticeInteger>, 
    BinaryNode<OpMultiply, Reference<LatticeInteger>,
    Reference<LatticeInteger> > > > &expr3 = b + c * d;
  num = forEach(expr3, CountLeaf(), SumCombine());
  nml << num << endl;
#endif
}
