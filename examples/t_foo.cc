// -*- C++ -*-
//
// $Id: t_foo.cc,v 1.8 2003-01-22 16:48:04 edwards Exp $
//
/*! \file
 *  \brief Silly little internal test code
 */


#include "qdp.h"
#include "proto.h"

//using namespace std;

//using namespace QDP;

struct Nearest : public MapFunc
{
  Nearest() {}

  virtual multi1d<int> operator() (const multi1d<int>& coord, int sign) const
    {
      multi1d<int> lc = coord;

      const multi1d<int>& nrow = Layout::lattSize();
      int m = 1;
      lc[m] = (coord[m] + 1*sgnum(sign) + 4*nrow[m]) % nrow[m];

      return lc;
    }

private:
  int sgnum(int x) const {return (x > 0) ? 1 : -1;}
}; 


struct NearestNeighborMapFunc : public ArrayMapFunc
{
  NearestNeighborMapFunc() {}

  virtual multi1d<int> operator() (const multi1d<int>& coord, int sign, int dir) const
    {
      multi1d<int> lc = coord;

      const multi1d<int>& nrow = Layout::lattSize();
      lc[dir] = (coord[dir] + sgnum(sign) + 4*nrow[dir]) % nrow[dir];

      return lc;
    }

  virtual int numArray() const {return Nd;}

private:
  int sgnum(int x) const {return (x > 0) ? 1 : -1;}
}; 


NmlWriter nml;


int main(int argc, char *argv[])
{
  // Put the machine into a known state
  QDP_initialize(&argc, &argv);

  // Setup the layout
  const int foo[] = {4,4,2,2};
  multi1d<int> nrow(Nd);
  nrow = foo;  // Use only Nd elements
  Layout::setLattSize(nrow);
  Layout::create();

//  NmlWriter nml("foo.nml");
  nml.open("foo.nml");

  Write(nml,nrow);

#if 1
  {
    Map near;
    Nearest bbb;
    near.make(bbb);

    LatticeReal fred,sue;
    random(fred);

    sue = near(fred);

    Write(nml,fred);
    Write(nml,sue);
  }
#endif

#if 1
  {
    BiDirectionalMap near;
    Nearest bbb;
    near.make(bbb);

    LatticeReal fred,sue,sam;
    random(fred);

    sue = near(fred,+1);
    sam = near(fred,-1);

    Write(nml,fred);
    Write(nml,sue);
    Write(nml,sam);
  }
#endif

#if 1
  {
    ArrayMap nearneigh;
    NearestNeighborMapFunc bbb;
    nearneigh.make(bbb);

    LatticeReal fred,sue;
    random(fred);

    Write(nml,fred);
    for(int m=0; m < Nd; ++m)
    {
      sue = nearneigh(fred,m);
      Write(nml,m);
      Write(nml,sue);
    }
  }
#endif

#if 1
  {
    ArrayBiDirectionalMap nearneigh;
    NearestNeighborMapFunc bbb;
    nearneigh.make(bbb);

//    LatticeReal fred,sue,sam;
    LatticeColorMatrix fred;
    random(fred);

    Write(nml,fred);
    for(int m=0; m < Nd; ++m)
    {
      LatticeColorMatrix sue = nearneigh(fred,+1,m);
      LatticeColorMatrix sam;
      sam = nearneigh(fred,-1,m);

      Write(nml,m);
      Write(nml,sue);
      Write(nml,sam);
    }
  }
#endif

//  exit(0);

  cerr << "try a string\n";
  nml << "write a string";

  Real one = 1.0;
  cerr << "try a scalar\n";
  nml << "write a scalar";
  write(nml,"test this",one);
//  exit(0);

  cerr << "latticereal\n";
  LatticeReal bar = 17.0;
  random(bar);

//  for(int i=0; i < Layout::subgridVol(); ++i)
//    cerr << "bar[" << i << "]=" << bar.elem(i) << endl;

  cerr << "write\n";
  Write(nml,bar);
  cerr << "done write\n";

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

  b[rb[1]] = zero;

  nml << "First set of b";
  Write(nml,b);

  random(b);

  nml << "Second set of b";
  Write(nml,b);

  b[rb[0]] = -17;

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
  LatticeColorMatrix barf = 1.0;
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
  LatticeColorMatrix barfagain = pokeColor(barf,fred,0,0);
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
    cerr << "enter some data";
    TextReader from("input");
    from >> x;
  }

  cerr << "you entered :" << x << ":";
  
  // Zero out a and read it again
  a = zero;

  {
    BinaryReader from("fred.bin");
    read(from,a);
  }

  nml << "Reset and reread a";
  Write(nml,a);

#endif

#if 0
  if (Nd == 4)
  {
    // Read a nersc file
    multi1d<LatticeColorMatrix> u(Nd);
    readArchiv(u, "archiv.cfg");
    nml << "Here is the nersc archive u field";
    Write(nml,u);
  }
#endif


#if 1
  if (Nd == 4)
  {
    // Read a szin gauge file
    multi1d<LatticeColorMatrix> u(Nd);
    Seed seed_old;
    cerr << "Calling szin reader" << endl;
    readSzin(u, 0, "szin.cfg", seed_old);
    nml << "Here is the SZIN u field";
    Write(nml,u);
  }
#endif


#if 0
  if (Nd == 4)
  {
    // Read a szin quark prop file
    LatticePropagator qprop;
    cerr << "Read a szin qprop" << endl;
    readSzinQprop(qprop, "propagator_0");
    nml << "Here is the szin quark prop";
    Write(nml,qprop);
  }
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

  // Time to bolt
  QDP_finalize();

  return 0;
}
