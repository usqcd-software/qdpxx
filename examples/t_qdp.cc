// $Id: t_qdp.cc,v 1.19 2003-06-07 19:09:32 edwards Exp $
//
/*! \file
 *  \brief Silly little internal test code
 *
 */

#include <iostream>
#include <cstdio>

#include "qdp.h"
#include "examples.h"
#include "qdp_util.h"

using namespace QDP;


int main(int argc, char **argv)
{
  // Put the machine into a known state
  QDP_initialize(&argc, &argv);

  // Setup the layout
  const int foo[] = {2,2,2,2};
  multi1d<int> nrow(Nd);
  nrow = foo;  // Use only Nd elements
  Layout::setLattSize(nrow);
  Layout::create();

  // Open a file for some sample output
  NmlWriter nml("t_qdp.nml");

  // Initialize the random number generator
  Seed seed;
  seed = 11;
  RNG::setrn(seed);
  cerr << "After setrn" << endl;

  // Time to play...
  Real r1;
  r1 = 17.0;
  cerr << "r1 after fill\n" << r1 << endl;

  for(int i=0; i < 10; ++i)
  {
    random(r1);
    cerr << "r1 after random\n" << r1 << endl;
  }

  // Check the multi-dim arrays
  multi2d<Real> boo(2,3);
  boo = 0.0;
  cerr << "Fill boo with 0\n";
  for(int j=0; j < 2; ++j)
    for(int i=0; i < 3; ++i)
      cerr << boo(j,i) << endl;

  cerr << "Fill boo with random\n";
  for(int j=0; j < 2; ++j)
    for(int i=0; i < 3; ++i)
      random(boo(j,i));

  cerr << "Check boo filled with random\n";
  for(int j=0; j < 2; ++j)
    for(int i=0; i < 3; ++i)
      cerr << boo(j,i) << endl;

  cerr << "Test indexing of boo\n";
  for(int j=0; j < 2; ++j)
    for(int i=0; i < 3; ++i)
      cerr << boo[j][i] << endl;

  // Check the multi-dim arrays
  multi3d<Real> goo(2,3,2);
  goo = 0.0;
  cerr << "Fill goo with 0\n";
  for(int k=0; k < 2; ++k)
    for(int j=0; j < 3; ++j)
      for(int i=0; i < 2; ++i)
	cerr << goo(k,j,i) << endl;

  cerr << "Fill goo with random\n";
  for(int k=0; k < 2; ++k)
    for(int j=0; j < 3; ++j)
      for(int i=0; i < 2; ++i)
      {
	random(goo(k,j,i));
	cerr << goo(k,j,i) << endl;
      }


  cerr << "Check goo filled with random\n";
  for(int k=0; k < 2; ++k)
    for(int j=0; j < 3; ++j)
      for(int i=0; i < 2; ++i)
      {
	goo[k][j][i] = goo(k,j,i);
	cerr << goo(k,j,i) << endl; 
      }
  
  cerr << "Test indexing of goo\n";
  for(int k=0; k < 2; ++k)
    for(int j=0; j < 3; ++j)
      for(int i=0; i < 2; ++i)
	cerr << goo[k][j][i] << endl;

  // Test out lattice fields
  LatticeColorMatrix b1,b2,b3;

  b1 = 1.0;
  cerr << "b1 after fill\n" << endl;
  Write(nml,b1);

  random(b1);
  cerr << "b1 after random\n" << endl;
  Write(nml,b1);

  random(b2);
  gaussian(b3);
  cerr << "b3 after gaussian\n";
  push(nml,"test_stuff");
  write(nml,"b3",b3);
  Write(nml,b3);
  pop(nml);
  
#if 0
  Double dsum;
  dsum = norm2(b1);
  cerr << "dsum = " << dsum << endl;
  nml << "dsum = ";
  Write(nml,dsum);

  junk(nml,b3,b1,b2,all);
#endif

#if 1
  // Test comparisons and mask operations
  LatticeBoolean lbtmp1, lbtmp2;
  LatticeReal lftmp1, lftmp2;

  random(lftmp1);
  random(lftmp2);
  lbtmp1 = lftmp1 < lftmp2;
#endif

  //! Example of calling a plaquette routine
  /*! NOTE: the STL is *not* used to hold gauge fields */
  cerr << "Initialize vector of latticegauge\n";
  multi1d<LatticeColorMatrix> u(Nd);
  cerr << "After initialize vector of latticegauge\n";
  Double w_plaq, s_plaq, t_plaq, link;

  cerr << "Start gaussian\n";
  for(int m=0; m < u.size(); ++m)
    gaussian(u[m]);

  MesPlq(u, w_plaq, s_plaq, t_plaq, link);
  cerr << "w_plaq = " << w_plaq << endl;
  cerr << "link = " << link << endl;

#if 1
  // Play with gamma matrices - they should be implemented...
  //! Test out propagators
  LatticePropagator quark_prop_1, quark_prop_2;
  gaussian(quark_prop_1);
  gaussian(quark_prop_2);

  int j_decay = Nd-1;
  int length = Layout::lattSize()[j_decay];
  multi2d<Real> meson_prop(Ns*Ns, length);
  multi1d<int> t_source(Nd);
  t_source = 0;

  mesons(quark_prop_1, quark_prop_2, meson_prop, t_source, j_decay);
//  for(int n=0; n < Ns*Ns; ++n)
  for(int n=Ns*Ns-1; n < Ns*Ns; ++n)
    for(int t=0; t < length; ++t)
      cerr << "meson_prop["<<n<<","<<t<<"] = " << meson_prop[n][t] << endl;

#if 1
  multi2d<Complex> baryon_prop(9, length);
  baryon(quark_prop_1, baryon_prop, t_source, j_decay, 1);
  for(int n=0; n < 9; ++n)
    for(int t=0; t < length; ++t)
      Write(nml,baryon_prop);
#endif

#endif

  // More frivolity and gaiety.
  LatticeFermion psi, chi;
  random(psi);
  chi = zero;
  dslash(chi, u, psi, +1, 0);

  Write(nml,psi);
  Write(nml,chi);

  // Time to bolt
  QDP_finalize();

  exit(0);
}
