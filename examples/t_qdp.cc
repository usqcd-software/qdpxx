// $Id: t_qdp.cc,v 1.20 2003-09-03 19:50:42 edwards Exp $
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
  cout << "After setrn" << endl;

  // Time to play...
  Real r1;
  r1 = 17.0;
  cout << "r1 after fill\n" << r1 << endl;

  for(int i=0; i < 10; ++i)
  {
    random(r1);
    cout << "r1 after random\n" << r1 << endl;
  }

  // Check the multi-dim arrays
  multi2d<Real> boo(2,3);
  boo = 0.0;
  cout << "Fill boo with 0\n";
  for(int j=0; j < 2; ++j)
    for(int i=0; i < 3; ++i)
      cout << boo(j,i) << endl;

  cout << "Fill boo with random\n";
  for(int j=0; j < 2; ++j)
    for(int i=0; i < 3; ++i)
      random(boo(j,i));

  cout << "Check boo filled with random\n";
  for(int j=0; j < 2; ++j)
    for(int i=0; i < 3; ++i)
      cout << boo(j,i) << endl;

  cout << "Test indexing of boo\n";
  for(int j=0; j < 2; ++j)
    for(int i=0; i < 3; ++i)
      cout << boo[j][i] << endl;

  // Check the multi-dim arrays
  multi3d<Real> goo(2,3,2);
  goo = 0.0;
  cout << "Fill goo with 0\n";
  for(int k=0; k < 2; ++k)
    for(int j=0; j < 3; ++j)
      for(int i=0; i < 2; ++i)
	cout << goo(k,j,i) << endl;

  cout << "Fill goo with random\n";
  for(int k=0; k < 2; ++k)
    for(int j=0; j < 3; ++j)
      for(int i=0; i < 2; ++i)
      {
	random(goo(k,j,i));
	cout << goo(k,j,i) << endl;
      }


  cout << "Check goo filled with random\n";
  for(int k=0; k < 2; ++k)
    for(int j=0; j < 3; ++j)
      for(int i=0; i < 2; ++i)
      {
	goo[k][j][i] = goo(k,j,i);
	cout << goo(k,j,i) << endl; 
      }
  
  cout << "Test indexing of goo\n";
  for(int k=0; k < 2; ++k)
    for(int j=0; j < 3; ++j)
      for(int i=0; i < 2; ++i)
	cout << goo[k][j][i] << endl;

  // Test out lattice fields
  LatticeColorMatrix b1,b2,b3;

  b1 = 1.0;
  cout << "b1 after fill\n" << endl;
  Write(nml,b1);

  random(b1);
  cout << "b1 after random\n" << endl;
  Write(nml,b1);

  random(b2);
  gaussian(b3);
  cout << "b3 after gaussian\n";
  push(nml,"test_stuff");
  write(nml,"b3",b3);
  Write(nml,b3);
  pop(nml);
  
#if 0
  Double dsum;
  dsum = norm2(b1);
  cout << "dsum = " << dsum << endl;
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
  cout << "Initialize vector of latticegauge\n";
  multi1d<LatticeColorMatrix> u(Nd);
  cout << "After initialize vector of latticegauge\n";
  Double w_plaq, s_plaq, t_plaq, link;

  cout << "Start gaussian\n";
  for(int m=0; m < u.size(); ++m)
    gaussian(u[m]);

  MesPlq(u, w_plaq, s_plaq, t_plaq, link);
  cout << "w_plaq = " << w_plaq << endl;
  cout << "link = " << link << endl;

#if 1
  // Play with gamma matrices - they should be implemented...
  //! Test out propagators
  LatticePropagator quark_prop_1, quark_prop_2;
  gaussian(quark_prop_1);
  gaussian(quark_prop_2);

  int j_decay = Nd-1;
  int length = Layout::lattSize()[j_decay];
  multi2d<Real> meson_prop;
  multi1d<int> t_source(Nd);
  t_source = 0;

  mesons(quark_prop_1, quark_prop_2, meson_prop, t_source, j_decay);
  Write(nml,meson_prop);

#if 1
  multi2d<Complex> baryon_prop;
  Write(nml,quark_prop_1);
  baryon(quark_prop_1, baryon_prop, t_source, j_decay, 1);
  Write(nml,t_source);
  Write(nml,j_decay);
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
