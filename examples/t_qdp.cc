// $Id: t_qdp.cc,v 1.4 2002-09-26 21:27:36 edwards Exp $

#include <iostream>
#include <cstdio>

#include "tests.h"

using namespace QDP;


int main(int argc, char **argv)
{
  // Setup the geometry
  const int foo[Nd] = {2,2};
  multi1d<int> nrow(Nd);
  nrow = foo;
  geom.Init(nrow);

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
  LatticeGauge b1,b2,b3;

  b1 = 1.0;
  cerr << "b1 after fill\n" << b1 << endl;

  random(b1);
  cerr << "b1 after random\n" << b1 << endl;

  random(b2);
  gaussian(b3);
  cerr << "b3 after gaussian\n";
  Push(cerr,"test_stuff");
  Write(cerr,"b3",b3);
  WRITE_NAMELIST(cerr,b3);
  Pop(cerr);
  
#if 1
  junk(b3,b1,b2,all);
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
  multi1d<LatticeGauge> u(Nd);
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
  int length = geom.LattSize()[j_decay];
  multi2d<Real> meson_prop(Ns*Ns, length);
  multi1d<int> t_source(Nd);
  t_source = 0;

  mesons(quark_prop_1, quark_prop_2, meson_prop, t_source);
//  for(int n=0; n < Ns*Ns; ++n)
  for(int n=Ns*Ns-1; n < Ns*Ns; ++n)
    for(int t=0; t < length; ++t)
      cerr << "meson_prop["<<n<<","<<t<<"] = " << meson_prop[n][t] << endl;

#if 1
  multi2d<Complex> baryon_prop(9, length);
  baryon(quark_prop_1, baryon_prop, t_source, 1);
  for(int n=0; n < 9; ++n)
    for(int t=0; t < length; ++t)
      cerr << "baryon_prop["<<n<<","<<t<<"] = " << baryon_prop[n][t] << endl;
#endif

#endif

  // More frivolity and gaiety.
  LatticeFermion psi, chi;
  random(psi);
  zero(chi);
  dslash(chi, u, psi, +1, 0);

  WRITE_NAMELIST(cerr,psi);
  WRITE_NAMELIST(cerr,chi);

#if 1
  //! SU(N) fiddling
  cerr << "Fiddle with SU(2) matrices\n" << endl;
  LatticeReal r_0,r_1,r_2,r_3;

  cerr << "u[0] = " << u[0] << endl;
  cerr << "Start extract\n";
  for(int su2_index=0; su2_index < Nc*(Nc-1)/2; ++su2_index)
  {
    su2_extract(r_0,r_1,r_2,r_3, su2_index, u[0]);

    cerr << "su2_index="<<su2_index<<"\n";
    cerr << "r_0 = " << r_0 << endl;
    cerr << "r_1 = " << r_1 << endl;
    cerr << "r_2 = " << r_2 << endl;
    cerr << "r_3 = " << r_3 << endl;
  }
  cerr << "Start filling\n";
  for(int su2_index=0; su2_index < Nc*(Nc-1)/2; ++su2_index)
  {
    cerr << "su2_index="<<su2_index<<"\n";
    sun_fill(u[1], su2_index, r_0,r_1,r_2,r_3);

    cerr << "u[1] = " << u[1] << endl;
  }
#endif
}
