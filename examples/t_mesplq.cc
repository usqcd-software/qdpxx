// $Id: t_mesplq.cc,v 1.1 2002-09-12 18:22:17 edwards Exp $

#include <iostream>
#include <cstdio>

#include "tests.h"

using namespace QDP;


int main(int argc, char **argv)
{
  // Setup the geometry
  const int foo[Nd] = {4,4};
  multi1d<int> nrow(Nd);
  for(int i=0; i < nrow.size(); ++i)
    nrow[i] = foo[i];
  geom.Init(nrow);

  // Initialize the random number generator
  multi1d<int> seed(4);
  seed = 0;
  seed[0] = 11;
  RNG::setrn(seed);

  //! Test out propagators
  multi1d<LatticeGauge> u(Nd);
  for(int m=0; m < u.size(); ++m)
    Gaussian(u[m]);

  LatticePropagator quark_prop_1, quark_prop_2;
  Gaussian(quark_prop_1);
  Gaussian(quark_prop_2);

  int j_decay = Nd-1;
  int length = geom.LattSize()[j_decay];
  multi1d<int> t_source(Nd);
  t_source = 0;

  int t_sink = 3;

  FormFac(u, quark_prop_1, quark_prop_2, t_source, t_sink, j_decay);
}
