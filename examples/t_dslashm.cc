// $Id: t_dslashm.cc,v 1.5 2002-11-13 02:33:53 edwards Exp $

#include <iostream>
#include <cstdio>

#include "tests.h"

using namespace QDP;


int main(int argc, char **argv)
{
  // Setup the geometry
  const int foo[] = {LX0,LX1,LX2,LX3};
  multi1d<int> nrow(Nd);
  nrow = foo;  // Use only Nd elements
  geom.init(nrow);

  //! Test out propagators
  multi1d<LatticeGauge> u(Nd);
  for(int m=0; m < u.size(); ++m)
    gaussian(u[m]);

  LatticeFermion psi, chi;
  random(psi);
  chi = zero;
  dslash_2d_plus(chi, u, psi, 0);

//  dslash(chi, u, psi, +1, 0);

  NmlWriter nml("t_dslashm.nml");
  Write(nml,psi);
  Write(nml,chi);

}
