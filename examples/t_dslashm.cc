// $Id: t_dslashm.cc,v 1.2 2002-09-26 20:02:36 edwards Exp $

#include <iostream>
#include <cstdio>

#include "tests.h"

using namespace QDP;


int main(int argc, char **argv)
{
  // Setup the geometry
  const int foo[Nd] = {LX0,LX1};
  multi1d<int> nrow(Nd);
  nrow = foo;
  geom.Init(nrow);

  //! Test out propagators
  multi1d<LatticeGauge> u(Nd);
  for(int m=0; m < u.size(); ++m)
    gaussian(u[m]);

  LatticeFermion psi, chi;
  random(psi);
  zero(chi);
  dslash_2d_plus(chi, u, psi, 0);

//  dslash(chi, u, psi, +1, 0);

  WRITE_NAMELIST(cerr,psi);
  WRITE_NAMELIST(cerr,chi);

}
