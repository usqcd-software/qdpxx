// $Id: t_dslashm.cc,v 1.11 2003-01-15 21:46:51 edwards Exp $
/*! \file
 *  \brief Test the Wilson-Dirac operator (dslash)
 */

#include <iostream>
#include <cstdio>

#include "qdp.h"
#include "examples.h"

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

  //! Test out propagators
  multi1d<LatticeColorMatrix> u(Nd);
  for(int m=0; m < u.size(); ++m)
    gaussian(u[m]);

  LatticeFermion psi, chi;
  random(psi);
  chi = zero;
  dslash_2d_plus(chi, u, psi, 0);

//  dslash(chi, u, psi, +1, 0);

  NmlWriter nml("t_dslashm.nml");
  Write(nml,Nd);
  Write(nml,Nc);
  Write(nml,Ns);
  Write(nml,nrow);
  Write(nml,psi);
  Write(nml,chi);

  // Time to bolt
  QDP_finalize();

  return 0;
}
