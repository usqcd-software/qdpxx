// $Id: t_formfac.cc,v 1.15 2004-02-11 10:33:09 bjoo Exp $
/*! \file
 *  \brief Test the form-factor routine
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
  const int foo[] = {2,2,2,4};
  multi1d<int> nrow(Nd);
  nrow = foo;  // Use only Nd elements
  Layout::setLattSize(nrow);
  Layout::create();

  NmlWriter nml("t_formfac.nml");

  //! Test out propagators
  multi1d<LatticeColorMatrix> u(Nd);
  for(int m=0; m < u.size(); ++m)
    gaussian(u[m]);

  push(nml,"lattice");
  write(nml,"Nd", Nd);
  write(nml,"Nc", Nc);
  write(nml,"Ns", Ns);
  write(nml,"nrow", nrow);
  pop(nml);

//  write(nml,"u", u);

  LatticePropagator quark_prop_1, quark_prop_2;
  gaussian(quark_prop_1);
  gaussian(quark_prop_2);

  int j_decay = Nd-1;
  int length = Layout::lattSize()[j_decay];
  multi1d<int> t_source(Nd);
  t_source = 0;

  int t_sink = length-1;

  FormFac(u, quark_prop_1, quark_prop_2, t_source, t_sink, j_decay, nml);

  // Time to bolt
  QDP_finalize();

  exit(0);
}
