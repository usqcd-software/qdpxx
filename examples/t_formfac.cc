// $Id: t_formfac.cc,v 1.5 2002-11-13 02:33:53 edwards Exp $

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

  NmlWriter nml("formfac.nml");

  //! Test out propagators
  multi1d<LatticeGauge> u(Nd);
  for(int m=0; m < u.size(); ++m)
    gaussian(u[m]);

  push(nml,"lattice");
  Write(nml,Nd);
  Write(nml,Nc);
  Write(nml,Ns);
  Write(nml,nrow);
  pop(nml);

//  Write(nml,u);

  LatticePropagator quark_prop_1, quark_prop_2;
  gaussian(quark_prop_1);
  gaussian(quark_prop_2);

  int j_decay = Nd-1;
  int length = geom.LattSize()[j_decay];
  multi1d<int> t_source(Nd);
  t_source = 0;

  int t_sink = length-1;

  FormFac(u, quark_prop_1, quark_prop_2, t_source, t_sink, nml);
}
