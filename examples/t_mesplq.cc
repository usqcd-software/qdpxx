// $Id: t_mesplq.cc,v 1.6 2002-11-13 02:33:53 edwards Exp $

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

  NmlWriter nml("t_mesplq.nml");

  push(nml,"lattis");
  Write(nml,Nd);
  Write(nml,Nc);
  Write(nml,nrow);
  pop(nml);

  //! Example of calling a plaquette routine
  /*! NOTE: the STL is *not* used to hold gauge fields */
  multi1d<LatticeGauge> u(Nd);
  Double w_plaq, s_plaq, t_plaq, link;

  cerr << "Start gaussian\n";
  for(int m=0; m < u.size(); ++m)
    gaussian(u[m]);

  MesPlq(u, w_plaq, s_plaq, t_plaq, link);
  cerr << "w_plaq = " << w_plaq << endl;
  cerr << "link = " << link << endl;

  // Write out the results
  push(nml,"observables");
  Write(nml,w_plaq);
  Write(nml,link);
  pop(nml);
}
