// $Id: t_nersc.cc,v 1.2 2003-08-26 21:38:38 edwards Exp $

#include <iostream>
#include <cstdio>

#include "qdp.h"
#include "qdp_util.h"
#include "examples.h"

using namespace QDP;

int main(int argc, char *argv[])
{
  // Put the machine into a known state
  QDP_initialize(&argc, &argv);

  // Setup the layout
  const int foo[] = {4,4,4,4};
  multi1d<int> nrow(Nd);
  nrow = foo;  // Use only Nd elements
  Layout::setLattSize(nrow);
  Layout::create();

  NmlWriter nml("t_mesplq.nml");

  push(nml,"lattis");
  Write(nml,Nd);
  Write(nml,Nc);
  Write(nml,nrow);
  pop(nml);

  //! Example of calling a plaquette routine
  /*! NOTE: the STL is *not* used to hold gauge fields */
  multi1d<LatticeColorMatrix> u(Nd);
  Double w_plaq, s_plaq, t_plaq, link;

  cout << "Trying to read NERSC Archive  t_nersc.cfg\n"
       << "  make sure it is in your current directory" << endl;
  XMLReader xml;
  readArchiv(xml, u, "t_nersc.cfg");
 
  // Try out the plaquette routine
  cout << "Start mesplq\n";
  MesPlq(u, w_plaq, s_plaq, t_plaq, link);
  cout << "w_plaq = " << w_plaq << endl;
  cout << "link = " << link << endl;

  // Write out the results
  push(nml,"observables");
  Write(nml,w_plaq);
  Write(nml,link);
  pop(nml);

  // Time to bolt
  QDP_finalize();

  exit(0);
}
