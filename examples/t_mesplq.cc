// $Id: t_mesplq.cc,v 1.22 2004-11-22 19:31:31 edwards Exp $

#include <iostream>
#include <cstdio>

#include "qdp.h"
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

  XMLFileWriter xml("t_mesplq.xml");
  push(xml,"t_mesplq");

  push(xml,"lattis");
  write(xml,"Nd",Nd);
  write(xml,"Nc",Nc);
  write(xml,"nrow",nrow);
  pop(xml);

  //! Example of calling a plaquette routine
  /*! NOTE: the STL is *not* used to hold gauge fields */
  multi1d<LatticeColorMatrix> u(Nd);
  Double w_plaq, s_plaq, t_plaq, link;

  QDPIO::cout << "Start gaussian\n";
  for(int m=0; m < u.size(); ++m)
    gaussian(u[m]);

  // Reunitarize the gauge field
  QDPIO::cout << "Start reunit\n";
  for(int m=0; m < u.size(); ++m)
    reunit(u[m]);

  // Try out the plaquette routine
  QDPIO::cout << "Start mesplq\n";
  MesPlq(u, w_plaq, s_plaq, t_plaq, link);
  QDPIO::cout << "w_plaq = " << w_plaq << endl;
  QDPIO::cout << "link = " << link << endl;

  // Write out the results
  push(xml,"observables");
  write(xml,"w_plaq",w_plaq);
  write(xml,"link",link);
  pop(xml);

  pop(xml);
  xml.close();

  // Time to bolt
  QDP_finalize();

  exit(0);
}
