// $Id: t_nersc.cc,v 1.4 2003-10-15 17:20:08 edwards Exp $

#include <iostream>
#include <cstdio>

#include "qdp.h"
#include "qdp_iogauge.h"
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

  XMLFileWriter xml("t_nersc.xml");
  push(xml, "t_nersc");

  push(xml,"lattis");
  Write(xml,Nd);
  Write(xml,Nc);
  Write(xml,nrow);
  pop(xml);

  {
    multi1d<LatticeColorMatrix> u(Nd);
    Double w_plaq, s_plaq, t_plaq, link;

    QDPIO::cout << "Start gaussian\n";
    for(int m=0; m < u.size(); ++m)
      gaussian(u[m]);

    // Reunitarize the gauge field
    QDPIO::cout << "Start reunit\n";
    for(int m=0; m < u.size(); ++m)
      reunit(u[m]);

    QDPIO::cout << "Start mesplq\n";
    MesPlq(u, w_plaq, s_plaq, t_plaq, link);
    QDPIO::cout << "w_plaq = " << w_plaq << endl;
    QDPIO::cout << "link = " << link << endl;

    // Write out the results
    push(xml,"Initial_observables");
    Write(xml,w_plaq);
    Write(xml,link);
    pop(xml);

    // Now write the gauge field in NERSC format
    QDPIO::cout << "Trying to write NERSC Archive  t_nersc.cfg" << endl;
    writeArchiv(u, "t_nersc.cfg");
  }

  {
    multi1d<LatticeColorMatrix> u(Nd);
    Double w_plaq, s_plaq, t_plaq, link;

    QDPIO::cout << "Trying to read back config" << endl;

    XMLReader gauge_xml;
    readArchiv(gauge_xml, u, "t_nersc.cfg");
 
    // Try out the plaquette routine
    QDPIO::cout << "Start mesplq\n";
    MesPlq(u, w_plaq, s_plaq, t_plaq, link);
    QDPIO::cout << "w_plaq = " << w_plaq << endl;
    QDPIO::cout << "link = " << link << endl;

    // Write out the results
    push(xml,"Final_observables");
    Write(xml,w_plaq);
    Write(xml,link);
    pop(xml);
  }

  pop(xml);

  // Time to bolt
  QDP_finalize();

  exit(0);
}
