// $Id: t_mesplq2.cc,v 1.1 2003-08-20 09:39:20 bjoo Exp $

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
  const int foo[] = {8,8,8,8};
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

  if( argc == 1 ) { 
    cerr << "Start gaussian\n";
    for(int m=0; m < u.size(); ++m)
      gaussian(u[m]);

  // Reunitarize the gauge field
    cerr << "Start reunit\n";
    for(int m=0; m < u.size(); ++m)
      reunit(u[m]);
  }
  else { 
    if ( argc == 2 ) {
      cerr << "Trying to read NERSC Archive from " << argv[1] << endl;
      readArchiv(u, argv[1]);
    }
    else {
      cerr << "Usage: " << argv[0] << " <optional 8x8x8x8 gauge filename (NERSC format) " << endl;
    }
  }
 
 // Try out the plaquette routine
  cerr << "Start mesplq\n";
  MesPlq(u, w_plaq, s_plaq, t_plaq, link);
  cerr << "w_plaq = " << w_plaq << endl;
  cerr << "link = " << link << endl;

  // Write out the results
  push(nml,"observables");
  Write(nml,w_plaq);
  Write(nml,link);
  pop(nml);

  nml.flush();

  // Time to bolt
  QDP_finalize();

  exit(0);
}
