// $Id: t_mesplq.cc,v 1.18 2004-02-03 15:11:33 bjoo Exp $

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


  XMLBufferWriter nml;
  push(nml, "mesplqTest");

  push(nml,"lattis");
  try { 
  write(nml, "Nd" , Nd);
  write(nml, "Nc", Nc);
  write(nml, "nrow", nrow);
  pop(nml);
  }
  catch ( const std::string& e ) { 
	QDPIO::cout << "exception raised : " << e << endl;
  }
    
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
  try { 
  push(nml,"observables");
  write(nml, "w_plaq", w_plaq);
  write(nml, "link", link);
  pop(nml);
 
  pop(nml); // Pop root tag
  } 
  catch (const std::string& e) { 
    QDPIO::cout << "Exception Raised : " << e << endl;
  }
 
  QDPIO::cout << "XML Output" << endl ;
  std::string str;
  try { 
    QDPIO::cout <<  nml.printRoot() << endl;
  }
  catch (const std::string& e) {
     QDPIO::cerr << "Exception raised : " << e << endl;
  }
  // Time to bolt
  QDP_finalize();

  exit(0);
}
