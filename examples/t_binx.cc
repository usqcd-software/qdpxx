// $Id: t_binx.cc,v 1.3 2004-06-05 15:43:31 edwards Exp $
//
// Write out binary with some XML markup
// in the binx format.
//

#include <iostream>
#include <cstdio>

#include "qdp.h"

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

  LatticeReal a;
  double d = 17;
  int bb = 4 ;
  random(a);

  QDPIO::cout << "Test the binx IO routines\n";

  BinxWriter tobinary("t_io.bin");
  //  write(tobinary, a);
  write(tobinary, d);
  write(tobinary, bb);
  float cc = 3.2 ;
  write(tobinary, cc);
  tobinary.close();
  // Time to bolt
  QDP_finalize();

  exit(0);
}
