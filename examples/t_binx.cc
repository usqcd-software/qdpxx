// $Id: t_binx.cc,v 1.1 2004-03-25 14:01:02 mcneile Exp $
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
  Double d = 17;
  random(a);

  QDPIO::cout << "Test the binx IO routines\n";

  BinxWriter tobinary("t_io.bin");
  //  write(tobinary, a);
  write(tobinary, d);
  tobinary.close();
  // Time to bolt
  QDP_finalize();

  exit(0);
}
