// $Id: t_exotic.cc,v 1.2 2003-09-10 02:05:33 edwards Exp $
/*! \file
 *  \brief Test various exotic qdp routines
 */

#include <iostream>
#include <cstdio>

#include "qdp.h"

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

  NmlWriter nml("t_exotic.nml");

  push(nml,"lattis");
  Write(nml,Nd);
  Write(nml,Nc);
  Write(nml,nrow);
  pop(nml);

  {
    // Try out colorContract
    LatticeColorMatrix a,b,c;
    gaussian(a);
    gaussian(b);
    gaussian(c);

    LatticeComplex lc1 = colorContract(a,b,c);

    push(nml,"color_contract_orig");
    Write(nml,lc1);
    pop(nml);
   
  }

  nml.flush();

  // Time to bolt
  QDP_finalize();

  exit(0);
}
