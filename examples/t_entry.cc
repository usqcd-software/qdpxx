// $Id: t_entry.cc,v 1.1 2003-06-20 02:19:34 edwards Exp $

/*! \file
 *  \brief Test entry/exit routines
 *
 */

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

  LatticeFermion la;
  multi1d<Fermion> sa(Layout::sitesOnNode());
  random(la);

  // Suspend communications
  // NOTE: this is not necessary for an extraction
  QDP_suspend();

  // Now can do stuff outside of QDP, e.g. call communications directly
  // but QDP is still ``turned on''
  int x=1;
  x++;

  // Resume communications
  // NOTE: this is not necessary for an extraction
  QDP_resume();

  // Must make sure all communications are allowed
  NmlWriter tonml("t_entry.nml");

  // Pull out the data
//  Subset& even = rb[0];

  sa = zero;
  QDP_extract(sa, la, even);

  la = zero;
  QDP_insert(la, sa, even);

  push(tonml, "Site_field");
  Write(tonml,sa);
  pop(tonml);

  push(tonml, "Lattice_field");
  Write(tonml,la);
  pop(tonml);

  tonml.close();

  // Time to bolt
  QDP_finalize();

  exit(0);
}
