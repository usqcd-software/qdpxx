// $Id: t_entry.cc,v 1.2 2003-06-20 02:38:54 edwards Exp $

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

  // WARNING: the variable "sa" will be node dependent, e.g. on each node
  // there will be different values. So, the write statement below of 
  // sa will only print that from the **primary** node. However, that
  // is the point of extract/insert - it allows the user to manipulate their
  // data into a QDP field on each node, from their own private data format.
  multi1d<Fermion> sa(Layout::sitesOnNode());
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
