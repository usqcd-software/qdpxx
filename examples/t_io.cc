// $Id: t_io.cc,v 1.11 2003-06-08 04:50:57 edwards Exp $

#include <iostream>
#include <cstdio>

#include <unistd.h>

#include "qdp.h"
#include "examples.h"

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

  BinaryWriter tobinary("dog");
  write(tobinary, a);
  write(tobinary, d);
  tobinary.close();

  LatticeReal aa;
  Double dd = 0.0;
  random(aa);

  BinaryReader frombinary("dog");
  read(frombinary, aa);
  read(frombinary, dd);
  frombinary.close();

  NmlWriter tonml("cat");
  Write(tonml,a);
  Write(tonml,aa);
  tonml.flush();
  tonml.close();

  float x = 42.1;
  cerr << "Write some data to file input\n";
  TextWriter totext("input");
  totext << x;
  totext.flush();
  totext.close();

  x = -1;
  cerr << "Read some data from file input\n";
  TextReader fromtext("input");
  fromtext >> x;
  fromtext.close();

  cerr << "you entered :" << x << ":" << endl;
  
  // Time to bolt
  QDP_finalize();

  exit(0);
}
