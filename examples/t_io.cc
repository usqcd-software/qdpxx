// $Id: t_io.cc,v 1.5 2002-11-28 02:56:50 edwards Exp $

#include <iostream>
#include <cstdio>

#include "tests.h"

using namespace QDP;


int main(int argc, char **argv)
{
  // Setup the geometry
  const int foo[] = {2,2,2,2};
  multi1d<int> nrow(Nd);
  nrow = foo;  // Use only Nd elements
  geom.init(nrow);

  LatticeReal a;
  Double d = 17;
  random(a);

  NmlWriter tonml("cat");
  Write(tonml,a);
  tonml.close();

  BinaryWriter tobinary("dog");
  write(tobinary, a);
  write(tobinary, d);
  tobinary.close();

  float x;
  cerr << "Read some data from file input\n";
  TextReader from("input");
  from >> x;
  from.close();

  cerr << "you entered :" << x << ":\n";
  
}
