// $Id: t_io.cc,v 1.4 2002-11-13 02:33:53 edwards Exp $

#include <iostream>
#include <cstdio>

#include "tests.h"

using namespace QDP;


int main(int argc, char **argv)
{
  // Setup the geometry
  const int foo[] = {LX0,LX1,LX2,LX3};
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
