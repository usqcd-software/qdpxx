// $Id: t_io.cc,v 1.1 2002-10-01 16:24:41 edwards Exp $

#include <iostream>
#include <cstdio>

#include "tests.h"

using namespace QDP;


int main(int argc, char **argv)
{
  // Setup the geometry
  const int foo[Nd] = {LX0,LX1};
  multi1d<int> nrow(Nd);
  nrow = foo;
  geom.Init(nrow);

  LatticeReal a;
  Double d = 17;
  random(a);

  TextWriter totext("cat");
  totext << a;
  totext.close();

  BinaryWriter tobinary("dog");
  tobinary << a;
  tobinary << d;
  tobinary.close();

  float x;
  cerr << "Read some data from file input\n";
  TextReader from("input");
  from >> x;
  from.close();

  cerr << "you entered :" << x << ":\n";
  
}
