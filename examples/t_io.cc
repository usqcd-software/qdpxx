// $Id: t_io.cc,v 1.17 2003-10-09 19:59:39 edwards Exp $

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

  BinaryWriter tobinary("t_io.bin");
  write(tobinary, a);
  write(tobinary, d);
  tobinary.close();

  LatticeReal aa;
  Double dd = 0.0;
  random(aa);

  BinaryReader frombinary("t_io.bin");
  read(frombinary, aa);
  read(frombinary, dd);
  frombinary.close();

  NmlWriter tonml("t_io.nml");
  Write(tonml,a);
  Write(tonml,aa);
  tonml.flush();
  tonml.close();

  Real x = 42.1;
  QDPIO::cout << "Write some data to file t_io.txt\n";
  TextWriter totext("t_io.txt");
  totext << x;
  totext.flush();
  totext.close();

  x = -1;
  QDPIO::cout << "Read some data from file t_io.txt\n";
  TextReader fromtext("t_io.txt");
  fromtext >> x;
  fromtext.close();

  QDPIO::cout << "you entered :" << x << ":" << endl;
  
  // Time to bolt
  QDP_finalize();

  exit(0);
}
