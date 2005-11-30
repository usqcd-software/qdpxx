// $Id: t_partfile.cc,v 1.1 2005-11-30 00:56:10 bjoo Exp $

#include <iostream>
#include <cstdio>

#include "qdp.h"
#include "unittest.h"

// The Unit Tests Themselves
#include "testOpenPartFile.h"

using namespace QDP;

int main(int argc, char **argv)
{
  // Initialize QDP++ with argc, and argv. Set Lattice Dimensions
  const int latdims[] = {4,4,8,8};

  // Initialize UnitTest jig
  TestRunner  testjig(&argc, &argv, latdims);
  
  // Add a test -- to open a partfile
  testjig.addTest(new TestOpenPartFile());

  // Run all tests
  testjig.run();

  // Testjig is destroyed
  testjig.summary();

}

