// $Id: t_aff.cc,v 1.1.2.1 2008-03-15 14:28:54 edwards Exp $

#include <iostream>
#include <cstdio>

#include "qdp.h"
#include "qdp_aff_imp.h"

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

  QDPIO::cout << "line= " << __LINE__ << endl;
  try
  {
    AFFFileWriterImp toaff("t_aff.input1");

    int rob = -5;
    toaff.openTag("rob");
    toaff.openTag("bar");
    toaff.write("rob", rob);
    string fred="the life";
    toaff.write("fred", fred);
    toaff.closeTag();
    toaff.closeTag();

    toaff.close();
  }
  catch(const string& e)
  {
    QDPIO::cerr << "Error: basic aff write tests: error=" << e << endl;
    QDP_abort(1);
  }
  catch(std::bad_cast) 
  {
    QDPIO::cerr << "Error: caught cast error here: line= " << __LINE__ << endl;
    QDP_abort(1);
  }

  QDPIO::cout << "line= " << __LINE__ << endl;
  try
  {
    LAFFReaderImp orig("t_aff.input1");
    LAFFReaderImp fromaff(orig, "rob");

    fromaff.exist("/rob/bar/rob");
    fromaff.exist("/rob/fred/rob");
    fromaff.exist("rob");
    fromaff.exist("bar");

    int rob;
    fromaff.read("/rob/bar/rob", rob);
    QDPIO::cout << "rob= " << rob << endl;

    fromaff.read("bar/rob", rob);
    QDPIO::cout << "bar= " << rob << endl;

    LAFFReaderImp faff(fromaff, "bar");

    faff.read("rob", rob);
    QDPIO::cout << "bar= " << rob << endl;

    string fred;
    faff.read("fred", fred);
    QDPIO::cout << "fred= XX" << fred << "XX" << endl;

    fromaff.close();
  }
  catch(const string& e)
  {
    QDPIO::cerr << "Error: basic aff write tests: error=" << e << endl;
    QDP_abort(1);
  }
  catch(std::bad_cast) 
  {
    QDPIO::cerr << "Error: caught cast error here: line= " << __LINE__ << endl;
    QDP_abort(1);
  }

  // Time to bolt
  QDP_finalize();

  QDPIO::cout << "exiting: line= " << __LINE__ << endl;

  exit(0);
}
