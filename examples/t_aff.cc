// $Id: t_aff.cc,v 1.1.2.2 2008-03-16 02:40:03 edwards Exp $

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
    AFFFileWriter aff("t_aff.input1");

    int a = 0;
    multi1d<double> bar(5);
    bar = 2.3;

    push(aff, "root");
    write(aff, "a", a);
    write(aff, "bar", bar);
    pop(aff);
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
    AFFReader aff("t_aff.input1");

    int a;

    TreeReader tree(aff, "/root");
    read(tree, "a", a);

    multi1d<double> bar;
    read(tree, "bar", bar);
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
