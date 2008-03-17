// $Id: t_aff.cc,v 1.1.2.3 2008-03-17 03:55:36 edwards Exp $

#include <iostream>
#include <cstdio>

#include "qdp.h"
#include "qdp_aff_imp.h"

struct FooTest_t
{
  int fred;
};

void read(TreeReader& tree, const std::string& path, FooTest_t& param)
{ 
  QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
  TreeReader top(tree, path);
  QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
  read(top, "fred", param.fred);
  QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
}

void write(TreeWriter& tree, const std::string& path, const FooTest_t& param)
{
  push(tree, path);
  write(tree, "fred", param.fred);
  pop(tree);
}


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
    multi1d<Real> bar(5);
    bar = 2.3;

    push(aff, "root");
    write(aff, "a", a);
    write(aff, "bar", bar);

  QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    TreeArrayWriter tree_array(aff, 3);
  QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    push(tree_array, "array");
  QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    for(int i=0; i < tree_array.size(); ++i)
    {
      pushElem(tree_array);
      write(tree_array, "fred", a);
      popElem(tree_array);
    }
  QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    pop(tree_array);
  QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    pop(aff);
  QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
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

    multi1d<Real> bar;
    read(tree, "bar", bar);

    multi1d<FooTest_t> test;
    read(tree, "array", test);
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
