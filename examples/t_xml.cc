// $Id: t_xml.cc,v 1.2 2003-05-20 03:32:33 edwards Exp $

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

  {
    XMLDataWriter toxml("cat.xml");

    push(toxml,"fred");
    Write(toxml,d);

    push(toxml,"my_life");
    int rob = -5;
    Write(toxml,rob);
    pop(toxml);

    pop(toxml);
  }

  {
    XMLReader fromxml;
    fromxml.open("cat.xml");

    cout << "Here is the contents of  cat.xml" << endl;
    fromxml.print(cout);

    int rob;
    read(fromxml,"/fred/my_life/rob",rob);
    cout << "found rob = " << rob << endl;
  }

  // Time to bolt
  QDP_finalize();

  return 0;
}
