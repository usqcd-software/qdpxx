// $Id: t_xml.cc,v 1.6 2003-06-09 19:33:52 edwards Exp $

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
    XMLFileWriter toxml("cat.xml");

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

  {
    // Test reading some xml snippet and dumping it back out
    XMLReader fromxml;
    fromxml.open("cat.xml");

    XMLBufferWriter toxml_1;
    toxml_1 << fromxml;

    XMLBufferWriter toxml_2;
    push(toxml_2,"imbed_some_xml");
    write(toxml_2,"this_is_my_xml",fromxml);
    pop(toxml_2);

    XMLFileWriter toxml_3("dog1.xml");
    toxml_3 << toxml_1;

    XMLFileWriter toxml_4("dog2.xml");
    write(toxml_4,"imbed_some_more",toxml_2);
  }

  {
    // Test writing some more complex snippets
    XMLFileWriter toxml("dog3.xml");
    push(toxml,"complex_xml");

    Real a = 0.2;
    write(toxml,"realThingy",a);

    Complex b = cmplx(a,-1.2);
    write(toxml,"complexThingy",b);

    multi1d<int> arrayInt(3);
    for(int i=0; i < arrayInt.size(); ++i)
      arrayInt[i] = i+37;
    write(toxml,"arrayInt",arrayInt);

    multi1d<Complex> arrayComplex(2);
    for(int i=0; i < arrayComplex.size(); ++i)
      arrayComplex[i] = cmplx(Real(1.0),Real(i+42));
    write(toxml,"arrayComplex",arrayComplex);

    ColorVector c;
    random(c);
    write(toxml,"colorVectorThingy",c);

    SpinMatrix d;
    random(d);
    write(toxml,"spinMatrixThingy",d);

    LatticeColorMatrix u;
    random(u);
    write(toxml,"latticeColorMatrixThingy",u);
    toxml.flush();

    pop(toxml);
  }

  // Time to bolt
  QDP_finalize();

  exit(0);
}
