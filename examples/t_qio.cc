// $Id: t_qio.cc,v 1.8 2004-01-30 22:15:13 edwards Exp $

#include <iostream>
#include <cstdio>

#include "qdp.h"

using namespace QDP;


int main(int argc, char **argv)
{
  // Put the machine into a known state
  QDP_initialize(&argc, &argv);

  // Setup the layout
  const int foo[] = {4,4,4,8};
  multi1d<int> nrow(Nd);
  nrow = foo;  // Use only Nd elements
  Layout::setLattSize(nrow);
  Layout::create();

#if 1
  // SciDAC output format - move up into switch statement or a subroutine
  {
    LatticePropagator quark_propagator = zero;
    string source_filename = "t_qio.test";

    XMLBufferWriter file_xml;
    push(file_xml, "MyFileXML");
    write(file_xml, "foobar", 1);
    pop(file_xml);

    QDPSerialFileWriter to(file_xml,source_filename);
    QDPIO::cout << "QDPSerialFile Writer opened" << endl << flush;

    XMLBufferWriter record_xml;
    push(record_xml, "MyRecordXML");
    write(record_xml, "muchies", 17);
    pop(record_xml);

    to.write(record_xml,quark_propagator);  // can keep repeating writes for more records
    to.close();
  }
#endif

#if 1
  {
    XMLBufferWriter file_xml;

    push(file_xml,"file_fred");
    Double d = 17;
    Write(file_xml,d);
    push(file_xml,"file_sally");
    int rob = -5;
    Write(file_xml,rob);
    pop(file_xml);
    pop(file_xml);

    XMLBufferWriter record_xml;
    push(record_xml,"record_fred");
    Write(record_xml,d);
    push(record_xml,"record_sally");
    Write(record_xml,rob);
    pop(record_xml);
    pop(record_xml);

    QDPSerialFileWriter to(file_xml,"t_qio.dime");

    LatticeComplex a;
    random(a);
    to.write(record_xml,a);

    LatticeColorMatrix b;
    random(b);
    to.write(record_xml,b);

    to.close();
  }

  {
    XMLReader file_xml;
    QDPSerialFileReader from(file_xml,"t_qio.dime");

    QDPIO::cout << "Here is the contents of  file_xml" << endl;
    file_xml.print(cout);

    XMLReader record_xml;
    LatticeComplex a;
    from.read(record_xml,a);

    QDPIO::cout << "Here is the contents of first  record_xml" << endl;
    record_xml.print(cout);

    LatticeColorMatrix b;
    from.read(record_xml,b);

    QDPIO::cout << "Here is the contents of second  record_xml" << endl;
    record_xml.print(cout);
  }
#endif

  // Time to bolt
  QDP_finalize();

  exit(0);
}
