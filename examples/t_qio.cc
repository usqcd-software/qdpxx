// $Id: t_qio.cc,v 1.14 2004-03-11 17:09:06 edwards Exp $

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

  QDP_serialparallel_t serpar = QDPIO_SERIAL;

  for(int i=0; i < 2; ++i)
  {
    QDP_volfmt_t volfmt;

    if (i == 0)
    {
      volfmt = QDPIO_SINGLEFILE;

      QDPIO::cout << "\n\n\n\n***************SINGLEFILE tests*************\n" << endl;
    }
    else
    {
      volfmt = QDPIO_MULTIFILE; 

      QDPIO::cout << "\n\n***************MULTIFILE tests*************\n" << endl;
    }


    QDPIO::cout << "\n\n***************TEST WRITING*************\n" << endl;

    {
      XMLBufferWriter file_xml;

      push(file_xml,"file_fred");
      Double d = 17;
      write(file_xml,"d", d);
      push(file_xml,"file_sally");
      int rob = -5;
      write(file_xml,"rob", rob);
      pop(file_xml);
      pop(file_xml);

      XMLBufferWriter record_xml;
      push(record_xml,"record_fred");
      write(record_xml,"d", d);
      push(record_xml,"record_sally");
      write(record_xml,"rob", rob);
      pop(record_xml);
      pop(record_xml);

      QDPFileWriter to(file_xml,"t_qio.lime",volfmt,serpar,QDPIO_OPEN);

      LatticeComplex a;
      random(a);
      write(to,record_xml,a);

      QDPIO::cout << "First record test: innerProduct(a,shift(a,0))=" 
		  << Real(innerProductReal(a,shift(a,FORWARD,0))) << endl;

      LatticeColorMatrix b;
      random(b);
      write(to,record_xml,b);

      QDPIO::cout << "Second record test: innerProduct(b,shift(b,0))=" 
		  << Real(innerProductReal(b,shift(b,FORWARD,0))) << endl;

      close(to);
    }

    QDPIO::cout << "\n\n***************TEST READING*******************\n" << endl;

    {
      XMLReader file_xml;
      QDPFileReader from(file_xml,"t_qio.lime",serpar);

      QDPIO::cout << "Here is the contents of  file_xml" << endl;
      file_xml.print(cout);

      XMLReader record_xml;
      LatticeComplex a;
      read(from,record_xml,a);

      QDPIO::cout << "Here is the contents of first  record_xml" << endl;
      record_xml.print(cout);

      QDPIO::cout << "First record check: innerProduct(a,shift(a,0))=" 
		  << Real(innerProductReal(a,shift(a,FORWARD,0))) << endl;

      LatticeColorMatrix b;
      read(from,record_xml,b);

      QDPIO::cout << "Here is the contents of second  record_xml" << endl;
      record_xml.print(cout);

      QDPIO::cout << "Second record check: innerProduct(b,shift(b,0))=" 
		  << Real(innerProductReal(b,shift(b,FORWARD,0))) << endl;
    }
  }


  // Time to bolt
  QDP_finalize();

  exit(0);
}
