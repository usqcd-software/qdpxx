// $Id: t_qio.cc,v 1.17 2004-03-29 21:29:31 edwards Exp $

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

  XMLFileWriter xml_out("t_qio.xml");
  push(xml_out, "t_qio");

  QDP_serialparallel_t serpar = QDPIO_SERIAL;

  for(int i=0; i < 2; ++i)
  {
    QDP_volfmt_t volfmt;

    if (i == 0)
    {
      volfmt = QDPIO_SINGLEFILE;

      QDPIO::cout << "\n\n\n\n***************SINGLEFILE tests*************\n" << endl;

      push(xml_out, "Singlefile");
    }
    else
    {
      volfmt = QDPIO_MULTIFILE; 

      QDPIO::cout << "\n\n***************MULTIFILE tests*************\n" << endl;

      push(xml_out, "Multifile");
    }


    QDPIO::cout << "\n\n***************TEST WRITING*************\n" << endl;

    {
      push(xml_out, "Writing");

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
      write(xml_out, "file_xml", file_xml);

#if 1
      if (volfmt == QDPIO_SINGLEFILE)   // there seems to be a bug in multifile here
      {
        multi1d<Complex> ff(5);
        Double fsum = 0;
        for(int i=0; i < ff.size(); ++i)
        {
    	  random(ff[i]);
	  fsum += norm2(ff[i]);
        }
        write(to,record_xml,ff);

        QDPIO::cout << "First record test: fsum=" << fsum << endl;
        push(xml_out, "Record1");
        write(xml_out, "record_xml", record_xml);
        write(xml_out, "fsum", fsum);
        write(xml_out, "ff", ff);
        pop(xml_out);
      }
#endif

      LatticeComplex a;
      random(a);
      write(to,record_xml,a);

      Real atest = Real(innerProductReal(a,shift(a,FORWARD,0)));
      QDPIO::cout << "Second record test: innerProduct(a,shift(a,0))=" 
		  << atest << endl;
      push(xml_out, "Record2");
      write(xml_out, "record_xml", record_xml);
      write(xml_out, "atest", atest);
      pop(xml_out);

#if 1
      LatticeColorMatrix b;
      random(b);
      write(to,record_xml,b);

      Real btest = Real(innerProductReal(b,shift(b,FORWARD,0)));
      QDPIO::cout << "Third record test: innerProduct(b,shift(b,0))=" 
		  << btest << endl;
      push(xml_out, "Record3");
      write(xml_out, "record_xml", record_xml);
      write(xml_out, "btest", btest);
      pop(xml_out);
#endif

      close(to);

      pop(xml_out);   // writing
    }

    QDPIO::cout << "\n\n***************TEST READING*******************\n" << endl;

    {
      push(xml_out, "Reading");

      XMLReader file_xml;
      QDPFileReader from(file_xml,"t_qio.lime",serpar);

      QDPIO::cout << "Here is the contents of  file_xml" << endl;
      file_xml.print(cout);
      write(xml_out, "file_xml", file_xml);

      XMLReader record_xml;

#if 1
      if (volfmt == QDPIO_SINGLEFILE)
      {
        multi1d<Complex> ff(5);   // needs to be free
        read(from,record_xml,ff);

        QDPIO::cout << "Here is the contents of first  record_xml" << endl;
        record_xml.print(cout);

        Double fsum = 0;
        for(int i=0; i < ff.size(); ++i)
  	  fsum += norm2(ff[i]);
        QDPIO::cout << "First record test: fsum=" << fsum << endl;
        push(xml_out, "Record1");
        write(xml_out, "record_xml", record_xml);
        write(xml_out, "fsum", fsum);
        write(xml_out, "ff", ff);
        pop(xml_out);
      }
#endif

      LatticeComplex a;
      read(from,record_xml,a);

      QDPIO::cout << "Here is the contents of second  record_xml" << endl;
      record_xml.print(cout);

      Real atest = Real(innerProductReal(a,shift(a,FORWARD,0)));
      QDPIO::cout << "Second record check: innerProduct(a,shift(a,0))=" 
		  << atest << endl;
      push(xml_out, "Record2");
      write(xml_out, "record_xml", record_xml);
      write(xml_out, "atest", atest);
      pop(xml_out);

#if 1
      LatticeColorMatrix b;
      read(from,record_xml,b);

      QDPIO::cout << "Here is the contents of third  record_xml" << endl;
      record_xml.print(cout);

      Real btest = Real(innerProductReal(b,shift(b,FORWARD,0)));
      QDPIO::cout << "Third record check: innerProduct(b,shift(b,0))=" 
		  << btest << endl;
      push(xml_out, "Record3");
      write(xml_out, "record_xml", record_xml);
      write(xml_out, "btest", btest);
      pop(xml_out);

#endif

      close(from);   // reading

      pop(xml_out);
    }

    pop(xml_out);  // single or multifile
  }


  pop(xml_out);   // t_qio

  // Time to bolt
  QDP_finalize();

  exit(0);
}
