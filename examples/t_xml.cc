// $Id: t_xml.cc,v 1.15 2003-08-27 18:56:34 edwards Exp $

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
    XMLBufferWriter toxml;

    push(toxml,"godzilla");
    int dog = -17;
    Write(toxml,dog);
    pop(toxml);

    cout << "buffer = XXX" << toxml.str() << "XXX" << endl;

//    std::istringstream list_stream(toxml.str()+"\n");
//    XMLReader fromxml(list_stream);

    XMLReader fromxml(toxml);
    int rob;
    read(fromxml,"/godzilla/dog",rob);
    cout << "found dog = " << rob << endl;
  }

  {
    XMLFileWriter toxml("t_xml.input1");

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
    fromxml.open("t_xml.input1");

    cout << "Here is the contents of  t_xml.input1" << endl;
    fromxml.print(cout);

    int rob;
    read(fromxml,"/fred/my_life/rob",rob);
    cout << "found rob = " << rob << endl;
  }

  {
    // Test reading some xml snippet and dumping it back out
    XMLReader fromxml;
    fromxml.open("t_xml.input1");

    XMLBufferWriter toxml_1;
    toxml_1 << fromxml;

    XMLBufferWriter toxml_2;
    push(toxml_2,"imbed_some_xml");
    write(toxml_2,"this_is_my_xml",fromxml);
    pop(toxml_2);

    XMLFileWriter toxml_3("t_xml.output1");
    toxml_3 << toxml_1;

    XMLFileWriter toxml_4("t_xml.output2");
    write(toxml_4,"imbed_some_more",toxml_2);
  }

  {
    // Test writing some more complex snippets
    XMLBufferWriter toxml;
    push(toxml,"complex_xml");

    write(toxml,"charStarThingy","whether tis nobler to suffer the slings and arrows");

    string stringThingy = "Sat Jun 16 00:35:57 2001";
    Write(toxml, stringThingy);

    Real a = 0.2;
    write(toxml,"realThingy",a);

    Complex b = cmplx(a,-1.2);
    write(toxml,"complexThingy",b);

    Seed seed = 1;
    write(toxml,"seedThingy",seed);

    multi1d<int> arrayInt(3);
    for(int i=0; i < arrayInt.size(); ++i)
      arrayInt[i] = i+37;
    write(toxml,"arrayInt",arrayInt);

    multi1d<Real> arrayReal(3);
    for(int i=0; i < arrayReal.size(); ++i)
      arrayReal[i] = i+107.5;
    write(toxml,"arrayReal",arrayReal);

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

    // Warning: this is going into a buffer writer and can be big
    LatticeColorMatrix u;
    random(u);
    write(toxml,"latticeColorMatrixThingy",u);
    toxml.flush();

    pop(toxml);

    // Play around - read this buffer back in
    XMLReader fromxml;
    fromxml.open(toxml);

    // Now dump it out to disk
    XMLFileWriter filexml("t_xml.input2");
    filexml << toxml;
  }

  {
    // Test reading some more complex snippets
    XMLReader fromxml;
    fromxml.open("t_xml.input2");

    Seed seed;
    read(fromxml,"/complex_xml/seedThingy",seed);
    cout << "seed = " << seed << endl;

    multi1d<int> arrayInt;
    read(fromxml,"/complex_xml/arrayInt",arrayInt);
    for(int i=0; i < arrayInt.size(); ++i)
      cout << "arrayInt[" << i << "] = " << arrayInt[i] << endl;

    multi1d<Real> arrayReal;
    read(fromxml,"/complex_xml/arrayReal",arrayReal);
    for(int i=0; i < arrayReal.size(); ++i)
      cout << "arrayReal[" << i << "] = " << arrayReal[i] << endl;

    multi1d<Complex> arrayComplex;
    for(int i=0; i < arrayComplex.size(); ++i)
      cout << "arrayComplex[" << i << "] = (" 
	   << Real(real(arrayComplex[i])) << ","            // The Real() shouldn't be necesary - 
	   << Real(imag(arrayComplex[i])) << ")" << endl;   // it converts a QDPExpr to a QDPType
  }

  {
    // Try out an array context
    try
    { 
      XMLFileWriter  xml_file_out("t_xml.output3");
      push(xml_file_out,"root_for_output3");
      XMLArrayWriter  xml_out(xml_file_out, 3);
      push(xml_out,"this_is_an_array");

      for(int i=0; i < xml_out.size(); ++i)
      {
	int x = -42;
#if 1
//	push(xml_out,"some_ignored_name");
	push(xml_out);    // note, can use name or unnamed version here - name ignored
	write(xml_out,"x",x);
	pop(xml_out);
#else
	write(xml_out,"some_ignored_name",x);  // will instead use "elem"
#endif
      }
    
      pop(xml_out);
      pop(xml_file_out);
    }
    catch( const string& error) { 
      cout << "Error: " << error << endl;
    }
  }

  // Time to bolt
  QDP_finalize();

  exit(0);
}
