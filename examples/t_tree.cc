// $Id: t_tree.cc,v 1.1.2.1 2008-03-17 03:55:36 edwards Exp $

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

  QDPIO::cout << "line= " << __LINE__ << endl;
  try
  {
    XMLBufferWriter toxml;
  QDPIO::cout << "line= " << __LINE__ << endl;

    push(toxml,"godzilla");
  QDPIO::cout << "line= " << __LINE__ << endl;
    int dog = -17;
  QDPIO::cout << "line= " << __LINE__ << endl;
    write(toxml,"dog", dog);
  QDPIO::cout << "line= " << __LINE__ << endl;
    pop(toxml);
  QDPIO::cout << "line= " << __LINE__ << endl;

//    std::istringstream list_stream(toxml.str()+"\n");
//    TreeReader fromxml(list_stream);

  QDPIO::cout << "line= " << __LINE__ << endl;
    TreeReader fromxml(toxml);
    int rob;
  QDPIO::cout << "line= " << __LINE__ << endl;
    read(fromxml,"/godzilla/dog",rob);
  QDPIO::cout << "line= " << __LINE__ << endl;
    QDPIO::cout << "found dog = " << rob << endl;
  QDPIO::cout << "line= " << __LINE__ << endl;
  }
  catch(const string& e)
  {
    QDPIO::cerr << "Error: TreeBufferWriter into a TreeReader test: error=" << e << endl;
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
    TreeFileWriter toxml("t_xml.input1");

    push(toxml,"fred");
    write(toxml,"d", d);

    push(toxml,"my_life");
    int rob = -5;
    write(toxml,"rob", rob);
    pop(toxml);

    pop(toxml);
    toxml.close();
  }
  catch(const string& e)
  {
    QDPIO::cerr << "Error: basic xml write tests: error=" << e << endl;
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
    TreeReader fromxml;
    fromxml.open("t_xml.input1");

    QDPIO::cout << "Here is the contents of  t_xml.input1" << endl;
    fromxml.print(cout);

    int rob;
    read(fromxml,"/fred/my_life/rob",rob);
    QDPIO::cout << "found rob = " << rob << endl;
  }
  catch(const string& e)
  {
    QDP_error_exit("Error reading some xml snippets: %s",e.c_str());
  }
  catch(std::bad_cast) 
  {
    QDPIO::cerr << "Error: caught cast error here: line= " << __LINE__ << endl;
    QDP_abort(1);
  }

  QDPIO::cout << "line= " << __LINE__ << endl;

  try
  {
    // Test reading some xml snippet and dumping it back out
    TreeReader fromxml;
    fromxml.open("t_xml.input1");

    TreeBufferWriter toxml_1;
    toxml_1 << fromxml;

//    TreeBufferWriter toxml_2;
//    push(toxml_2,"imbed_some_xml");
//    write(toxml_2,"this_is_my_xml",fromxml);
//    pop(toxml_2);

    TreeFileWriter toxml_3("t_xml.output1");
    toxml_3 << toxml_1;

    TreeFileWriter toxml_4("t_xml.output2");
//    write(toxml_4,"imbed_some_more",toxml_2);
  }
  catch(const string& e)
  {
    QDPIO::cerr << "Error: writing some xml snippets: error=" << e << endl;
    QDP_abort(1);
  }
  catch(std::bad_cast) 
  {
    QDPIO::cerr << "Error: caught cast error here: line= " << __LINE__ << endl;
    QDP_abort(1);
  }

  QDPIO::cout << "line= " << __LINE__ << endl;

  // Test writing some more complex snippets
  try
  {
    TreeBufferWriter toxml;
    push(toxml,"complex_xml");

    write(toxml,"charStarThingy","whether tis nobler to suffer the slings and arrows");

    string stringThingy = "Sat Jun 16 00:35:57 2001";
    write(toxml, "stringThingy", stringThingy);

    Double a = 0.2;
    write(toxml,"realThingy",a);

    Complex b = cmplx(a,-1.2);
    write(toxml,"complexThingy",b);

    QDP::Seed seed = 1;
    write(toxml,"seedThingy",seed);

    multi1d<int> arrayInt(3);
    for(int i=0; i < arrayInt.size(); ++i)
      arrayInt[i] = i+37;
    write(toxml,"arrayInt",arrayInt);

    multi1d<Double> arrayDouble(3);
    for(int i=0; i < arrayDouble.size(); ++i)
      arrayDouble[i] = i+107.5;
    write(toxml,"arrayDouble",arrayDouble);

    multi1d<Complex> arrayComplex(2);
    for(int i=0; i < arrayComplex.size(); ++i)
      arrayComplex[i] = cmplx(Double(1.0),Double(i+42));
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
    TreeReader fromxml;
    fromxml.open(toxml);

    // Now dump it out to disk
    TreeFileWriter filexml("t_xml.input2");
    filexml << toxml; 

  }
  catch(const string& e)
  {
    QDPIO::cerr << "Error: writing some complex snippets: error=" << e << endl;
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
    QDPIO::cout << "line= " << __LINE__ << endl;
    // Test reading some more complex snippets
    TreeReader fromxml_orig;
    fromxml_orig.open("t_xml.input2");

    QDPIO::cout << "line= " << __LINE__ << endl;
    TreeReader fromxml(fromxml_orig, "/complex_xml");

    QDPIO::cout << "line= " << __LINE__ << endl;
    TreeReader fromxml_tmp(fromxml, "seedThingy");
    ostringstream os;
    fromxml_tmp.printCurrentContext(os);
    QDPIO::cout << "Current context = XX" << os.str() << "XX" << endl;

    QDPIO::cout << "line= " << __LINE__ << endl;
    QDP::Seed seed;
    read(fromxml,"seedThingy",seed);
    QDPIO::cout << "seed = " << seed <<  "  node=" << Layout::nodeNumber() << endl;

    QDPIO::cout << "line= " << __LINE__ << endl;
    multi1d<int> arrayInt;
    read(fromxml,"arrayInt",arrayInt);
    for(int i=0; i < arrayInt.size(); ++i)
      QDPIO::cout << "arrayInt[" << i << "] = " << arrayInt[i]  << "  node=" << Layout::nodeNumber() << endl;

    QDPIO::cout << "line= " << __LINE__ << endl;
    multi1d<Double> arrayDouble;
    read(fromxml,"arrayDouble",arrayDouble);
    for(int i=0; i < arrayDouble.size(); ++i)
      QDPIO::cout << "arrayDouble[" << i << "] = " << arrayDouble[i] << "  node=" << Layout::nodeNumber() << endl;

    QDPIO::cout << "line= " << __LINE__ << endl;
    multi1d<Complex> arrayComplex;
    read(fromxml,"arrayComplex",arrayComplex);
    for(int i=0; i < arrayComplex.size(); ++i)
      QDPIO::cout << "arrayComplex[" << i << "] = ("
		  << real(arrayComplex[i]) << "," 
		  << imag(arrayComplex[i]) << ")" << endl;

    QDPIO::cout << "line= " << __LINE__ << endl;
    QDP_info("done with array snippet tests");
  }
  catch(const string& e)
  {
    QDPIO::cerr << "Error: reading array snippets: error=" << e << endl;
    QDP_abort(1);
  }
  catch(std::bad_cast) 
  {
    QDPIO::cerr << "Error: caught cast error here: line= " << __LINE__ << endl;
    QDP_abort(1);
  }

  QDPIO::cout << "line= " << __LINE__ << endl;

  // Try out a simple array context
  try
  { 
  QDPIO::cout << "line= " << __LINE__ << endl;

    TreeFileWriter  xml_file_out("t_xml.output3");
  QDPIO::cout << "line= " << __LINE__ << endl;

    push(xml_file_out,"root_for_output3");

  QDPIO::cout << "line= " << __LINE__ << endl;

    TreeArrayWriter  xml_out(xml_file_out, 3);
  QDPIO::cout << "line= " << __LINE__ << endl;

    push(xml_out,"this_is_an_array"); // This starts the array

  QDPIO::cout << "line= " << __LINE__ << endl;

    for(int i=0; i < xml_out.size(); ++i)
    {
      int x = -42;

      // Cannot use a writeElem because all the overloadings will not be present
  QDPIO::cout << "line= " << __LINE__ << endl;

      write(xml_out,xml_out.nextElem(),x);
  QDPIO::cout << "line= " << __LINE__ << endl;

    }
    
  QDPIO::cout << "line= " << __LINE__ << endl;

    pop(xml_out);  // This closes the array
  QDPIO::cout << "line= " << __LINE__ << endl;


    QDP_info("done with TreeArrayWrtiter tests");
  }
  catch (const string& e)
  {
    QDPIO::cerr << "Error: reading array: error=" << e << endl;
    QDP_abort(1);
  }
  catch(std::bad_cast) 
  {
    QDPIO::cerr << "Error: caught cast error here: line= " << __LINE__ << endl;
    QDP_abort(1);
  }

  QDPIO::cout << "line= " << __LINE__ << endl;

  // Try out a complex (complicated with structs) array context
  try
  { 
    TreeFileWriter  xml_file_out("t_xml.output3");
    push(xml_file_out,"root_for_output3");
    TreeArrayWriter  xml_out(xml_file_out, 3);
    push(xml_out,"this_is_an_array_of_structs"); // This starts the array

    for(int i=0; i < xml_out.size(); ++i)
    {
      int x = -42;
      pushElem(xml_out);  // NOTE: push element name within an array
      write(xml_out,"x",x);
      popElem(xml_out);
    }
    
    pop(xml_out);  // This closes the array

    QDP_info("done with TreeArrayWrtiter tests");
  }
  catch (const string& e)
  {
    QDPIO::cerr << "Error: writing array: error=" << e << endl;
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
    // Test moving around with derived readers
    TreeReader fromxml_orig;
    fromxml_orig.open("t_xml.input2");

    TreeReader fromxml(fromxml_orig, "/complex_xml");

    QDP::Seed seed;
    read(fromxml,"seedThingy",seed);
    QDPIO::cout << "seed = " << seed <<  "  node=" << Layout::nodeNumber() << endl;

    multi1d<int> arrayInt;
    read(fromxml,"arrayInt",arrayInt);
    for(int i=0; i < arrayInt.size(); ++i)
      QDPIO::cout << "arrayInt[" << i << "] = " << arrayInt[i]  << "  node=" << Layout::nodeNumber() << endl;

    multi1d<Double> arrayDouble;
    read(fromxml,"arrayDouble",arrayDouble);
    for(int i=0; i < arrayDouble.size(); ++i)
      QDPIO::cout << "arrayDouble[" << i << "] = " << arrayDouble[i] << "  node=" << Layout::nodeNumber() << endl;

    multi1d<Complex> arrayComplex;
    read(fromxml,"arrayComplex",arrayComplex);
    for(int i=0; i < arrayComplex.size(); ++i)
      QDPIO::cout << "arrayComplex[" << i << "] = ("
		  << real(arrayComplex[i]) << "," 
		  << imag(arrayComplex[i]) << ")" << endl;

    QDP_info("done with array snippet tests");
  }
  catch(const string& e)
  {
    QDPIO::cerr << "Error: reading array snippets: error=" << e << endl;
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
    // Try modifying a reader
    // First write something to modify
    TreeFileWriter  toxml("t_xml.input3");
    push(toxml, "root_for_input3");
    write(toxml, "Mass", Double(17.3));
    pop(toxml);
    toxml.close();

    // Try modifying a reader
    TreeReader fromxml;
    fromxml.open("t_xml.input3");

//    fromxml.set<QDP::Double>("/root_for_input3/Mass", Double(0.5));

    // turn back into a string
    TreeBufferWriter new_writer;
    new_writer << fromxml;

    std::string new_writer_string = new_writer.printCurrentContext();

    TreeFileWriter toxml_again("t_xml.compare_to_input3");
    push(toxml_again, "compare_to_input3");
    write(toxml_again, "content_of_writer", new_writer_string);
    pop(toxml_again);
    toxml_again.close();
  }
  catch(const string& e)
  {
    QDPIO::cerr << "Error: modifying a reader: error=" << e << endl;
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
