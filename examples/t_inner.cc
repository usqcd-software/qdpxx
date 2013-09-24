// $Id: t_basic.cc,v 1.4 2004-08-11 18:53:10 edwards Exp $
/*! \file
 *  \brief Test some simple basic routines
 */

#include "qdp.h"

using namespace QDP;


int main(int argc, char *argv[])
{
  // Put the machine into a known state
  QDP_initialize(&argc, &argv);

  QDP_PUSH_PROFILE(QDP::getProfileLevel());

  // Setup the layout
  const int foo[] = {32,4,4,8};
  // const int foo[] = {16,8,8,4};
  multi1d<int> nrow(Nd);
  nrow = foo;  // Use only Nd elements
  Layout::setLattSize(nrow);
  Layout::create();

  XMLFileWriter xml_out("t_inner.xml");
  push(xml_out, "t_inner");
  push(xml_out,"lattice");
  write(xml_out,"nrow", nrow);
  write(xml_out, "Nd", Nd);
  write(xml_out, "Nc", Nc);
  write(xml_out,"logicalSize",Layout::logicalSize());
  pop(xml_out); // lattice

  // Test 1
  {
    ILattice<float,8> src;

    src.elem(0) = 0.1;
    src.elem(1) = 0.2;
    src.elem(2) = 0.3;
    src.elem(3) = 0.4;
    src.elem(4) = 0.5;
    src.elem(5) = 0.6;
    src.elem(6) = 0.7;
    src.elem(7) = 0.8;

    ILattice<float, 8> d;
    d.elem(0) = -1.1;
    d.elem(1) = -1.2;
    d.elem(2) = -1.3;
    d.elem(3) = -1.4;
    d.elem(4) = -1.5;
    d.elem(5) = -1.6;
    d.elem(6) = -1.7;
    d.elem(7) = -1.8;

    ILattice<bool, 8> mask;
    mask.elem(0) = true;
    mask.elem(1) = true;
    mask.elem(2) = false;
    mask.elem(3) = true;
    mask.elem(4) = false;
    mask.elem(5) = false;
    mask.elem(6) = false;
    mask.elem(7) = true;

    ILattice<float,8> d2 = d;
    
    // Code from Unspecalized function
    for(int i=0; i < 8; ++i) {
      if( mask.elem(i) ) {
	d2.elem(i) = src.elem(i);
      }
    }
    
    // AVX one hopefully
    copy_inner_mask(d, mask, src);

    bool success = false;
    for(int i=0; i < 8; i++){ 
      cout << "d2(" << i << ") = " << d2.elem(i) <<"   d("<<i<<") = " << d.elem(i) << "  mask(" << i<<") = " << mask.elem(i) << endl;
    }
  }

  // Test 2
  {
    ILattice<double,4> src;

    src.elem(0) = 0.1;
    src.elem(1) = 0.2;
    src.elem(2) = 0.3;
    src.elem(3) = 0.4;

    ILattice<double,4> d;
    d.elem(0) = -1.1;
    d.elem(1) = -1.2;
    d.elem(2) = -1.3;
    d.elem(3) = -1.4;

    ILattice<bool, 4> mask;
    mask.elem(0) = true;
    mask.elem(1) = true;
    mask.elem(2) = false;
    mask.elem(3) = true;

    ILattice<double,4> d2 = d;
    
    // Code from Unspecalized function
    for(int i=0; i < 4; ++i) {
      if( mask.elem(i) ) {
	d2.elem(i) = src.elem(i);
      }
    }
    
    // AVX one hopefully
    copy_inner_mask(d, mask, src);

    bool success = false;
    for(int i=0; i < 4; i++){ 
      cout << "d2(" << i << ") = " << d2.elem(i) <<"   d("<<i<<") = " << d.elem(i) << "  mask(" << i<<") = " << mask.elem(i) << endl;
    }
  }


  pop(xml_out); // t_inner
  xml_out.close(); //close file

  QDP_POP_PROFILE();

  // Time to bolt
  QDP_finalize();

  exit(0);
}


