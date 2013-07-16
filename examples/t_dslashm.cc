// $Id: t_dslashm.cc,v 1.20 2007-02-24 01:00:29 bjoo Exp $
/*! \file
 *  \brief Test the Wilson-Dirac operator (dslash)
 */

#include <iostream>
#include <cstdio>

#include "qdp.h"
#include "examples.h"

#include <sys/time.h>

using namespace QDP;


int main(int argc, char **argv)
{
  // Put the machine into a known state
  QDP_initialize(&argc, &argv);

  multi1d<int> nrow(Nd);

  if (argc >= 5) {
    nrow[0] = atoi(argv[1]);
    nrow[1] = atoi(argv[2]);
    nrow[2] = atoi(argv[3]);
    nrow[3] = atoi(argv[4]);
    
    QDP_info ("Lattice size %dx%dx%dx%d\n", nrow[0], nrow[1], nrow[2], nrow[3]);
  }
  else {
    // Setup the layout
    // const int foo[] = {4,2,2,2};
    // const int foo[] = {16,2,16,2};
    const int foo[] = {32, 32, 32, 32};
    nrow = foo;  // Use only Nd elements
  }
  Layout::setLattSize(nrow);
  Layout::create();

  //! Test out propagators
  multi1d<LatticeColorMatrix> u(Nd);
  for(int m=0; m < u.size(); ++m)
    gaussian(u[m]);

  LatticeFermion psi, chi;
  random(psi);
  chi = zero;

  int iter = 100;

  QDP::StopWatch swatch;

#if 1
  swatch.reset();
  {
    int isign = +1;
    int cb = 0;
    QDPIO::cout << "Applying D" << endl;

    swatch.start();
    for(int i=0; i < iter; i++) {
      dslash(chi, u, psi, isign, cb);
    }
    swatch.stop();

    double mydt=swatch.getTimeInSeconds();

    double idt = 1.0e6*mydt/((double)(iter));

    mydt=1.0e6*mydt/((double)(iter*(Layout::vol()/2)));

      
    QDPIO::cout << "cb = " << cb << " isign = " << isign << endl;
    QDPIO::cout << "The time per lattice point is "<< mydt << " micro sec" 
		<< " each iteration takes " << idt << " micro sec" 
		<< " (" <<  (double)(1392.0f/mydt) << ") Mflops " << endl;

    // take norm2 for verification
    QDPIO::cout << "|chi|^2 = " << norm2(chi, rb[cb]) << endl;
  }

#endif

#if 1
  chi = zero;

  swatch.reset();
  {
    int isign = +1;
    int cb = 0;
    QDPIO::cout << "Applying D" << endl;
      
    swatch.start();
    for(int i=0; i < iter; i++)
      dslash2(chi, u, psi, isign, cb);
    swatch.stop();
      
    double mydt=swatch.getTimeInSeconds();
    mydt=1.0e6*mydt/((double)(iter*(Layout::vol()/2)));
      
    QDPIO::cout << "cb = " << cb << " isign = " << isign << endl;
    QDPIO::cout << "The time per lattice point is "<< mydt << " micro sec" 
		<< " (" <<  (double)(1392.0f/mydt) << ") Mflops " << endl;

    // take norm2 for verification
    QDPIO::cout << "|chi|^2 = " << norm2(chi, rb[cb]) << endl;
  }
#endif

#if 0
  XMLFileWriter xml("t_dslashm.xml");
  push(xml,"t_dslashm");
  write(xml,"Nd", Nd);
  write(xml,"Nc", Nc);
  write(xml,"Ns", Ns);
  write(xml,"nrow", nrow);
  write(xml,"psi", psi);
  write(xml,"chi", chi);
  pop(xml);
  xml.close();
#endif

#if 0
  // check overlapped sum
  {
    int isign = +1;
    int cb = 0;
    RealD sumtest = dslash3 (u, psi, isign, cb);

    QDPIO::cout << "cb = " << cb << " isign = " << isign << endl;

    // take norm2 for verification
    QDPIO::cout << "sum with shift test = " << sumtest << endl;
  }
#endif

#if 1
  LatticeFermion chi2;
  random(psi);
  chi = zero;
  chi2 = zero;
  for(int isign=-1; isign < 2; isign+=2) { 
    for(int cb=0; cb<2; cb++) {
      int otherCB= cb == 0 ? 1 : 0;
      dslash(chi, u, psi, isign, cb);
      dslash2(chi2, u, psi, isign, cb);
      LatticeFermion diff;
      diff[rb[cb]]= chi2 - chi;
      QDPIO::cout << "isign="<<isign<<" cb=" << cb << " Diff = " << sqrt( norm2(diff,rb[cb]) / norm2(psi, rb[otherCB]))<< endl;
    }
  }
#endif


  // Time to bolt
  QDP_finalize();

  exit(0);
}
