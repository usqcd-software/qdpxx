// $Id: t_dslashm.cc,v 1.14 2003-10-22 02:44:18 edwards Exp $
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

  // Setup the layout
  const int foo[] = {4,4,4,4};
  multi1d<int> nrow(Nd);
  nrow = foo;  // Use only Nd elements
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

  {
    int isign = +1;
    int cb = 0;
    QDPIO::cout << "Applying D" << endl;
      
    clock_t myt1=clock();
    for(int i=0; i < iter; i++)
      dslash(chi, u, psi, isign, cb);
    clock_t myt2=clock();
      
    double mydt=(double)(myt2-myt1)/((double)(CLOCKS_PER_SEC));
    mydt=1.0e6*mydt/((double)(iter*(Layout::vol()/2)));
      
    QDPIO::cout << "cb = " << cb << " isign = " << isign << endl;
    QDPIO::cout << "The time per lattice point is "<< mydt << " micro sec" 
		<< " (" <<  (double)(1392.0f/mydt) << ") Mflops " << endl;
  }

#if 0
  NmlWriter nml("t_dslashm.nml");
  Write(nml,Nd);
  Write(nml,Nc);
  Write(nml,Ns);
  Write(nml,nrow);
  Write(nml,psi);
  Write(nml,chi);
#endif

  // Time to bolt
  QDP_finalize();

  exit(0);
}
