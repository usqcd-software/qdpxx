// $Id: t_exotic.cc,v 1.4 2003-12-08 21:41:03 edwards Exp $
/*! \file
 *  \brief Test various exotic qdp routines
 */

#include <iostream>
#include <cstdio>

#include "qdp.h"
#include "examples.h"

using namespace QDP;

int main(int argc, char *argv[])
{
  // Put the machine into a known state
  QDP_initialize(&argc, &argv);

  // Setup the layout
  const int foo[] = {2,2,2,2};
  multi1d<int> nrow(Nd);
  nrow = foo;  // Use only Nd elements
  Layout::setLattSize(nrow);
  Layout::create();

  NmlWriter nml("t_exotic.nml");

  push(nml,"lattis");
  Write(nml,Nd);
  Write(nml,Nc);
  Write(nml,nrow);
  pop(nml);


  // Construct a random gauge transformation
  LatticeColorMatrix g;

  gaussian(g);
  taproj(g);
  expm12(g);
  reunit(g);

#if 0
  // Try out colorContract
  {
    LatticeColorMatrix a,b,c;
    gaussian(a);
    gaussian(b);
    gaussian(c);

    LatticeComplex lc1 = colorContract(a,b,c);

    push(nml,"color_contract_orig");
    Write(nml,lc1);
    pop(nml);
   
    // Do a random gauge transformation
    LatticeColorMatrix tmp;
    tmp = g * a * adj(g);  a = tmp;
    tmp = g * b * adj(g);  b = tmp;
    tmp = g * c * adj(g);  c = tmp;

    // Try colorcontract again
    LatticeComplex lc2 = colorContract(a,b,c);
    push(nml,"color_contract_gauge_transf");
    Write(nml,lc2);
    pop(nml);
  }
#endif

  // Try out chiralProject{Plus,Minus}
  {
    LatticeFermion psi, chi1, chi2;
    gaussian(psi);

    chi1 = 0.5*(psi + Gamma(Ns*Ns-1)*psi);
    chi2 = chiralProjectPlus(psi);
    QDPIO::cout << "|chi1|^2 = " << norm2(chi1) << endl
		<< "|chi2|^2 = " << norm2(chi2) << endl
		<< "|chi2 - chi1|^2 = " << norm2(chi2-chi1) << endl;

    chi1 = 0.5*(psi - Gamma(Ns*Ns-1)*psi);
    chi2 = chiralProjectMinus(psi);
    QDPIO::cout << "|chi1|^2 = " << norm2(chi1) << endl
		<< "|chi2|^2 = " << norm2(chi2) << endl
		<< "|chi2 - chi1|^2 = " << norm2(chi2-chi1) << endl;
  }

  nml.flush();

  // Time to bolt
  QDP_finalize();

  exit(0);
}
