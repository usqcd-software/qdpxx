// -*- C++ -*-
//
// $Id: t_shift.cc,v 1.1 2003-09-04 02:07:58 edwards Exp $
//
/*! \file
 *  \brief Silly little internal test code
 */


#include "qdp.h"
#include "qdp_util.h"

//using namespace std;

using namespace QDP;


int main(int argc, char *argv[])
{
  // Put the machine into a known state
  QDP_initialize(&argc, &argv);

  // Setup the layout
  const int foo[] = {4,4,4,4};
  multi1d<int> nrow(Nd);
  nrow = foo;  // Use only Nd elements
  Layout::setLattSize(nrow);
  Layout::create();

#if 1
  {
    NmlWriter nml("t_shift.nml");
    push(nml,"test1");

    LatticeReal a,b;
    random(a);

    Write(nml,a);
    for(int mu=0; mu < Nd; ++mu)
    {
      QDP_info("Newdir: mu= %d",mu);
      push(nml,"newdir");
      Write(nml,mu);
      b = shift(a,FORWARD,mu);
      Write(nml,b);
    }
    pop(nml);
  }
#endif
  
  // Time to bolt
  QDP_finalize();

  exit(0);
}
