// -*- C++ -*-
//
// $Id: t_shift.cc,v 1.4 2004-12-10 12:02:08 bjoo Exp $
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

  QDPIO::cout << "DONE STARTUP" << endl << flush;
#if 1
  {
    XMLFileWriter xml("t_shift.xml");
    push(xml,"test1");

    LatticeReal a,b;
    random(a);

    write(xml,"a", a);
    for(int mu=0; mu < Nd; ++mu)
    {
      QDP_info("Newdir: mu= %d",mu);
      push(xml,"newdir");
      write(xml,"mu", mu);

      b = shift(a,FORWARD,mu);
      write(xml,"b", b);
      pop(xml);
    }
      pop(xml);
  }
#endif
  
  // Time to bolt
  QDP_finalize();

  exit(0);
}
