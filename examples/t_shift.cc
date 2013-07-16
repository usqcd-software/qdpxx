// -*- C++ -*-
//
// $Id: t_shift.cc,v 1.5 2005-03-21 05:31:08 edwards Exp $
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
    const int foo[] = {4,4,4,4};
    nrow = foo;
  }
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

      b = shift(a,BACKWARD,mu);
      write(xml,"b", b);
      pop(xml);
    }
    pop(xml);
    xml.close();
  }
#endif
  
  // Time to bolt
  QDP_finalize();

  exit(0);
}
