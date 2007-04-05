// $Id: t_su3.cc,v 1.2 2007-04-05 20:54:43 bjoo Exp $

#include <iostream>
#include <cstdio>

#include <time.h>

#include "qdp.h"


using namespace QDP;

int main(int argc, char *argv[])
{
  // Put the machine into a known state
  QDP_initialize(&argc, &argv);

  // Setup the layout
  const int foo[] = {2,3,3,3};
  multi1d<int> nrow(Nd);
  nrow = foo;  // Use only Nd elements
  Layout::setLattSize(nrow);
  Layout::create();

#ifdef QDP_USE_BAGEL_QDP

  LatticeColorMatrix a, b, c;
  LatticeColorMatrix a2;
  LatticeColorMatrix diff;

  gaussian(b);
  gaussian(c);

  int num_sites=all.end()-all.start()+1;
  a = b*c;
  qdp_su3_mm(&(a2.elem(all.start()).elem().elem(0,0).real()),
  	     &(b.elem(all.start()).elem().elem(0,0).real()),
  	     &(c.elem(all.start()).elem().elem(0,0).real()),
  	     num_sites);

  diff = a - a2;

  QDPIO::cout << "MM: || diff || = " << sqrt(norm2(diff)) << endl;

  QDP::StopWatch swatch;
  swatch.reset();
  swatch.start();
  for(int i=0; i < 5000; i++) { 
    a = b*c;
  }
  swatch.stop();
  double original_secs = swatch.getTimeInSeconds();

  swatch.reset();
  swatch.start();
  for(int i=0; i < 5000; i++) {
    qdp_su3_mm(&(a2.elem(all.start()).elem().elem(0,0).real()),
	       &(b.elem(all.start()).elem().elem(0,0).real()),
	       &(c.elem(all.start()).elem().elem(0,0).real()),
	       num_sites);
  }
  swatch.stop();

  double qdp_secs = swatch.getTimeInSeconds();

  QDPIO::cout << "MM: original  seconds= " << original_secs << endl;
  QDPIO::cout << "MM: bagel_qdp seconds= " << qdp_secs << endl;

#endif


  // Time to bolt
  QDP_finalize();

  exit(0);
}
