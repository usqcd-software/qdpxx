// $Id: t_su3.cc,v 1.4 2007-06-08 20:07:38 bjoo Exp $

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
  const int foo[] = {6,6,6,4};
  multi1d<int> nrow(Nd);
  nrow = foo;  // Use only Nd elements
  Layout::setLattSize(nrow);
  Layout::create();

#if QDP_USE_BAGEL_QDP == 1

  
  LatticeColorMatrix a, b, c;
  LatticeColorMatrix a2;
  LatticeColorMatrix diff;

  // -----------------------------------------------------------------
  // MM
  // -----------------------------------------------------------------
  gaussian(b);
  gaussian(c);
  unsigned int num_sites=all.end()-all.start()+1;
  
  a = b*c;
  qdp_su3_mm(
	     &(a2.elem(all.start()).elem().elem(0,0).real()),
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


  //-----------------------------------------------------------------
  // MA
  //-----------------------------------------------------------------
  gaussian(b);
  gaussian(c);

  a = b*adj(c);
  qdp_su3_ma(&(a2.elem(all.start()).elem().elem(0,0).real()),
  	     &(b.elem(all.start()).elem().elem(0,0).real()),
  	     &(c.elem(all.start()).elem().elem(0,0).real()),
  	     num_sites);

  diff = a - a2;

  QDPIO::cout << "MA: || diff || = " << sqrt(norm2(diff)) << endl;

  swatch.reset();
  swatch.start();
  for(int i=0; i < 5000; i++) { 
    a = b*adj(c);
  }
  swatch.stop();
  original_secs = swatch.getTimeInSeconds();

  swatch.reset();
  swatch.start();
  for(int i=0; i < 5000; i++) {
    qdp_su3_ma(&(a2.elem(all.start()).elem().elem(0,0).real()),
	       &(b.elem(all.start()).elem().elem(0,0).real()),
	       &(c.elem(all.start()).elem().elem(0,0).real()),
	       num_sites);
  }
  swatch.stop();

  qdp_secs = swatch.getTimeInSeconds();

  QDPIO::cout << "MA: original  seconds= " << original_secs << endl;
  QDPIO::cout << "MA: bagel_qdp seconds= " << qdp_secs << endl;

  //-----------------------------------------------------------------
  // AM
  //-----------------------------------------------------------------
  gaussian(b);
  gaussian(c);

  a = adj(b)*c;
  qdp_su3_am(&(a2.elem(all.start()).elem().elem(0,0).real()),
  	     &(b.elem(all.start()).elem().elem(0,0).real()),
  	     &(c.elem(all.start()).elem().elem(0,0).real()),
  	     num_sites);

  diff = a - a2;

  QDPIO::cout << "AM: || diff || = " << sqrt(norm2(diff)) << endl;

  swatch.reset();
  swatch.start();
  for(int i=0; i < 5000; i++) { 
    a = adj(b)*c;
  }
  swatch.stop();
  original_secs = swatch.getTimeInSeconds();

  swatch.reset();
  swatch.start();
  for(int i=0; i < 5000; i++) {
    qdp_su3_am(&(a2.elem(all.start()).elem().elem(0,0).real()),
	       &(b.elem(all.start()).elem().elem(0,0).real()),
	       &(c.elem(all.start()).elem().elem(0,0).real()),
	       num_sites);
  }
  swatch.stop();

  qdp_secs = swatch.getTimeInSeconds();

  QDPIO::cout << "AM: original  seconds= " << original_secs << endl;
  QDPIO::cout << "AM: bagel_qdp seconds= " << qdp_secs << endl;

  //-----------------------------------------------------------------
  // AA
  //-----------------------------------------------------------------
  gaussian(b);
  gaussian(c);

  a = adj(b)*adj(c);
  qdp_su3_aa(&(a2.elem(all.start()).elem().elem(0,0).real()),
  	     &(b.elem(all.start()).elem().elem(0,0).real()),
  	     &(c.elem(all.start()).elem().elem(0,0).real()),
  	     num_sites);

  diff = a - a2;

  QDPIO::cout << "AA: || diff || = " << sqrt(norm2(diff)) << endl;

  swatch.reset();
  swatch.start();
  for(int i=0; i < 5000; i++) { 
    a = adj(b)*adj(c);
  }
  swatch.stop();
  original_secs = swatch.getTimeInSeconds();

  swatch.reset();
  swatch.start();
  for(int i=0; i < 5000; i++) {
    qdp_su3_aa(&(a2.elem(all.start()).elem().elem(0,0).real()),
	       &(b.elem(all.start()).elem().elem(0,0).real()),
	       &(c.elem(all.start()).elem().elem(0,0).real()),
	       num_sites);
  }
  swatch.stop();

  qdp_secs = swatch.getTimeInSeconds();

  QDPIO::cout << "AA: original  seconds= " << original_secs << endl;
  QDPIO::cout << "AA: bagel_qdp seconds= " << qdp_secs << endl;

#endif


  // Time to bolt
  QDP_finalize();

  exit(0);
}
