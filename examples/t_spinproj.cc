// $Id: t_spinproj.cc,v 1.1 2007-01-26 19:32:11 bjoo Exp $

#include <iostream>
#include <iomanip>
#include <cstdio>

#include <time.h>

#include "qdp.h"



using namespace std;
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

  LatticeHalfFermion H;
  HalfFermion h;
  Fermion v;

  LatticeHalfFermion H2;
  LatticeFermion V;
  LatticeFermion V2;
  LatticeFermion V3;
  gaussian(V);
  LatticeHalfFermion diff;
  LatticeFermion diff_v;

  /*
   *  Proj 0 Plus
   */

  // Old Way
  for(int site=all.start(); site <= all.end(); site++) { 
    v.elem() = V.elem(site);
    h = spinProjectDir0Plus(v);
    H2.elem(site) = h.elem();
  }

  // New Way
  H = spinProjectDir0Plus(V);
  diff = H2 - H;


  QDPIO::cout << "Proj0+: || old - new || / || old || = " << sqrt(norm2(diff)) << endl;

  /*
   *  Proj 1 Plus
   */

  // Old Way
  for(int site=all.start(); site <= all.end(); site++) { 
    v.elem() = V.elem(site);
    h = spinProjectDir1Plus(v);
    H2.elem(site) = h.elem();
  }

  // New Way
  H = spinProjectDir1Plus(V);
  diff = H2 - H;
  QDPIO::cout << "Proj1+: || old - new || / || old || = " << sqrt(norm2(diff)) << endl;

  /*
   *  Proj 2 Plus
   */

  // Old Way
  for(int site=all.start(); site <= all.end(); site++) { 
    v.elem() = V.elem(site);
    h = spinProjectDir2Plus(v);
    H2.elem(site) = h.elem();
  }

  // New Way
  H = spinProjectDir2Plus(V);
  diff = H2 - H;
  QDPIO::cout << "Proj2+: || old - new || / || old || = " << sqrt(norm2(diff)) << endl;

  /*
   *  Proj 3 Plus
   */

  // Old Way
  for(int site=all.start(); site <= all.end(); site++) { 
    v.elem() = V.elem(site);
    h = spinProjectDir3Plus(v);
    H2.elem(site) = h.elem();
  }

  // New Way
  H = spinProjectDir3Plus(V);
  diff = H2 - H;
  QDPIO::cout << "Proj3+: || old - new || / || old || = " << sqrt(norm2(diff)) << endl;

  /*
   *  Proj 0 Minus
   */

  // Old Way
  for(int site=all.start(); site <= all.end(); site++) { 
    v.elem() = V.elem(site);
    h = spinProjectDir0Minus(v);
    H2.elem(site) = h.elem();
  }

  // New Way
  H = spinProjectDir0Minus(V);
  diff = H2 - H;
  QDPIO::cout << "Proj0-: || old - new || / || old || = " << sqrt(norm2(diff)) << endl;

  /*
   *  Proj 1 Minus
   */

  // Old Way
  for(int site=all.start(); site <= all.end(); site++) { 
    v.elem() = V.elem(site);
    h = spinProjectDir1Minus(v);
    H2.elem(site) = h.elem();
  }

  // New Way
  H = spinProjectDir1Minus(V);
  diff = H2 - H;
  QDPIO::cout << "Proj1-: || old - new || / || old || = " << sqrt(norm2(diff)) << endl;

  /*
   *  Proj 2 Minus
   */

  // Old Way
  for(int site=all.start(); site <= all.end(); site++) { 
    v.elem() = V.elem(site);
    h = spinProjectDir2Minus(v);
    H2.elem(site) = h.elem();
  }

  // New Way
  H = spinProjectDir2Minus(V);
  diff = H2 - H;
  QDPIO::cout << "Proj2-: || old - new || / || old || = " << sqrt(norm2(diff)) << endl;

  /*
   *  Proj 2 Minus
   */

  // Old Way
  for(int site=all.start(); site <= all.end(); site++) { 
    v.elem() = V.elem(site);
    h = spinProjectDir2Minus(v);
    H2.elem(site) = h.elem();
  }

  // New Way
  H = spinProjectDir2Minus(V);
  diff = H2 - H;
  QDPIO::cout << "Proj2-: || old - new || / || old || = " << sqrt(norm2(diff)) << endl;

  /*
   *  Proj 3 Minus
   */

  // Old Way
  for(int site=all.start(); site <= all.end(); site++) { 
    v.elem() = V.elem(site);
    h = spinProjectDir3Minus(v);
    H2.elem(site) = h.elem();
  }

  // New Way
  H = spinProjectDir3Minus(V);
  diff = H2 - H;
  QDPIO::cout << "Proj3-: || old - new || / || old || = " << sqrt(norm2(diff)) << endl;

  gaussian(H);

  /* 
   * Recon 0 Plus
   */
  
  // Old Way
  for(int site=all.start(); site <= all.end(); site++) { 
    h.elem() = H.elem(site);
    v = spinReconstructDir0Plus(h);
    V.elem(site) = v.elem();
  }
  
  // New Way
  V2 = spinReconstructDir0Plus(H);
  diff_v = V - V2;
  QDPIO::cout << "Recon0+: || old - new || / || old || = " << sqrt(norm2(diff_v)) << endl;

  /* 
   * Recon 1 Plus
   */
  
  // Old Way
  for(int site=all.start(); site <= all.end(); site++) { 
    h.elem() = H.elem(site);
    v = spinReconstructDir1Plus(h);
    V.elem(site) = v.elem();
  }
  
  // New Way
  V2 = spinReconstructDir1Plus(H);
  diff_v = V - V2;
  QDPIO::cout << "Recon1+: || old - new || / || old || = " << sqrt(norm2(diff_v)) << endl;

  /* 
   * Recon 2 Plus
   */
  
  // Old Way
  for(int site=all.start(); site <= all.end(); site++) { 
    h.elem() = H.elem(site);
    v = spinReconstructDir2Plus(h);
    V.elem(site) = v.elem();
  }
  
  // New Way
  V2 = spinReconstructDir2Plus(H);
  diff_v = V - V2;
  QDPIO::cout << "Recon2+: || old - new || / || old || = " << sqrt(norm2(diff_v)) << endl;


  /* 
   * Recon 3 Plus
   */
  
  // Old Way
  for(int site=all.start(); site <= all.end(); site++) { 
    h.elem() = H.elem(site);
    v = spinReconstructDir3Plus(h);
    V.elem(site) = v.elem();
  }
  
  // New Way
  V2 = spinReconstructDir3Plus(H);
  diff_v = V - V2;
  QDPIO::cout << "Recon3+: || old - new || / || old || = " << sqrt(norm2(diff_v)) << endl;


  /* 
   * Recon 0 Minus
   */
  
  // Old Way
  for(int site=all.start(); site <= all.end(); site++) { 
    h.elem() = H.elem(site);
    v = spinReconstructDir0Minus(h);
    V.elem(site) = v.elem();
  }
  
  // New Way
  V2 = spinReconstructDir0Minus(H);
  diff_v = V - V2;
  QDPIO::cout << "Recon0-: || old - new || / || old || = " << sqrt(norm2(diff_v)) << endl;

  /* 
   * Recon 1 Minus
   */
  
  // Old Way
  for(int site=all.start(); site <= all.end(); site++) { 
    h.elem() = H.elem(site);
    v = spinReconstructDir1Minus(h);
    V.elem(site) = v.elem();
  }
  
  // New Way
  V2 = spinReconstructDir1Minus(H);
  diff_v = V - V2;
  QDPIO::cout << "Recon1-: || old - new || / || old || = " << sqrt(norm2(diff_v)) << endl;

  /* 
   * Recon 2 Minus
   */
  
  // Old Way
  for(int site=all.start(); site <= all.end(); site++) { 
    h.elem() = H.elem(site);
    v = spinReconstructDir2Minus(h);
    V.elem(site) = v.elem();
  }
  
  // New Way
  V2 = spinReconstructDir2Minus(H);
  diff_v = V - V2;
  QDPIO::cout << "Recon2-: || old - new || / || old || = " << sqrt(norm2(diff_v)) << endl;


  /* 
   * Recon 3 Minus
   */
  
  // Old Way
  for(int site=all.start(); site <= all.end(); site++) { 
    h.elem() = H.elem(site);
    v = spinReconstructDir3Minus(h);
    V.elem(site) = v.elem();
  }
  
  // New Way
  V2 = spinReconstructDir3Minus(H);
  diff_v = V - V2;
  QDPIO::cout << "Recon3-: || old - new || / || old || = " << sqrt(norm2(diff_v)) << endl;


  /* 
   * Add Recon 0 Plus
   */
  
  // Old Way
  gaussian(V);
  V2=V;
  for(int site=all.start(); site <= all.end(); site++) { 
    h.elem() = H.elem(site);
    v = spinReconstructDir0Plus(h);
    V2.elem(site) += v.elem();
  }
  
  // New Way
  V3 = V;
  V3 += spinReconstructDir0Plus(H);
  diff_v = V2 - V3;
  QDPIO::cout << "AddRecon0+: || old - new || / || old || = " << sqrt(norm2(diff_v)) << endl;

  /* 
   * Add Recon 1 Plus
   */
  
  // Old Way
  gaussian(V);
  V2=V;
  for(int site=all.start(); site <= all.end(); site++) { 
    h.elem() = H.elem(site);
    v = spinReconstructDir1Plus(h);
    V2.elem(site) += v.elem();
  }
  
  // New Way
  V3 = V;
  V3 += spinReconstructDir1Plus(H);
  diff_v = V2 - V3;
  QDPIO::cout << "AddRecon1+: || old - new || / || old || = " << sqrt(norm2(diff_v)) << endl;

  /* 
   * Add Recon 2 Plus
   */
  
  // Old Way
  gaussian(V);
  V2=V;
  for(int site=all.start(); site <= all.end(); site++) { 
    h.elem() = H.elem(site);
    v = spinReconstructDir2Plus(h);
    V2.elem(site) += v.elem();
  }
  
  // New Way
  V3 = V;
  V3 += spinReconstructDir2Plus(H);
  diff_v = V2 - V3;
  QDPIO::cout << "AddRecon2+: || old - new || / || old || = " << sqrt(norm2(diff_v)) << endl;

  /* 
   * Add Recon 3 Plus
   */
  
  // Old Way
  gaussian(V);
  V2=V;
  for(int site=all.start(); site <= all.end(); site++) { 
    h.elem() = H.elem(site);
    v = spinReconstructDir3Plus(h);
    V2.elem(site) += v.elem();
  }
  
  // New Way
  V3 = V;
  V3 += spinReconstructDir3Plus(H);
  diff_v = V2 - V3;
  QDPIO::cout << "AddRecon3+: || old - new || / || old || = " << sqrt(norm2(diff_v)) << endl;


  /* 
   * Add Recon 0 Minus
   */
  
  // Old Way
  gaussian(V);
  V2=V;
  for(int site=all.start(); site <= all.end(); site++) { 
    h.elem() = H.elem(site);
    v = spinReconstructDir0Minus(h);
    V2.elem(site) += v.elem();
  }
  
  // New Way
  V3 = V;
  V3 += spinReconstructDir0Minus(H);
  diff_v = V2 - V3;
  QDPIO::cout << "AddRecon0-: || old - new || / || old || = " << sqrt(norm2(diff_v)) << endl;

  /* 
   * Add Recon 1 Minus
   */
  
  // Old Way
  gaussian(V);
  V2=V;
  for(int site=all.start(); site <= all.end(); site++) { 
    h.elem() = H.elem(site);
    v = spinReconstructDir1Minus(h);
    V2.elem(site) += v.elem();
  }
  
  // New Way
  V3 = V;
  V3 += spinReconstructDir1Minus(H);
  diff_v = V2 - V3;
  QDPIO::cout << "AddRecon1-: || old - new || / || old || = " << sqrt(norm2(diff_v)) << endl;

  /* 
   * Add Recon 2 Minus
   */
  
  // Old Way
  gaussian(V);
  V2=V;
  for(int site=all.start(); site <= all.end(); site++) { 
    h.elem() = H.elem(site);
    v = spinReconstructDir2Minus(h);
    V2.elem(site) += v.elem();
  }
  
  // New Way
  V3 = V;
  V3 += spinReconstructDir2Minus(H);
  diff_v = V2 - V3;
  QDPIO::cout << "AddRecon2-: || old - new || / || old || = " << sqrt(norm2(diff_v)) << endl;

  /* 
   * Add Recon 3 Minus
   */
  
  // Old Way
  gaussian(V);
  V2=V;
  for(int site=all.start(); site <= all.end(); site++) { 
    h.elem() = H.elem(site);
    v = spinReconstructDir3Minus(h);
    V2.elem(site) += v.elem();
  }
  
  // New Way
  V3 = V;
  V3 += spinReconstructDir3Minus(H);
  diff_v = V2 - V3;
  QDPIO::cout << "AddRecon3-: || old - new || / || old || = " << sqrt(norm2(diff_v)) << endl;


  
  // Time to bolt
  QDP_finalize();

  exit(0);
}
  
