// $Id: t_spinproj.cc,v 1.3 2007-02-06 15:01:57 bjoo Exp $

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

  LatticeHalfFermion H2,H3;
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

  LatticeColorMatrix u;
  gaussian(u);
  gaussian(V);

  ColorMatrix colmat;
  colmat.elem() = u.elem(0).elem();
  Fermion ferm;
  ferm.elem() = V.elem(0);

  Fermion res = adj(colmat)*ferm;
  
  Fermion res2;
 
  _inline_mult_adj_su3_mat_vec( colmat.elem().elem(),
				ferm.elem().elem(0),
				res2.elem().elem(0));
  
  _inline_mult_adj_su3_mat_vec( colmat.elem().elem(),
				ferm.elem().elem(1),
				res2.elem().elem(1));

  _inline_mult_adj_su3_mat_vec( colmat.elem().elem(),
				ferm.elem().elem(2),
				res2.elem().elem(2));

  _inline_mult_adj_su3_mat_vec( colmat.elem().elem(),
				ferm.elem().elem(3),
				res2.elem().elem(3));
 
   Fermion diff_ferm = res - res2;
   QDPIO::cout << "Diff Ferm = " << diff_ferm << endl;

  
  H = adj(u)*spinProjectDir0Plus(V);
  H2= spinProjectDir0Plus(V);
  H3= adj(u)*H2;

  diff = H3 - H;
  QDPIO::cout << "AdjProj0+: || old - new || / || old || = " << sqrt(norm2(diff)/norm2(H)) << endl;
  
  H = adj(u)*spinProjectDir0Minus(V);
  H2= spinProjectDir0Minus(V);
  H3= adj(u)*H2;

  diff = H3 - H;
  QDPIO::cout << "AdjProj0-: || old - new || / || old || = " << sqrt(norm2(diff)/norm2(H)) << endl;

  H = adj(u)*spinProjectDir1Plus(V);
  H2= spinProjectDir1Plus(V);
  H3= adj(u)*H2;

  diff = H3 - H;
  QDPIO::cout << "AdjProj1+: || old - new || / || old || = " << sqrt(norm2(diff)/norm2(H)) << endl;
  
  H = adj(u)*spinProjectDir1Minus(V);
  H2= spinProjectDir1Minus(V);
  H3= adj(u)*H2;

  diff = H3 - H;
  QDPIO::cout << "AdjProj1-: || old - new || / || old || = " << sqrt(norm2(diff)/norm2(H)) << endl;

  H = adj(u)*spinProjectDir2Plus(V);
  H2= spinProjectDir2Plus(V);
  H3= adj(u)*H2;

  diff = H3 - H;
  QDPIO::cout << "AdjProj2+: || old - new || / || old || = " << sqrt(norm2(diff)/norm2(H)) << endl;
  
  H = adj(u)*spinProjectDir2Minus(V);
  H2= spinProjectDir2Minus(V);
  H3= adj(u)*H2;

  diff = H3 - H;
  QDPIO::cout << "AdjProj2-: || old - new || / || old || = " << sqrt(norm2(diff)/norm2(H)) << endl;

  H = adj(u)*spinProjectDir3Plus(V);
  H2= spinProjectDir3Plus(V);
  H3= adj(u)*H2;

  diff = H3 - H;
  QDPIO::cout << "AdjProj3+: || old - new || / || old || = " << sqrt(norm2(diff)/norm2(H)) << endl;
  
  H = adj(u)*spinProjectDir3Minus(V);
  H2= spinProjectDir3Minus(V);
  H3= adj(u)*H2;

  diff = H3 - H;
  QDPIO::cout << "AdjProj3-: || old - new || / || old || = " << sqrt(norm2(diff)/norm2(H)) << endl;


  gaussian(H);
  V = spinReconstructDir0Plus( u*H );
  H2 = u*H;
  V2 = spinReconstructDir0Plus( H2 );
  diff_v = V2 - V;
  QDPIO::cout << "ReconUPsiDir0+: || old - new || / || old || = " << sqrt(norm2(diff_v)/norm2(V2)) << endl;

  gaussian(H);
  V = spinReconstructDir0Minus( u*H );
  H2 = u*H;
  V2 = spinReconstructDir0Minus( H2 );
  diff_v = V2 - V;
  QDPIO::cout << "ReconUPsiDir0-: || old - new || / || old || = " << sqrt(norm2(diff_v)/norm2(V2)) << endl;

  gaussian(H);
  V = spinReconstructDir1Plus( u*H );
  H2 = u*H;
  V2 = spinReconstructDir1Plus( H2 );
  diff_v = V2 - V;
  QDPIO::cout << "ReconUPsiDir1+: || old - new || / || old || = " << sqrt(norm2(diff_v)/norm2(V2)) << endl;

  gaussian(H);
  V = spinReconstructDir1Minus( u*H );
  H2 = u*H;
  V2 = spinReconstructDir1Minus( H2 );
  diff_v = V2 - V;
  QDPIO::cout << "ReconUPsiDir1-: || old - new || / || old || = " << sqrt(norm2(diff_v)/norm2(V2)) << endl;

  gaussian(H);
  V = spinReconstructDir2Plus( u*H );
  H2 = u*H;
  V2 = spinReconstructDir2Plus( H2 );
  diff_v = V2 - V;
  QDPIO::cout << "ReconUPsiDir2+: || old - new || / || old || = " << sqrt(norm2(diff_v)/norm2(V2)) << endl;

  gaussian(H);
  V = spinReconstructDir2Minus( u*H );
  H2 = u*H;
  V2 = spinReconstructDir2Minus( H2 );
  diff_v = V2 - V;
  QDPIO::cout << "ReconUPsiDir2-: || old - new || / || old || = " << sqrt(norm2(diff_v)/norm2(V2)) << endl;

  gaussian(H);
  V = spinReconstructDir3Plus( u*H );
  H2 = u*H;
  V2 = spinReconstructDir3Plus( H2 );
  diff_v = V2 - V;
  QDPIO::cout << "ReconUPsiDir3+: || old - new || / || old || = " << sqrt(norm2(diff_v)/norm2(V2)) << endl;

  gaussian(H);
  V = spinReconstructDir3Minus( u*H );
  H2 = u*H;
  V2 = spinReconstructDir3Minus( H2 );
  diff_v = V2 - V;
  QDPIO::cout << "ReconUPsiDir3-: || old - new || / || old || = " << sqrt(norm2(diff_v)/norm2(V2)) << endl;




  gaussian(H);
  gaussian(V);
  V2 = V;

  V += spinReconstructDir0Plus( u*H );
  H2 = u*H;
  V2 += spinReconstructDir0Plus( H2 );
  diff_v = V2 - V;
  QDPIO::cout << "ReconUPsiDir0+=: || old - new || / || old || = " << sqrt(norm2(diff_v)/norm2(V2)) << endl;

  gaussian(H);
  gaussian(V);
  V2=V;

  V += spinReconstructDir0Minus( u*H );
  H2 = u*H;
  V2 += spinReconstructDir0Minus( H2 );
  diff_v = V2 - V;
  QDPIO::cout << "ReconUPsiDir0-=: || old - new || / || old || = " << sqrt(norm2(diff_v)/norm2(V2)) << endl;

  gaussian(H);
  gaussian(V);
  V2 = V;

  V += spinReconstructDir1Plus( u*H );
  H2 = u*H;
  V2 += spinReconstructDir1Plus( H2 );
  diff_v = V2 - V;
  QDPIO::cout << "ReconUPsiDir1+=: || old - new || / || old || = " << sqrt(norm2(diff_v)/norm2(V2)) << endl;

  gaussian(H);
  gaussian(V);
  V2=V;

  V += spinReconstructDir1Minus( u*H );
  H2 = u*H;
  V2 += spinReconstructDir1Minus( H2 );
  diff_v = V2 - V;
  QDPIO::cout << "ReconUPsiDir1-=: || old - new || / || old || = " << sqrt(norm2(diff_v)/norm2(V2)) << endl;

  gaussian(H);
  gaussian(V);
  V2 = V;

  V += spinReconstructDir2Plus( u*H );
  H2 = u*H;
  V2 += spinReconstructDir2Plus( H2 );
  diff_v = V2 - V;
  QDPIO::cout << "ReconUPsiDir2+=: || old - new || / || old || = " << sqrt(norm2(diff_v)/norm2(V2)) << endl;

  gaussian(H);
  gaussian(V);
  V2=V;

  V += spinReconstructDir2Minus( u*H );
  H2 = u*H;
  V2 += spinReconstructDir2Minus( H2 );
  diff_v = V2 - V;
  QDPIO::cout << "ReconUPsiDir2-=: || old - new || / || old || = " << sqrt(norm2(diff_v)/norm2(V2)) << endl;

  gaussian(H);
  gaussian(V);
  V2=V;

  V += spinReconstructDir3Plus( u*H );
  H2 = u*H;
  V2 += spinReconstructDir3Plus( H2 );
  diff_v = V2 - V;
  QDPIO::cout << "ReconUPsiDir3+=: || old - new || / || old || = " << sqrt(norm2(diff_v)/norm2(V2)) << endl;

  gaussian(H);
  gaussian(V);
  V2=V;

  V += spinReconstructDir3Minus( u*H );
  H2 = u*H;
  V2 += spinReconstructDir3Minus( H2 );
  diff_v = V2 - V;
  QDPIO::cout << "ReconUPsiDir3-=: || old - new || / || old || = " << sqrt(norm2(diff_v)/norm2(V2)) << endl;



  // Time to bolt
  QDP_finalize();

  exit(0);
}
  
   
