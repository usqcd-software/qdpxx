/*
 *  $Id: t_spectrum.cc,v 1.13 2003-09-03 19:52:22 edwards Exp $
 *
 *  This is a test program for spectroscopy using qdp++
 *
 *  It is specific to 4 D
 */

#include <iostream>
#include <cstdio>

#include "qdp.h"
#include "examples.h"
#include "qdp_util.h"

using namespace QDP;



int main(int argc, char **argv)
{
  // Put the machine into a known state
  QDP_initialize(&argc, &argv);


  //  Begin by specifying the lattice sizes, of dimension Nd, defined in QDP

  multi1d<int> nsize(Nd);	// Note that the number of dimensions

  // defined in params.h

  //  For those new to C++, note that the vector class contains more 
  //  information than a simple array of elements, and in particular its size

  /*  
      for(int mu = 0; mu < nsize.size(); mu++){
      printf("Specify the size of dimension %d? ",mu);
      scanf("%d", &nsize[mu]);
      }
  */

  const int foo[] = {4, 4, 4, 8};
  nsize = foo;
  // Initialise the layout
  Layout::setLattSize(nsize);
  Layout::create();

  int j_decay = Nd-1;
  int length = Layout::lattSize()[j_decay]; // Define the temporal direction

  NmlWriter nml("t_spectrum.nml"); // Open file for sample output


  multi1d<LatticeColorMatrix> u(Nd);

#if 1

  /*
   *  Create a Random gauge field
   */

  for(int mu = 0; mu < u.size(); mu++)
    gaussian(u[mu]);
#endif

#if 0
  /*
   *  Read in a gauge field in the usual NERSC format
   *  GTF: moved filename decl into scope to avoid compiler warning when #if 0
   */

  {
    char filename[100];

    cerr << "Reading in NERSC file...";
    printf("Gauge file name? ");
    scanf("%s",filename);
    readArchiv(u, filename);
    cerr << "...done\n";

    cerr << "Gauge field is ";
    for(int mu = 0; mu < Nd; mu++){
      Write(nml, mu);
      Write(nml, u[mu]);
    }
  }
#endif

  /*
   *  Evaluate the plaquette on the gauge field
   */
 {
   Double w_plaq, s_plaq, t_plaq, link;
   cerr << "Evaluating the plaquette...";
   MesPlq(u, w_plaq, s_plaq, t_plaq, link);
   cerr << "...done\n";

   cerr << "w_plaq = " << w_plaq << endl;
   cerr << "link = " << link << endl;
 }





  /*
   *  Now the smeared lattice gauge field, which for the moment we will just
   *  set equal to the original gauge field
   */

  multi1d<LatticeColorMatrix> u_smr(Nd);

  for(int mu = 0; mu < u.size(); mu++)
    u_smr[mu] = u[mu];

  /*
   *  Read in two lattice propagators from disk
   */

  LatticePropagator quark_prop_1;
  gaussian(quark_prop_1);

  multi1d<int> t_source(Nd);	// Source coordinate of propagators
  t_source = 0;

  cerr << "Computing simple meson spectroscopy..." << endl;

  {

    multi2d<Real> meson_prop;
  
    mesons(quark_prop_1, quark_prop_1, meson_prop, t_source, j_decay);

    /*
     *  Print the results
     */

    push(nml,"Point_Point_Wilson_Mesons");
    Write(nml, j_decay);
    Write(nml, meson_prop);
    pop(nml);
  }

  cerr << "...done" << endl;

  cerr << "Computing simple baryon spectroscopy..." << endl;

  {

    multi2d<Complex> baryon_prop;
  
    baryon(quark_prop_1, baryon_prop, t_source, j_decay, 1);

    /*
     *  Print the results
     */

    push(nml,"Point_Point_Wilson_Baryons");
    Write(nml, j_decay);
    Write(nml, baryon_prop);
    pop(nml);
  }

  cerr << "...done" << endl;

  // Time to bolt
  QDP_finalize();

  exit(0);
}
