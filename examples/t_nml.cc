// -*- C++ -*-
//
// $Id: t_nml.cc,v 1.8 2004-02-11 10:33:09 bjoo Exp $
//
/*! \file
 *  \brief Silly little internal test code
 */


#include "qdp.h"
#include "qdp_util.h"

//using namespace std;

using namespace QDP;

int
main(int argc, char *argv[])
{
  QDP_initialize(&argc, &argv);

  NmlWriter nml_out("t_nml.output");
  push(nml_out, "Test");

  NmlReader nml_in("t_nml.input");
  push(nml_in, "IO_version");

  int version;
  read(nml_in,"version", version);
  write(nml_out, "version", version);
  pop(nml_in);

  push(nml_in, "param");
  int FermTypeP;
  read(nml_in,"FermTypeP", FermTypeP);
  write(nml_out,"FermTypeP", FermTypeP);

  int numKappa;
  read(nml_in,"numKappa", numKappa);
  write(nml_out,"numKappa", numKappa);

  multi1d<Real> Kappa(numKappa);
  read(nml_in,"Kappa",Kappa);
  write(nml_out,"Kappa", Kappa);

  int numSeq_src;
  read(nml_in,"numSeq_src",numSeq_src);
  write(nml_out,"numSeq_src", numSeq_src);

  multi1d<int> Seq_src(numSeq_src);
  read(nml_in,"Seq_src",Seq_src);
  write(nml_out,"Seq_src", Seq_src);

  bool my_bool;
  read(nml_in,"my_bool", my_bool);
  write(nml_out,"my_bool", my_bool);

  Boolean my_bool2;
  read(nml_in,"my_bool2",my_bool2);
  write(nml_out,"mu_bool2", my_bool2);

  Seed seed;
  read(nml_in,"seed",seed);
  write(nml_out,"seed", seed);

  Complex my_complex;
  read(nml_in,"my_complex", my_complex);
  write(nml_out,"my_complex",my_complex);

  pop(nml_in);

  push(nml_in, "Cfg");

  string cfg_file;
  read(nml_in,"cfg_file", cfg_file);
  write(nml_out,"cfg_file",cfg_file);

  write(nml_out,"my_string","hello world");

  pop(nml_in);
  nml_in.close();

  pop(nml_out);
  nml_out.close();

  QDP_finalize();
  exit(0);
}
