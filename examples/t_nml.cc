// -*- C++ -*-
//
// $Id: t_nml.cc,v 1.4 2003-04-27 02:50:22 edwards Exp $
//
/*! \file
 *  \brief Silly little internal test code
 */


#include "qdp.h"
#include "qdp_util.h"

//using namespace std;

//using namespace QDP;

int
main(int argc, char *argv[])
{
  QDP_initialize(&argc, &argv);

  NmlWriter nml_out("t_nml.output");
  push(nml_out, "Test");

  NmlReader nml_in("t_nml.input");
  push(nml_in, "IO_version");

  int version;
  Read(nml_in, version);
  Write(nml_out, version);
  pop(nml_in);

  push(nml_in, "param");
  int FermTypeP;
  Read(nml_in,FermTypeP);
  Write(nml_out,FermTypeP);

  int numKappa;
  Read(nml_in,numKappa);
  Write(nml_out,numKappa);

  multi1d<Real> Kappa(numKappa);
  Read(nml_in,Kappa);
  Write(nml_out,Kappa);

  int numSeq_src;
  Read(nml_in,numSeq_src);
  Write(nml_out,numSeq_src);

  multi1d<int> Seq_src(numSeq_src);
  Read(nml_in,Seq_src);
  Write(nml_out,Seq_src);

  bool my_bool;
  Read(nml_in,my_bool);
  Write(nml_out,my_bool);

  Boolean my_bool2;
  Read(nml_in,my_bool2);
  Write(nml_out,my_bool2);

  Seed seed;
  Read(nml_in,seed);
  Write(nml_out,seed);

  Complex my_complex;
  Read(nml_in,my_complex);
  Write(nml_out,my_complex);

  pop(nml_in);

  push(nml_in, "Cfg");

  string cfg_file;
  Read(nml_in,cfg_file);
  Write(nml_out,cfg_file);

  pop(nml_in);
  nml_in.close();

  pop(nml_out);
  nml_out.close();

  QDP_finalize();
  return 0;
}
