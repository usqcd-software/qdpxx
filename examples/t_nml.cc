// -*- C++ -*-
//
// $Id: t_nml.cc,v 1.1 2003-04-24 05:31:27 edwards Exp $
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

  NmlReader nml_in("t_nml.input");
  NmlWriter nml_out("t_nml.output");

  push(nml_out, "Test");

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

  int numGamma;
  Read(nml_in,numGamma);
  Write(nml_out,numGamma);

  multi1d<int> Gamma_list(numGamma);
  Read(nml_in,Gamma_list);
  Write(nml_out,Gamma_list);

  pop(nml_in);
  pop(nml_out);

  nml_in.close();
  nml_out.close();

  QDP_finalize();

  return 0;
}
