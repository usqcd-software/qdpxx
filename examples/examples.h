// -*- C++ -*-
// $Id: examples.h,v 1.1 2002-12-18 21:31:46 edwards Exp $
//
// Include file for test suite

#include "qdp.h"

using namespace QDP;

#define START_CODE(a)
#define END_CODE(a)

enum Reunitarize {REUNITARIZE, REUNITARIZE_ERROR, REUNITARIZE_LABEL};


void reunit(LatticeColorMatrix& xa);
void reunit(LatticeColorMatrix& xa, LatticeBoolean& bad, int& numbad, enum Reunitarize ruflag);


void junk(NmlWriter&, LatticeColorMatrix& b3, const LatticeColorMatrix& b1, const LatticeColorMatrix& b2, const Subset& s);
void MesPlq(const multi1d<LatticeColorMatrix>& u, Double& w_plaq, Double& s_plaq, 
	    Double& t_plaq, Double& link);
void mesons(const LatticePropagator& quark_prop_1, const LatticePropagator& quark_prop_2, 
	    multi2d<Real>& meson_propagator, 
	    const multi1d<int>& t_source, int j_decay);
void baryon(LatticePropagator& quark_propagator, 
	    multi2d<Complex>& barprop, 
	    const multi1d<int>& t_source, int j_decay, int bc_spec);
void dslash_2d_plus(LatticeFermion& chi, const multi1d<LatticeColorMatrix>& u, const LatticeFermion& psi,
	    int cb);
void dslash(LatticeFermion& chi, const multi1d<LatticeColorMatrix>& u, const LatticeFermion& psi,
	    int isign, int cb);

void FormFac(const multi1d<LatticeColorMatrix>& u, const LatticePropagator& quark_propagator,
	     const LatticePropagator& seq_quark_prop, const multi1d<int>& t_source, 
	     int t_sink, int j_decay, NmlWriter& nml);
