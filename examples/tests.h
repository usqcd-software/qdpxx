// -*- C++ -*-
// $Id: tests.h,v 1.5 2002-10-25 03:33:26 edwards Exp $
//
// Include file for test suite

#include "qdp.h"
#include "geom.h"

using namespace QDP;

void junk(NmlWriter&, LatticeGauge& b3, const LatticeGauge& b1, const LatticeGauge& b2, const Subset& s);
void MesPlq(const multi1d<LatticeGauge>& u, Double& w_plaq, Double& s_plaq, 
	    Double& t_plaq, Double& link);
void mesons(const LatticePropagator& quark_prop_1, const LatticePropagator& quark_prop_2, 
	    multi2d<Real>& meson_propagator, 
	    const multi1d<int>& t_source);
void baryon(LatticePropagator& quark_propagator, 
	    multi2d<Complex>& barprop, 
	    const multi1d<int>& t_source, int bc_spec);
void dslash_2d_plus(LatticeFermion& chi, const multi1d<LatticeGauge>& u, const LatticeFermion& psi,
	    int cb);
void dslash(LatticeFermion& chi, const multi1d<LatticeGauge>& u, const LatticeFermion& psi,
	    int isign, int cb);

void FormFac(const multi1d<LatticeColorMatrix>& u, const LatticePropagator& quark_propagator,
	     const LatticePropagator& seq_quark_prop, const multi1d<int>& t_source, 
	     int t_sink, NmlWriter& nml);
