//  $Id: mesons_w.cc,v 1.1 2002-09-12 18:22:17 edwards Exp $

/*! This routine is specific to Wilson fermions!
 *
 * Construct meson propagators
 * The two propagators can be identical or different.
 *
 * quark_prop_1 -- first quark propagator ( Read )
 * quark_prop_2 -- second (anti-) quark propagator ( Read )
 * meson_propagator -- Ns^2 mesons ( Modify )
 * meson_dispersion -- num_mom * Ns^2 mesons at nonzero momenta ( Modify )
 * num_mom -- number of different momenta ( Read )
 * t_source -- cartesian coordinates of the source ( Read )
 * j_decay -- direction of the exponential decay ( Read )
 *
 *        ____
 *        \
 * m(t) =  >  < m(t_source, 0) m(t + t_source, x) >
 *        /
 *        ----
 *          x
 *
 * NOTE: commented code is let over from the version that computes 2-pt
 *  functions at non-zero momenta. There are still some parameters listed
 *  above that are also left over
 */

#include "tests.h"

using namespace QDP;


void mesons(const LatticePropagator& quark_prop_1, const LatticePropagator& quark_prop_2, 
	    multi2d<Real>& meson_propagator, 
	    int num_mom, const multi1d<int>& t_source, 
	    int j_decay)

// multi3d<Real>& meson_dispersion,
{
  int length = layout.LattSize()[j_decay];

//  multi2d<Complex> disp(num_mom,length);
  multi1d<Double> hsum(length);

  int t0 = t_source[j_decay];
  int G5 = Ns*Ns-1;

  // Initialize the propagator so that we just add to it below
  meson_propagator = 0.0;

  // Contruct the antiquark prop
  LatticePropagator anti_quark_prop =  Gamma(G5) * quark_prop_2 * Gamma(G5);

  for(int n = 0; n < (Ns*Ns); ++n)
  {
    LatticeReal psi_sq;

    psi_sq = real(trace(conj(anti_quark_prop) * Gamma(n) * quark_prop_1 * Gamma(n)));

    // Do a slice-wise sum.
    hsum = slice_sum(psi_sq, j_decay);

//    if ( num_mom != 0 )
//    {
//      LatticeComplex corr_fn =  cmplx(psi_sq, 0.0);
//      sftmom(corr_fn, disp, FftP, num_mom, j_decay);
//    }

    for(int t = 0; t < length; ++t)
    {
      int t_eff = (t - t0 + length) % length;

      meson_propagator[n][t_eff] += Real(hsum[t]);

//      for(int m = 0; m < num_mom; ++m)
//	meson_dispersion[t_eff][m][n] += real(disp[t][m]);
    }
  }
}
