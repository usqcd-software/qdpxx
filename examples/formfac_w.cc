// $Id: formfac_w.cc,v 1.2 2002-09-14 20:13:24 edwards Exp $

/*! Compute contractions for current insertion 3-point functions.
 *
 * This routine is specific to Wilson fermions!
 *
 * u    -- gauge fields (used for non-local currents) ( Read )
 * quark_propagator -- quark propagator ( Read )
 * seq_quark_prop -- sequential quark propagator ( Read )
 * t_source -- cartesian coordinates of the source ( Read )
 * t_sink -- time coordinate of the sink ( Read )
 * j_decay -- direction of the exponential decay ( Read ) 
 */

#include "tests.h"
#include "proto.h"

using namespace QDP;

void FormFac(const multi1d<LatticeColorMatrix>& u, 
	     const LatticePropagator& quark_propagator,
	     const LatticePropagator& seq_quark_prop, 
	     const multi1d<int>& t_source, 
	     int t_sink, int j_decay)
{
  // Length of lattice in j_decay direction and 3pt correlations fcns
  int length = layout.LattSize()[j_decay];
  multi1d<Complex> local_cur3ptfn(length);
  multi1d<Complex> nonlocal_cur3ptfn(length);
  
//  START_CODE("formfac");

  int t0 = t_source[j_decay];
  int G5 = Ns*Ns-1;
  
  /*
   * Coordinates for insertion momenta
   */
  multi1d<LatticeInteger> my_coord(Nd);
  for(int mu = 0; mu < Nd; ++mu)
    my_coord[mu] = latticeCoordinate(mu);

  
  /*
   * Construct the anti-quark propagator from the seq. quark prop.
   */ 
  LatticePropagator anti_quark_prop = Gamma(G5) * seq_quark_prop * Gamma(G5);


  /*
   * Loop over mu of insertion current
   */
  for(int mu = 0; mu < Nd; ++mu)
  {
    /* 
     * The local non-conserved vector-current matrix element 
     */
    int n = 1 << mu;
    LatticeComplex corr_local_fn = trace(conj(anti_quark_prop) * Gamma(n) * quark_propagator);

    /* 
     * Construct the conserved non-local vector-current matrix element 
     * NOTE: this is only really conserved for the Wilson fermion quark action
     *
     * The form of J_mu = (1/2)*[psibar(x+mu)*U^dag_mu*(1+gamma_mu)*psi(x) -
     *                           psibar(x)*U_mu*(1-gamma_mu)*psi(x+mu)]
     * NOTE: the 1/2  is included down below in the slice_sum stuff
     */
    LatticeComplex corr_nonlocal_fn = 
      trace(conj(u[mu] * shift(anti_quark_prop, FORWARD, mu)) * 
	    (quark_propagator + Gamma(n) * quark_propagator));

    LatticePropagator tmp_prop1 = u[mu] * shift(quark_propagator, FORWARD, mu);
    corr_nonlocal_fn -= trace(conj(anti_quark_prop) * (tmp_prop1 - Gamma(n) * tmp_prop1));


    /*
     * Loop over non-zero insertion momenta
     * Do this by constructing a 5^(Nd-1) grid in momenta centered about the
     * origin. Loop lexicographically over all the "sites" (momenta value)
     * and toss out ones considered too large to give any reasonable statistics
     *
     * NOTE: spatial anisotropy is no allowed here
     */
    multi1d<int> mom_size(Nd-1);
    int Ndm1 = Nd-1;
    int L = 5;
    int mom_vol = 1;

    for(int nu=0; nu < Ndm1; ++nu)
    {
      mom_vol *= L;
      mom_size[nu] = L;
    }

    for(int n = 0; n < mom_vol; ++n)
    {
      multi1d<int> inser_mom = crtesn(n, mom_size);

      int q_sq = 0;
      for(int nu = 0; nu < Ndm1; ++nu)
      {
	inser_mom[nu] -= (L-1)/2;
	q_sq += inser_mom[nu]*inser_mom[nu];
      }

      // Arbitrarily set the cutoff on max allowed momentum to be [2,1,0]
      if (q_sq > 4) continue;

      LatticeReal p_dot_x(float(0.0));

      int j = 0;
      for(int nu = 0; nu < Nd; ++nu)
      {
	const Real twopi = 6.283185307179586476925286;

	if (nu == j_decay)
	  continue;

	p_dot_x += LatticeReal(my_coord[nu]) * twopi
	  * Real(inser_mom[j]) / layout.LattSize()[nu];
	j++;
      }

      // The complex phases  exp(i p.x )
      LatticeComplex  phasefac = cmplx(cos(p_dot_x), sin(p_dot_x));

      // Local current
      multi1d<DComplex> hsum(length);

      LatticeComplex corr_local_tmp = phasefac * corr_local_fn;
      hsum = slice_sum(corr_local_tmp, j_decay);

      for(int t = 0; t < length; ++t)
      {
	int t_eff = (t - t0 + length) % length;

	local_cur3ptfn[t_eff] = Complex(hsum[t]);
      }


      // Non-local current
      LatticeComplex corr_nonlocal_tmp = phasefac * corr_nonlocal_fn;
      hsum = slice_sum(corr_nonlocal_tmp, j_decay);

      for(int t = 0; t < length; ++t)
      {
	int t_eff = (t - t0 + length) % length;

	nonlocal_cur3ptfn[t_eff] = 0.5 * Complex(hsum[t]);
      }

      // Print out the results
      Push(cerr,"Wilson_Current_3Pt_fn");
      WRITE_NAMELIST(cerr,mu);
      WRITE_NAMELIST(cerr,j_decay);
      WRITE_NAMELIST(cerr,inser_mom);
      WRITE_NAMELIST(cerr,local_cur3ptfn);
      WRITE_NAMELIST(cerr,nonlocal_cur3ptfn);
      Pop(cerr);
    }
  }
                            
//  END_CODE();
}
