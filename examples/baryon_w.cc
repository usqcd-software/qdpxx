// $Id: baryon_w.cc,v 1.1 2002-09-12 18:22:17 edwards Exp $ 

/*!
 * This routine is specific to Wilson fermions! 
 *
 * Construct baryon propagators for the Proton and the Delta^+ with
 * degenerate "u" and "d" quarks, as well as the Lambda for, in
 * addition, a degenerate "s" quark. For these degenerate quarks, the
 * Lambda is degenerate with the Proton, but we keep it for compatibility
 * with the sister routine that treats non-degenerate quarks.

 * The routine also computes time-charge reversed baryons and adds them
 * in for increased statistics.

 * quark_propagator -- quark propagator ( Read )
 * barprop -- baryon propagator ( Modify )
 * bardisp -- baryon props. at non-zero momenta ( Modify )
 * num_mom -- number of non-zero momenta ( Read )
 * t_source -- cartesian coordinates of the source ( Read )
 * j_decay -- direction of the exponential decay ( Read )
 * bc_spec  -- boundary condition for spectroscopy ( Read )
 * FftP    -- flag for use of fft or sft ( Read )

 *        ____
 *        \
 * b(t) =  >  < b(t_source, 0) b(t + t_source, x) >
 *        /                    
 *        ----
 *          x

 * For the Proton we take

 * |P_1, s_z=1/2> = (d C gamma_5 u) "u_up"

 * for the Lambda

 * |L_1, s_z=1/2> = 2*(u C gamma_5 d) "s_up" + (s C gamma_5 d) "u_up"
 *                  + (u C gamma_5 s) "d_up"

 * and for the Delta^+

 * |D_1, s_z=3/2> = 2*(d C gamma_- u) "u_up" + (u C gamma_- u) "d_up".

 * We have put "q_up" in quotes, since this is meant in the Dirac basis,
 * not in the 'DeGrand-Rossi' chiral basis used in the program!

 * For all baryons we compute a 'B_2' that differs from the 'B_1' above
 * by insertion of a gamma_4 between C and the gamma_{5,-}.
 * And finally, we also compute the non-relativistic baryons, 'B_3',
 * which up to a factor 1/2 are just the difference B_1 - B_2, as can
 * be seen by projecting to the "upper" components in the Dirac basis,
 * achieved by (1 + gamma_4)/2 q, for quark q.

 * The Proton_k is baryon 3*(k-1), the Lambda_k is baryon 3*(k-1)+1
 * and the Delta^+_k is baryon 3*(k-1)+2. 
 */

#include "tests.h"

using namespace QDP;

void baryon(LatticePropagator& quark_propagator, 
	    multi2d<Complex>& barprop, 
	    const multi1d<int>& t_source, int j_decay, int bc_spec)

// int num_mom
// multi3d& bardisp
// COMPLEX(barprop, length, 9);
// COMPLEX(bardisp, length, num_mom, 9);
{
  
  int length = layout.LattSize()[j_decay];

//   Complex disp(num_mom, length);
  
  if ( Ns != 4 || Nc != 3 )		/* Code is specific to Ns=4 and Nc=3. */
    return;

  int t0 = t_source[j_decay];
  
  SpinMatrix Cgm;
  SpinMatrix Cg4m;
  SpinMatrix CgmNR;

  SpinMatrix g_one;
  SpinMatrix g_tmp1;
  SpinMatrix g_tmp2;

  g_one = 1.0;

  /* C = Gamma(10) */
  g_tmp1 = Gamma(1) * g_one;
  g_tmp2 = Gamma(2) * g_one  +  multiplyI(g_tmp1);
  g_tmp1 = g_tmp2 * 0.5;
  Cgm = Gamma(10) * g_tmp1;

  g_tmp2 = Gamma(8) * g_tmp1;
  Cg4m = Gamma(10) * g_tmp2;
  CgmNR =  Cgm;
  CgmNR -=  Cg4m;

  SpinMatrix S_proj = 
    0.5*((g_one + Gamma(8) * g_one) - multiplyI(Gamma(3) * g_one  +  Gamma(11) * g_one));

  LatticePropagator q1_tmp;
  LatticePropagator q2_tmp;
  LatticeComplex b_prop;
  LatticeComplex b_prop_l;
  multi1d<DComplex> hsum(length);

  /*Loop over time-charge reversals */
  for(int time_rev = 0; time_rev < 2; ++time_rev)
  {
    /* Loop over baryons */
    for(int baryons = 0; baryons < 9; ++baryons)
    {
      LatticePropagator di_quark;
      LatticeColorMatrix col_mat;
      LatticeSpinMatrix spin_mat;
      LatticeComplex b_corr;

      switch (baryons)
      {
      case 0:
        /* Proton_1; use also for Lambda_1! */
	/* C gamma_5 = Gamma(5) */
	q1_tmp = quark_propagator * Gamma(5);
	q2_tmp = Gamma(5) * quark_propagator;
	di_quark = quarkContract13(q1_tmp,  q2_tmp);
	col_mat = trace(di_quark);
	q1_tmp = quark_propagator * col_mat;
	spin_mat = trace(q1_tmp);
	b_prop = trace(S_proj *   spin_mat);

	spin_mat = trace(quark_propagator *   di_quark);
	b_corr = trace(S_proj *   spin_mat);
	b_prop +=  b_corr;
	b_prop_l =  b_prop;
	break;

      case 1:
        /* Lambda_1 = 3*Proton_1 (for compatibility with heavy-light routine) */
	b_prop *= 3.0;
	break;

      case 2:
        /* Delta^+_1 */
	q1_tmp = quark_propagator * Cgm;
	q2_tmp = Cgm * quark_propagator;
	di_quark = quarkContract13(q1_tmp,  q2_tmp);
	col_mat = trace(di_quark);
	q1_tmp = quark_propagator * col_mat;
	spin_mat = trace(q1_tmp);
	b_prop = trace(S_proj *   spin_mat);

	spin_mat = trace(quark_propagator *   di_quark);
	b_corr = trace(S_proj *   spin_mat);
	b_prop +=  b_corr * 2.0;

	/* Multiply by 3 for compatibility with heavy-light routine */
	b_prop *= 3.0;
	break;

      case 3:
        /* Proton_2; use also for Lambda_2! */
	/* C gamma_5 gamma_4 = - Gamma(13) */
	q1_tmp = quark_propagator * Gamma(13);
	q2_tmp = Gamma(13) * quark_propagator;
	di_quark = quarkContract13(q1_tmp,  q2_tmp);
	col_mat = trace(di_quark);
	q1_tmp = quark_propagator * col_mat;
	spin_mat = trace(q1_tmp);
	b_prop = trace(S_proj *   spin_mat);

	spin_mat = trace(quark_propagator *   di_quark);
	b_corr = trace(S_proj *   spin_mat);
	b_prop +=  b_corr;
	b_prop_l =  b_prop;
	break;

      case 4:
        /* Lambda_2 = 3*Proton_2 (for compatibility with heavy-light routine) */
	b_prop *= 3.0;
	break;

      case 5:
        /* Sigma^{*+}_2 */
	q1_tmp = quark_propagator * Cg4m;
	q2_tmp = Cg4m * quark_propagator;
	di_quark = quarkContract13(q1_tmp,  q2_tmp);
	col_mat = trace(di_quark);
	q1_tmp = quark_propagator * col_mat;
	spin_mat = trace(q1_tmp);
	b_prop = trace(S_proj *   spin_mat);

	spin_mat = trace(quark_propagator *   di_quark);
	b_corr = trace(S_proj *   spin_mat);
	b_prop +=  b_corr * 2.0;

	/* Multiply by 3 for compatibility with heavy-light routine */
	b_prop *= 3.0;
	break;

      case 6:
        /* Proton^+_3; use also for Lambda_3! */
	/* C gamma_5 - C gamma_5 gamma_4 = Gamma(5) + Gamma(13) */
	q1_tmp = quark_propagator * Gamma(5);
	q1_tmp +=  quark_propagator *   Gamma(13);
	q2_tmp = Gamma(5) * quark_propagator;
	q2_tmp +=  Gamma(13) *   quark_propagator;
	di_quark = quarkContract13(q1_tmp,  q2_tmp);
	col_mat = trace(di_quark);
	q1_tmp = quark_propagator * col_mat;
	spin_mat = trace(q1_tmp);
	b_prop = trace(S_proj *   spin_mat);

	spin_mat = trace(quark_propagator *   di_quark);
	b_corr = trace(S_proj *   spin_mat);
	b_prop +=  b_corr;
	b_prop_l =  b_prop;
	break;

      case 7:
        /* Lambda_3 = 3*Proton_3 (for compatibility with heavy-light routine) */
	b_prop *= 3.0;
	break;

      case 8:
        /* Sigma^{*+}_3 */
	q1_tmp = quark_propagator * CgmNR;
	q2_tmp = CgmNR * quark_propagator;
	di_quark = quarkContract13(q1_tmp,  q2_tmp);
	col_mat = trace(di_quark);
	q1_tmp = quark_propagator * col_mat;
	spin_mat = trace(q1_tmp);
	b_prop = trace(S_proj *   spin_mat);

	spin_mat = trace(quark_propagator *   di_quark);
	b_corr = trace(S_proj *   spin_mat);
	b_prop +=  b_corr * 2.0;

	/* Multiply by 3 for compatibility with heavy-light routine */
	b_prop *= 3.0;
	break;

      default:
//	SZ_ERROR("Unknown baryon",baryons);
	SZ_ERROR("Unknown baryon");
      }

      /* Project on zero momentum: Do a slice-wise sum. */
      hsum = slice_sum(b_prop, j_decay);

      switch (time_rev)
      {
      case 0:
        /* forward */
        for(int t = 0; t < length; ++t)
        {
          int t_eff = (t - t0 + length) % length;

          if ( bc_spec < 0 && (t_eff+t0) >= length)
          {
            barprop(baryons,t_eff) = - 0.5 * Complex(hsum[t]);
          }
          else
            barprop(baryons,t_eff) = 0.5 * Complex(hsum[t]);
        }
	break;

      case 1:
        /* backward */
        for(int t = 0; t < length; ++t)
        {
          int t_eff = (length - t + t0) % length;

          if ( bc_spec < 0 && (t_eff-t0) > 0)
          {
            barprop(baryons,t_eff) -=  0.5 * Complex(hsum[t]);
          }
          else
            barprop(baryons,t_eff) +=  0.5 * Complex(hsum[t]);
        }
      }

#if 0
      /* Project onto non-zero momentum if desired */
      if ( num_mom != 0 )
      {
	CALL(sftmom, b_prop, disp, FftP, num_mom, j_decay);

	for(int m = 0; m < num_mom; ++m)
	{
	  switch (time_rev)
	  {
          case 0:
            /* forward */
            for(int t = 0; t < length; ++t)
            {
              int t_eff = (t - t0 + length) % length;
              if ( bc_spec < 0 && (t_eff+t0) >= length)
              {
                bardisp[baryons][m][t_eff], disp(m = -t) *  0.5;
              }
              else
                bardisp[baryons][m][t_eff], disp(m = t) *  0.5;
            }
	    break;

          case 1:
            /* backward */
            for(int t = 0; t < length; ++t)
            {
              int t_eff = (length - t + t0) % length;
              if ( bc_spec < 0 && (t_eff-t0) > 0)
              {
                bardisp[baryons][m][t_eff] -=  disp[m][t] *   0.5;
              }
              else
                bardisp[baryons][m][t_eff] +=  disp[m][t] *   0.5;
            }
	  }
	}
      }
#endif

    } /* end loop over baryons */

    /* Time-charge reverse the quark propagators */
    /* S_{CT} = gamma_5 gamma_4 = gamma_1 gamma_2 gamma_3 = Gamma(7) */
    q1_tmp = - (Gamma(7) *   quark_propagator);
    quark_propagator = q1_tmp * Gamma(7);
  }
}
