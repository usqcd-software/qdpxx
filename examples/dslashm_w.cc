// $Id: dslashm_w.cc,v 1.5 2002-09-26 20:02:36 edwards Exp $

/*! 
 * DSLASH
 *
 * This routine is specific to Wilson fermions!
 *
 * Description:
 *
 * This routine applies the operator D' to Psi, putting the result in Chi.
 *
 *	       Nd-1
 *	       ---
 *	       \
 *   chi(x)  :=  >  U  (x) (1 - isign gamma  ) psi(x+mu)
 *	       /    mu			  mu
 *	       ---
 *	       mu=0
 *
 *	             Nd-1
 *	             ---
 *	             \    +
 *                +    >  U  (x-mu) (1 + isign gamma  ) psi(x-mu)
 *	             /    mu			   mu
 *	             ---
 *	             mu=0
 *
 * Arguments:
 *
 *  Chi	      Pseudofermion field				(Write)
 *  U	      Gauge field					(Read)
 *  Psi	      Pseudofermion field				(Read)
 *		      +
 *  ISign      D' or D'  ( +1 | -1 ) respectively		(Read)
 *  CB	      Checkerboard of output vector			(Read) 
 */

#include "tests.h"

using namespace QDP;

void dslash_2d_plus(LatticeFermion& chi, const multi1d<LatticeGauge>& u, const LatticeFermion& psi,
		    int cb)
{
  Context s(rb[cb]);

  // NOTE: this is unrolled for 2 dimensions. Tests or some preproc hooks needed
  // for other Nd. Also computes only Dslash and not Dslash_dag
  
  /*     F 
   *   a2  (x)  :=  U  (x) (1 - isign gamma  ) psi(x)
   *     mu          mu                    mu
   */
  /*     B           +
   *   a2  (x)  :=  U  (x-mu) (1 + isign gamma  ) psi(x-mu)
   *     mu          mu                       mu
   */
  // Recontruct the bottom two spinor components from the top two
  /*                        F           B
   *   chi(x) :=  sum_mu  a2  (x)  +  a2  (x)
   *                        mu          mu
   */
  chi = spinReconstructDir0Minus(u[0] * shift(spinProjectDir0Minus(psi), FORWARD, 0))
      + spinReconstructDir0Plus(shift(u[0] * spinProjectDir0Plus(psi), BACKWARD, 0))
      + spinReconstructDir1Minus(u[1] * shift(spinProjectDir1Minus(psi), FORWARD, 1))
      + spinReconstructDir1Plus(shift(u[1] * spinProjectDir1Plus(psi), BACKWARD, 1));
}




//! General dslash call
void dslash(LatticeFermion& chi, const multi1d<LatticeGauge>& u, const LatticeFermion& psi,
	    int isign, int cb)
{
  Context s(rb[cb]);

  /*     F 
   *   a2  (x)  :=  U  (x) (1 - isign gamma  ) psi(x)
   *     mu          mu                    mu
   */
  /*     B           +
   *   a2  (x)  :=  U  (x-mu) (1 + isign gamma  ) psi(x-mu)
   *     mu          mu                       mu
   */
  // Recontruct the bottom two spinor components from the top two
  /*                        F           B
   *   chi(x) :=  sum_mu  a2  (x)  +  a2  (x)
   *                        mu          mu
   */
#if 1
  // NOTE: this is unrolled for 2 dimensions tests or some preproc hooks needed
  // for other Nd
  if (isign > 0)
  {
    chi = spinReconstructDir0Minus(u[0] * shift(spinProjectDir0Minus(psi), FORWARD, 0))
        + spinReconstructDir0Plus(shift(u[0] * spinProjectDir0Plus(psi), BACKWARD, 0))
        + spinReconstructDir1Minus(u[1] * shift(spinProjectDir1Minus(psi), FORWARD, 1))
        + spinReconstructDir1Plus(shift(u[1] * spinProjectDir1Plus(psi), BACKWARD, 1));
  }
  else
  {
    chi = spinReconstructDir0Plus(u[0] * shift(spinProjectDir0Plus(psi), FORWARD, 0))
        + spinReconstructDir0Minus(shift(u[0] * spinProjectDir0Minus(psi), BACKWARD, 0))
        + spinReconstructDir1Plus(u[1] * shift(spinProjectDir1Plus(psi), FORWARD, 1))
        + spinReconstructDir1Minus(shift(u[1] * spinProjectDir1Minus(psi), BACKWARD, 1));
  }
#else

  // NOTE: the loop is not unrolled - it should be all in a single line for
  // optimal performance
  zero(chi);

  // NOTE: temporarily has conversion call of LatticeHalfFermion - will be removed
  for(int mu = 0; mu < Nd; ++mu)
  {
    chi += spinReconstruct(LatticeHalfFermion(u[mu] * shift(spinProject(psi,mu,-isign), FORWARD, mu)),mu,-isign)
         + spinReconstruct(LatticeHalfFermion(shift(u[mu] * spinProject(psi,mu,+isign), BACKWARD, mu)),mu,+isign);
  }
#endif

}


