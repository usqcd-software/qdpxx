// $Id: dslashm_w.cc,v 1.4 2002-09-24 03:12:13 edwards Exp $

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
 *  CB	      Checkerboard of input vector			(Read) 
 */

#include "tests.h"

using namespace QDP;

void dslash(LatticeFermion& chi, const multi1d<LatticeGauge>& u, const LatticeFermion& psi,
	    int isign, int cb)
{
  Context s(rb[1-cb]);

  /*     F 
   *   a2  (x)  :=  U  (x) (1 - isign gamma  ) psi(x)
   *     mu          mu                    mu
   */
  /*     B           +
   *   a2  (x)  :=  U  (x-mu) (1 + isign gamma  ) psi(x-mu)
   *     mu          mu                       mu
   */
  // Recontruct the bottom two spinor components from the top two
  // NOTE: the loop is not unrolled - it should be all in a single line for
  // optimal performance
  zero(chi);

  // NOTE: temporarily has conversion call of LatticeHalfFermion - will be removed
  for(int mu = 0; mu < Nd; ++mu)
  {
    chi += spinReconstruct(LatticeHalfFermion(u[mu] * shift(spinProject(psi,mu,-isign), FORWARD, mu)),mu,-isign)
         + spinReconstruct(LatticeHalfFermion(shift(u[mu] * spinProject(psi,mu,+isign), BACKWARD, mu)),mu,+isign);
  }

}
