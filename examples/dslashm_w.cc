// $Id: dslashm_w.cc,v 1.1 2002-09-12 18:22:17 edwards Exp $

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

#define F_SWAP 1
#define B_SWAP 0

void dslash(LatticeFermion& chi, const multi1d<LatticeGauge>& u, const LatticeFermion& psi,
	    int isign, int cb)
{
  multi2d<LatticeHalfFermion> a1(2,Nd);
  multi2d<LatticeHalfFermion> a2(2,Nd);

  Context s(rb[1-cb]);

  for(int mu = 0; mu < Nd; ++mu)
  {
    Context s(rb[cb]);

    /*     F
     *   a1  (x) := (1 - isign gamma  ) psi(x)
     *     mu		       mu
     *
     *     B
     *   a1  (x) := (1 + isign gamma  ) psi(x)
     *     mu		       mu 
     */

    /* [Only the first two components of a1 are computed] */

    a1[F_SWAP][mu] = spinProject(psi,mu,-isign);
    a1[B_SWAP][mu] = spinProject(psi,mu,+isign);
  }

  for(int mu = 0; mu < Nd; ++mu)
  {	
    Context s(rb[1-cb]);

    /*     B           +          B             +
     *   a2  (x)  :=  U  (x-mu) a1  (x-mu)  =  U  (x-mu) (1 + isign gamma  ) psi(x-mu)
     *     mu          mu         mu            mu                       mu 
     */

    a2[B_SWAP][mu] = shift(LatticeHalfFermion(u[mu]*a1[B_SWAP][mu]), BACKWARD, mu);
  
    /*     F                    F
     *   a 2  (x)  :=  U  (x) a1  (x+mu)
     *     mu           mu      mu
     */

    a2[F_SWAP][mu] = u[mu] * shift(a1[F_SWAP][mu], FORWARD, mu);
  }
  

  /*  Recontruct the bottom two spinor components from the top two */
  chi  = spinReconstruct(a2[F_SWAP][0],0,-isign);
  chi += spinReconstruct(a2[B_SWAP][0],0,+isign);

  for(int mu = 1; mu < Nd; ++mu)
  {
    chi += spinReconstruct(a2[F_SWAP][mu],mu,-isign);
    chi += spinReconstruct(a2[B_SWAP][mu],mu,+isign);
  }
}
