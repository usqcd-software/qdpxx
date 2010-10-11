//! Yet another random number generator
/*!
 *      It is a linear congruential with modulus m = 2**48, increment c = 1,
 *      and multiplier a = (2**36)*m1 + (2**24)*m2 + (2**12)*m3 + m4. 
 *      The multiplier is stored in common (see subroutine setrn)
 *      and is set to a = 31167285 (recommended by Knuth, vol. 2,
 *      2nd ed., p. 102).
 *
 *     Multiplier is 31167285 = (2**24) + 3513*(2**12) + 821.
 *        Recommended by Knuth, vol. 2, 2nd ed., p. 102.
 *     (Generator is linear congruential with odd increment
 *        and maximal period, so seed is unrestricted: it can be
 *        either even or odd.)
 */

#include "qdp_rannyu.h"
#include "qdp.h"

namespace QDP
{
  namespace RANNYU
  {
    namespace
    {
      int m[4] = {0, 1, 3513, 821};
      int l[4] = {13, 11, 5, 1};
    }

    double random()
    {
      double twom12 = 1/4096.0;
      int i[4];

      i[0] = l[0]*m[3] + l[1]*m[2] + l[2]*m[1] + l[3]*m[0];
      i[1] = l[1]*m[3] + l[2]*m[2] + l[3]*m[1];
      i[2] = l[2]*m[3] + l[3]*m[2];
      i[3] = l[3]*m[3] + 1;
      l[3] = i[3] & 4095;
      i[2] = i[2] + (i[3] >> 12);
      l[2] = i[2] & 4095;
      i[1] = i[1] + (i[2] >> 12);
      l[1] = i[1] & 4095;
      l[0] = (i[0] + (i[1] >> 12)) >> 12;
      return twom12*((double)l[0] + twom12*((double)l[1] + twom12*((double)l[2] + twom12*((double)l[3]))));
    }

    //! Seed has been set by default - this allows one to override it
    void setrn(const multi1d<int>& iseed)
    {
      if (iseed.size() != 4)
      {
	QDPIO::cerr << __func__ << ": rannyu seed is not length 4\n";
	QDP_abort(1);
      }
	
      for(int i=0; i < 4; ++i)
	l[i] = iseed[i];
    }

    //! Recover the seed
    multi1d<int> savern()
    {
      multi1d<int> iseed(4);

      for(int i=0; i < 4; ++i)
	iseed[i] = l[i];

      return iseed;
    }

  } // namespace RANNYU

} // namespace QDP

