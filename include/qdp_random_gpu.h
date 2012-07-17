// -*- C++ -*-
//
// QDP data parallel interface
//
// Random number support

#ifndef QDP_RANDOM_H
#define QDP_RANDOM_H

namespace QDP {


//! Random number generator namespace
/*!
 * A collection of routines and data for supporting random numbers
 * 
 * It is a linear congruential with modulus m = 2**47, increment c = 0,
 * and multiplier a = (2**36)*m3 + (2**24)*m2 + (2**12)*m1 + m0.  
 */

namespace RNG
{


  template <class T>
  __device__ inline float sranf( PScalar<PSeed<RScalar<T> > >& seed, 
				 PScalar<PSeed<RScalar<T> > >& skewed_seed, 
				 const PScalar<PSeed<RScalar<T> > >& seed_mult)
  {
    /* Calculate the random number and update the seed according to the
     * following algorithm
     *
     * FILL(twom11,TWOM11);
     * FILL(twom12,TWOM12);
     * i3 = ran_seed(3)*ran_mult(0) + ran_seed(2)*ran_mult(1)
     *    + ran_seed(1)*ran_mult(2) + ran_seed(0)*ran_mult(3);
     * i2 = ran_seed(2)*ran_mult(0) + ran_seed(1)*ran_mult(1)
     *    + ran_seed(0)*ran_mult(2);
     * i1 = ran_seed(1)*ran_mult(0) + ran_seed(0)*ran_mult(1);
     * i0 = ran_seed(0)*ran_mult(0);
     *
     * ran_seed(0) = mod(i0, 4096);
     * i1          = i1 + i0/4096;
     * ran_seed(1) = mod(i1, 4096);
     * i2          = i2 + i1/4096;
     * ran_seed(2) = mod(i2, 4096);
     * ran_seed(3) = mod(i3 + i2/4096, 2048);
     *
     * sranf = twom11*(TO_REAL32(VALUE(ran_seed(3)))
     *       + twom12*(TO_REAL32(VALUE(ran_seed(2)))
     *       + twom12*(TO_REAL32(VALUE(ran_seed(1)))
     *       + twom12*(TO_REAL32(VALUE(ran_seed(0)))))));
     */

    PScalar< PScalar< RScalar<REAL> > > _sranf;
    float _ssranf;
    PScalar<PSeed<RScalar<T> > > ran_tmp;

    _sranf = seedToFloat(skewed_seed);
    cast_rep(_ssranf, _sranf);

    ran_tmp = seed * seed_mult;
    seed.elem() = ran_tmp.elem();

    ran_tmp.elem() = skewed_seed.elem() * seed_mult.elem();
    skewed_seed.elem() = ran_tmp.elem();

    return _ssranf;
  }


}


//! dest  = random
template<class T1, class T2>
PETE_DEVICE inline void
fill_random(float& d, T1& seed, T2& skewed_seed, const T1& seed_mult)
{
  d = float(RNG::sranf(seed, skewed_seed, seed_mult));
}


//! dest  = random
template<class T1, class T2>
PETE_DEVICE inline void
fill_random(double& d, T1& seed, T2& skewed_seed, const T1& seed_mult)
{
  d = double(RNG::sranf(seed, skewed_seed, seed_mult));
}


//! dest  = random
template<class T1, class T2, int N>
PETE_DEVICE inline void
fill_random(float* d, T1& seed, T2& skewed_seed, const T1& seed_mult)
{
  RNG::sranf(d, N, seed, skewed_seed, seed_mult);
}



} // namespace QDP

#endif
