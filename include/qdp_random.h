// -*- C++ -*-
// $Id: qdp_random.h,v 1.4 2003-08-31 21:10:00 edwards Exp $
//
// QDP data parallel interface
//
// Random number support

QDP_BEGIN_NAMESPACE(QDP);

//! Random number generator namespace
/*!
 * A collection of routines and data for supporting random numbers
 * 
 * It is a linear congruential with modulus m = 2**47, increment c = 0,
 * and multiplier a = (2**36)*m3 + (2**24)*m2 + (2**12)*m1 + m0.  
 */

namespace RNG
{
  //! Default initialization of the RNG
  /*! Uses arbitrary internal seed to initialize the RNG */
  void initDefaultRNG(void);

  //! Initialize the internals of the RNG
  void initRNG(void);

  //! Initialize the RNG seed
  /*!
   * Seeds are big-ints
   */
  void setrn(const Seed& lseed);

  //! Recover the current RNG seed
  /*!
   * Seeds are big-ints
   */
  void savern(Seed& lseed);


  //! Internal seed multiplier
  float sranf(Seed& seed, Seed&, const Seed&);

  //! Internal seed multiplier
  void sranf(float* d, int N, Seed& seed, ILatticeSeed&, const Seed&);
}

QDP_END_NAMESPACE();
