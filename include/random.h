// -*- C++ -*-
// $Id: random.h,v 1.3 2002-10-09 16:54:08 edwards Exp $
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
  void InitDefaultRNG(void);

  //! Initialize the RNG
  /*!
   * Seeds are big-ints
   */
  void setrn(const Seed& lseed);

  //! Recover the current seed
  /*!
   * Seeds are big-ints
   */
  void savern(Seed& lseed);


  //! Internal seed multiplier
  float sranf(Seed& seed, Seed&, const Seed&);
};

QDP_END_NAMESPACE();
