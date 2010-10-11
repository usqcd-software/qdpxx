// -*- C++ -*-

/*! \file
 * \brief Yet another random number generator
 *
 * This is an RNG independent from the vanilla QDP RNG.
 *
 * Seeds are four 12 bit integers
 */

#ifndef QDP_RANNYU_H
#define QDP_RANNYU_H

#include "qdp.h"

namespace QDP
{
  namespace RANNYU
  {
    //! The RNG
    double random();

    //! Seed has been set by default - this allows one to override it
    void setrn(const multi1d<int>& iseed);

    //! Recover the seed
    multi1d<int> savern();

  } // namespace RANNYU

} // namespace QDP

#endif
