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

#include <vector>

namespace QDP
{
  namespace RANNYU
  {
    //! The RNG
    double random();

    //! Seed has been set by default - this allows one to override it
    void setrn(const std::vector<int>& iseed);

    //! Recover the seed
    std::vector<int> savern();

  } // namespace RANNYU

} // namespace QDP

#endif
