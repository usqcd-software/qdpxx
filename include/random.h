// -*- C++ -*-
// $Id: random.h,v 1.1 2002-09-12 18:22:16 edwards Exp $
//
// QDP data parallel interface
//
// Random number support

QDP_BEGIN_NAMESPACE(QDP);

//! Random number generator namespace
/*! A collection of routines and data for supporting random numbers */
namespace RNG
{
  //! Initialize the RNG
  /*!
     * There are 2 ways to set the seed, either by a 4 long int array of ints
     * or by a Seed object
     */
  void setrn(const Seed& lseed);

    //! Recover the current seed
    /*!
     * There are 2 ways to recover the seed, either by a 4 long int array of ints
     * or by a Seed object
     */
  void savern(Seed& lseed);


  //! Internal seed multiplier
  float sranf(Seed& seed, Seed&, const Seed&);
};

QDP_END_NAMESPACE();
