// -*- C++ -*-
// $Id: random.h,v 1.2 2002-09-23 18:19:25 edwards Exp $
//
// QDP data parallel interface
//
// Random number support

QDP_BEGIN_NAMESPACE(QDP);

//! Random number generator namespace
/*! A collection of routines and data for supporting random numbers */
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
