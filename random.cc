// $Id: random.cc,v 1.3 2002-10-02 20:29:37 edwards Exp $
//
// Random number generator support


#include "qdp.h"

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
  //! Global (current) seed
  Seed ran_seed;
  //! RNG multiplier
  Seed ran_mult;
  //! RNG multiplier raised to the volume+1
  Seed ran_mult_n;
  //! The lattice of skewed RNG multipliers
  LatticeSeed *lattice_ran_mult;

    //! Find the number of bits required to represent x.
  int numbits(int x)
  {
    int num = 1;
    int iceiling = 2;
    while (iceiling <= x)
    {
      num++;
      iceiling *= 2;
    }

    return num;
  }


  //! Initialize the random number generator
  void InitDefaultRNG()
  {
    fprintf(stderr,"Entering setrn\n");
    Seed seed = 11;
    fprintf(stderr,"Really entering setrn\n");
    RNG::setrn(seed);
    fprintf(stderr,"Finished setrn\n");
  }


  //! Initialize the random number generator
  void setrn(const Seed& seed_tmp)
  {
    ran_seed = seed_tmp;

    /* Multiplier used. Use big integer arithmetic */
    Seed seed_tmp3;
    Seed seed_tmp2;
    Seed seed_tmp1;
    Seed seed_tmp0;

    seed_tmp3 = 1222;
    seed_tmp2 = (seed_tmp3 << 12) | 1498;
    seed_tmp1 = (seed_tmp2 << 12) | 712;
    seed_tmp0 = (seed_tmp1 << 12) | 1645;

    ran_mult = seed_tmp0;

    /* Allocate space for the lexicographic ordering of the lattice sites */
    /*   and initialize. */
    /* NOTE!!!: increment the size by 1 to hold the lexicographic coordinate */
    /*   at site  LATTICEVOLUME-1 (assumes ordering starts at zero). */
    /* Since the lattice size is a power of 2, adding one will not overflow */
    /*   onto requiring more bits. */
    int nbits = numbits(layout.Vol());

    /* Get the NEWS coordinate of each vp (note the origin is 0) and
     *   build up a lexicographic ordering for the lattice. The definition
     *   here is totally arbitrary and only this routine needs to worry
     *   about it.  The lexicographic value of site K is
     *
     *     lexoc(k) = sum_{i = 1, ndim} x(k,i)*L^i     +   1
     */
    LatticeInteger lexoc;
    lexoc = latticeCoordinate(Nd-1);

    for(int m=Nd-2; m>=0; --m)
    {
      lexoc *= layout.LattSize()[m];
      lexoc += latticeCoordinate(m);
    }

    lexoc += 1;

    /*
     * Setup single multiplier ( a^1 ) on each site 
     */
    LatticeSeed laa;
    laa = ran_mult;

    /*
     * Calculate the multiplier  a^n  where n = lexicographic numbering of the site.
     *   Put one into each an_f, then multiply them by  a^(2^i) under the context
     *   flag where i is the bit number of the lexicographic numbering.
     *   In other words, the very first site is multiplied by a.
     */
    LatticeSeed lattice_ran_mult_tmp;
    lattice_ran_mult_tmp = 1;

    LatticeSeed laamult;
    LatticeBoolean lbit;

    for(int i=0; i<nbits; ++i)
    {
      lbit = (lexoc & 1) > 0;

      laamult = lattice_ran_mult_tmp * laa;
      copymask(lattice_ran_mult_tmp,lbit,laamult);

      lexoc >>= 1;
      laamult = laa * laa;
      laa = laamult;
    }

    /* Calculate separately the multiplier for the highest lexicographically ordered site. */
    bool bit;
    Seed aa;
    Seed aamult;

    int ibit = layout.Vol();
    aa = ran_mult;
    ran_mult_n = 1;

    for(int i=0; i<nbits; ++i)
    {
      bit = (ibit & 1) > 0;

      aamult = ran_mult_n * aa;
      if (bit)
	ran_mult_n = aamult;

      ibit >>= 1;
      aamult = aa * aa;
      aa = aamult;
    }

    lattice_ran_mult = new LatticeSeed;

    *lattice_ran_mult = lattice_ran_mult_tmp;

#if 0
    Push(cerr,"setrn");
    WRITE_NAMELIST(cerr,ran_seed);
    WRITE_NAMELIST(cerr,ran_mult);
    WRITE_NAMELIST(cerr,ran_mult_n);
    WRITE_NAMELIST(cerr,lattice_ran_mult_tmp);
    Pop(cerr);
#endif

    cerr << "exiting setrn: destructors will be called\n";
  }


  //! Return a copy of the random number seed
  void savern(Seed& seed)
  {
    seed = ran_seed;
  }


  //! Scalar random number generator. Done on the front end. */
  /*! 
   * It is linear congruential with modulus m = 2**47, increment c = 0,
   * and multiplier a = (2**36)*m3 + (2**24)*m2 + (2**12)*m1 + m0.  
   */
  float sranf(Seed& seed, Seed& skewed_seed, const Seed& seed_mult)
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
    Real _sranf;
    float _ssranf;
    Seed ran_tmp;

    _sranf = seedToFloat(skewed_seed);
    cast_rep(_ssranf, _sranf);

    ran_tmp = seed * seed_mult;
    seed = ran_tmp;

    ran_tmp = skewed_seed * seed_mult;
    skewed_seed = ran_tmp;

    return _ssranf;
  }

};

QDP_END_NAMESPACE();
