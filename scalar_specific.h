// -*- C++ -*-
// $Id: scalar_specific.h,v 1.5 2002-09-14 19:48:26 edwards Exp $
//
// QDP data parallel interface
//
//! Outer lattice routines specific to a scalar platform 
/*! Scalar platform - single processor single box */

QDP_BEGIN_NAMESPACE(QDP);

// Use separate defs here. This will cause subroutine calls under g++

//! Hack a-rooney - for now barf on boolean subset representation - need better method 
void diefunc();


//-----------------------------------------------------------------------------
//! OLattice Op Scalar(Expression(source))
/*! 
 * OLattice Op Expression, where Op is some kind of binary operation 
 * involving the destination 
 */
template<class T, class T1, class Op, class RHS>
//inline
void evaluate(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs)
{
  Subset s = global_context->Sub();

//  cerr << "In evaluate(olattice,oscalar)\n";

  if (! s.IndexRep())
    for(int i=s.Start(); i <= s.End(); ++i) 
    {
//      fprintf(stderr,"eval(olattice,oscalar): site %d\n",i);
//      op(dest.elem(i), forEach(rhs, ElemLeaf(), OpCombine()));
      op(dest.elem(i), forEach(rhs, EvalLeaf1(0), OpCombine()));
    }
  else
    diefunc();
}


//! OLattice Op OLattice(Expression(source))
template<class T, class T1, class Op, class RHS>
//inline
void evaluate(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs)
{
  Subset s = global_context->Sub();

//  cerr << "In evaluate(olattice,olattice)\n";

  if (! s.IndexRep())
    for(int i=s.Start(); i <= s.End(); ++i) 
    {
//    fprintf(stderr,"eval(olattice,olattice): site %d\n",i);
      op(dest.elem(i), forEach(rhs, EvalLeaf1(i), OpCombine()));
    }
  else
    diefunc();
}



//-----------------------------------------------------------------------------
//! dest = (mask) ? s1 : dest
template<class T1, class T2> 
void copymask(OLattice<T2>& dest, const OLattice<T1>& mask, const OLattice<T2>& s1) 
{
  Subset s = global_context->Sub();

  if (! s.IndexRep())
  {
    for(int i=s.Start(); i <= s.End(); ++i) 
      copymask(dest.elem(i), mask.elem(i), s1.elem(i));
  }
  else
    diefunc();
}


//-----------------------------------------------------------------------------
// Auxilliary operations
//! coord[mu]  <- mu  : fill with lattice coord in mu direction
LatticeInteger latticeCoordinate(int mu);



//! Indexing(dest,source,coordinate) : put lattice scalar source into dest at coordinate
template<class T, class T1>
void indexing(OLattice<T>& d, const OScalar<T1>& s1, const multi1d<int>& coord)
{
  int linearsite = layout.LinearSiteIndex(coord);
  d.elem(linearsite) = s1.elem();
}


//-----------------------------------------------------------------------------
// Seed_to_float
//! dest [float type] = source [seed type]
template<class T, class T1>
void seed_to_float(OLattice<T>& d, const OLattice<T1>& s1)
{
  Subset s = global_context->Sub();

  if (! s.IndexRep())
  {
    for(int i=s.Start(); i <= s.End(); ++i) 
      seed_to_float(d.elem(i), s1.elem(i));
  }
  else
    diefunc();
}


//-----------------------------------------------------------------------------
// Random numbers
namespace RNG
{
  extern Seed ran_seed;
  extern Seed ran_mult;
  extern Seed ran_mult_n;
  extern LatticeSeed *lattice_ran_mult;
};


//! dest  = random  
/*! This implementation is correct for no inner grid */
template<class T>
void random(OScalar<T>& d)
{
  Seed seed = RNG::ran_seed;
  Seed skewed_seed = RNG::ran_seed * RNG::ran_mult;

  fill_random(d.elem(), seed, skewed_seed, RNG::ran_mult);

  RNG::ran_seed = seed;  // The seed from any site is the same as the new global seed
}

//! dest  = random  
template<class T>
void random(OLattice<T>& d)
{
  Subset s = global_context->Sub();

  if (! s.IndexRep())
  {
    Seed seed;
    Seed skewed_seed;

    for(int i=s.Start(); i <= s.End(); ++i) 
    {
      seed = RNG::ran_seed;
      skewed_seed.elem() = RNG::ran_seed.elem() * RNG::lattice_ran_mult->elem(i);

//      cerr << "site = " << i << endl;
//      WRITE_NAMELIST(cerr,seed);
//      WRITE_NAMELIST(cerr,skewed_seed);

      fill_random(d.elem(i), seed, skewed_seed, RNG::ran_mult_n);

//      WRITE_NAMELIST(cerr,seed);
//      WRITE_NAMELIST(cerr,skewed_seed);
    }

    RNG::ran_seed = seed;  // The seed from any site is the same as the new global seed
  }
  else
    diefunc();
}

//! dest  = random  
template<class T>
void gaussian(OLattice<T>& d)
{
  OLattice<T>  r1, r2;

  random(r1);
  random(r2);

  Subset s = global_context->Sub();

  if (! s.IndexRep())
  {
    for(int i=s.Start(); i <= s.End(); ++i) 
      fill_gaussian(d.elem(i), r1.elem(i), r2.elem(i));
  }
  else
    diefunc();
}


//-----------------------------------------------------------------------------
// Broadcast operations
//! dest  = 0 
template<class T> 
void zero(OLattice<T>& dest) 
{
  Subset s = global_context->Sub();

  if (! s.IndexRep())
  {
    for(int i=s.Start(); i <= s.End(); ++i) 
      zero(dest.elem(i));
  }
  else
    diefunc();
}



//-----------------------------------------------
// Global sums
//! OScalar = sum(OLattice)
/*!
 * Allow a global sum that sums over the lattice, but returns an object
 * of the same primitive type. E.g., contract only over lattice indices
 */
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnSum>::Type_t
sum(const QDPExpr<RHS,OLattice<T> >& s1)
{
  typename UnaryReturn<OLattice<T>, FnSum>::Type_t  d;
  Subset s = global_context->Sub();

  if (! s.IndexRep())
  {
    // Must initialize to zero since we do not know if the loop will be entered
    zero(d.elem());

    for(int i=s.Start(); i <= s.End(); ++i) 
    {
      d.elem() += forEach(s1, EvalLeaf1(i), OpCombine());   // SINGLE NODE VERSION FOR NOW
    }
  }
  else
    diefunc();

  return d;
}


//-----------------------------------------------------------------------------
// Global slice sums
//! dest  = slice_sum(source1,mu) 
/*!
 * Compute the global sum along the hypersurfuce orthogonal to
 * the direction mu
 *
 * This is a very simple implementation. There is no need for
 * anything fancier unless global sums are just so extraordinarily
 * slow. Otherwise, slice_sums happen so infrequently the slow
 * version is fine.
   */
template<class T>
struct UnaryReturn<OLattice<T>, FnSliceSum > {
  typedef multi1d<OScalar<typename UnaryReturn<T, FnSliceSum>::Type_t> >  Type_t;
};


template<class T>
typename UnaryReturn<OLattice<T>, FnSliceSum>::Type_t
slice_sum(const OLattice<T>& s1, int mu)
{
  if (mu < 0 || mu >= Nd)
    SZ_ERROR("direction out of bounds in slice sum");

  typename UnaryReturn<OLattice<T>, FnSliceSum>::Type_t  dest(layout.LattSize()[mu]);
  Subset s = global_context->Sub();

  int len2 = 1;
  for(int dir=0; dir < mu; ++dir)
    len2 = len2 * layout.LattSize()[dir];
  int len1 = len2 * layout.LattSize()[mu];

  // Initialize result with zero
  for(int i=0; i < layout.LattSize()[mu]; ++i)
    zero(dest[i]);

  if (! s.IndexRep())
  {
    for(int i=s.Start(); i <= s.End(); ++i) 
    {
      int site   = layout.LexicoSiteIndex(i);
      int hypsec = site % len1;
      int plane  = hypsec / len2;

      dest[plane].elem() += s1.elem(site);
    }
  }
  else
    diefunc();

  return dest;
}



//-----------------------------------------------------------------------------
// Multiple global sums 
//! dest  = sumMulti(source1,Set) 
/*!
 * Compute the global sum on multiple subsets specified by Set 
 *
 * This is a very simple implementation. There is no need for
 * anything fancier unless global sums are just so extraordinarily
 * slow. Otherwise, generalized sums happen so infrequently the slow
 * version is fine.
 */
template<class T>
struct UnaryReturn<OLattice<T>, FnSumMulti > {
  typedef multi1d<OScalar<typename UnaryReturn<T, FnSumMulti>::Type_t> >  Type_t;
};

template<class T>
typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t
sumMulti(const OLattice<T>& s1, const Set& ss)
{
  typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t  dest(ss.NumSubsets());

  // Initialize result with zero
  for(int i=0; i < ss.NumSubsets(); ++i)
  {
    Subset s = ss[i];

    zero(dest[i]);

    if (! s.IndexRep())
    {
      for(int i=s.Start(); i <= s.End(); ++i) 
	dest[i].elem() += s1.elem(layout.LexicoSiteIndex(i));
    }
    else
      diefunc();
  }

  return dest;
}



//-----------------------------------------------
//! OLattice<T> = Shift(OLattice<T1>, int isign, int dir)
/*! 
 * shift(source,isign,dir)
 * isign = parity of direction (+1 or -1)
 * dir   = direction ([0,...,Nd-1])
 *
 * Implements:  dest(x) = s1(x+isign*dir)
 * There are cpp macros called  FORWARD and BACKWARD that are +1,-1 resp.
 * that are often used as arguments
 *
 * Shifts on a OLattice are non-trivial.
 * Notice, there may be an ILattice underneath which requires shift args.
 * This routine is very architecture dependent.
 */
template<class T> 
OLattice<T> shift(const OLattice<T>& s1, int isign, int dir)
{
  OLattice<T> d;
  Subset s = global_context->Sub();
  const int *soff = s.Offsets()->slice(dir,(isign+1)>>1);

  if (! s.IndexRep())
    for(int i=s.Start(); i <= s.End(); ++i) 
    {
//    fprintf(stderr,"Shift site %d\n",i);
      d.elem(i) = s1.elem(soff[i]);  // SINGLE NODE VERSION FOR NOW
    }
  else
    diefunc();

  return d;
}



//-----------------------------------------------
//! OLattice<T> = Shift(OLattice<T1>, int isign, int dir)
/*!
   * Shifts on a OLattice are non-trivial.
   * Notice, there may be an ILattice underneath which requires shift args.
   * This routine is very architecture dependent.
   */
template<class T, class RHS>
OLattice<T> shift(const QDPExpr<RHS,OLattice<T> >& s1, int isign, int dir)
{
  OLattice<T> d;
  Subset s = global_context->Sub();
  const int *soff = s.Offsets()->slice(dir,(isign+1)>>1);

  if (! s.IndexRep())
    for(int i=s.Start(); i <= s.End(); ++i)
    {
//    fprintf(stderr,"Shift expr site %d\n",i);
      d.elem(i) = forEach(s1, EvalLeaf1(soff[i]), OpCombine());   // SINGLE NODE VERSION FOR NOW
    }
  else
    diefunc();

  return d;
}



//-----------------------------------------------
// Su2_extract
//! (OLattice<T1>,OLattice<T1>,OLattice<T1>,OLattice<T1>,su2_index) <- OLattice<T>
template<class T, class T1> 
void
su2_extract(OLattice<T1>& r_0, OLattice<T1>& r_1, 
	    OLattice<T1>& r_2, OLattice<T1>& r_3, 
	    const OLattice<T>& s1, 
	    int i1, int i2)
{
  Subset s = global_context->Sub();

  if (! s.IndexRep())
  {
    for(int i=s.Start(); i <= s.End(); ++i) 
      su2_extract(r_0.elem(i), r_1.elem(i), r_2.elem(i), r_3.elem(i), 
		  i1, i2, s1.elem(i));
  }
  else
    diefunc();
}


//-----------------------------------------------
// Sun_fill
//! OLattice<T> <- (OLattice<T1>,OLattice<T1>,OLattice<T1>,OLattice<T1>,su2_index)
template<class T, class T1>
void
sun_fill(OLattice<T>& d, 
	 const OLattice<T1>& r_0, const OLattice<T1>& r_1, 
	 const OLattice<T1>& r_2, const OLattice<T1>& r_3, 
	 int i1, int i2)
{
  Subset s = global_context->Sub();

  if (! s.IndexRep())
  {
    for(int i=s.Start(); i <= s.End(); ++i) 
      sun_fill(d.elem(i), 
	       i1, i2,
	       r_0.elem(i), r_1.elem(i), r_2.elem(i), r_3.elem(i));
  }
  else
    diefunc();
}


#if 0
//-----------------------------------------------------------------------------
// Spin project
//! OLattice = spinProject(OLattice)
template<class T>
struct UnaryReturn<OLattice<T>, FnSpinProject > {
  typedef OLattice<OScalar<typename UnaryReturn<T, FnSpinProject>::Type_t> >  Type_t;
};

template<class T>
typename UnaryReturn<OLattice<T>, FnSpinProject>::Type_t
spin_project(const OLattice<T>& s1, int mu, int isign)
{
  typename UnaryReturn<OLattice<T>, FnSpinProject>::Type_t  d;
  Subset s = global_context->Sub();

  if (! s.IndexRep())
  {
    for(int i=s.Start(); i <= s.End(); ++i) 
      d.elem(i) = spin_project(d.elem(i), s1.elem(i));
  }
  else
    diefunc();
}
#endif



//-----------------------------------------------------------------------------
//! Ascii output
template<class T>  ostream& operator<<(ostream& s, const OLattice<T>& d)
{
  s << "   [OUTER]\n";
  for(int site=0; site < layout.Vol()-1; ++site) 
  {
    int i = layout.LinearSiteIndex(site);
    s << "   Site =  " << site << "   = " << d.elem(i) << ",\n";
  }

  int site = layout.Vol()-1;
  int i = layout.LinearSiteIndex(site);
  s << "   Site =  " << site << "   = " << d.elem(i) << ",";

  return s;
}


QDP_END_NAMESPACE();
