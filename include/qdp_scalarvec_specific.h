// -*- C++ -*-
// $Id: qdp_scalarvec_specific.h,v 1.4 2003-08-21 02:47:11 edwards Exp $
//
// QDP data parallel interface
//
// Outer/inner lattice routines specific to a scalarvec platform 

QDP_BEGIN_NAMESPACE(QDP);

//-----------------------------------------------------------------------------
// Layout stuff specific to a scalar architecture
namespace Layout
{
  //! coord[mu]  <- mu  : fill with lattice coord in mu direction
  LatticeInteger latticeCoordinate(int mu);

  //! Number of sites in the outer grid
  int outerSitesOnNode() QDP_CONST;
}


//-----------------------------------------------------------------------------
// Internal ops designed to look like those in parscalar
// These dummy routines exist just to make code more portable
namespace Internal
{
  //! Dummy broadcast from primary node to all other nodes
  template<class T>
  void broadcast(T& dest) {}

  //! Dummy broadcast from primary node to all other nodes
  inline void broadcast(void* dest, unsigned int nbytes) {}

  //! Dummy broadcast from primary node to all other nodes
  inline void stupidBroadcast(void* dest, unsigned int nbytes) {}
}

#define QDP_NOT_IMPLEMENTED

//-----------------------------------------------------------------------------
//! OLattice Op Scalar(Expression(source)) under an UnorderedSubset
/*! 
 * OLattice Op Expression, where Op is some kind of binary operation 
 * involving the destination 
 */
template<class T, class T1, class Op, class RHS>
//inline
void evaluate(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs,
	      const UnorderedSubset& s)
{
//  cerr << "In evaluateUnorderedSubet(olattice,oscalar)\n";

#if ! defined(QDP_NOT_IMPLEMENTED)
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
//    fprintf(stderr,"eval(olattice,oscalar): site %d\n",i);
//    op(dest.elem(i), forEach(rhs, ElemLeaf(), OpCombine()));
    op(dest.elem(i), forEach(rhs, EvalLeaf1(0), OpCombine()));
  }
#else
  QDP_error("evaluateUnorderedSubset not implemented");
#endif
}

//! OLattice Op Scalar(Expression(source)) under an OrderedSubset
/*! 
 * OLattice Op Expression, where Op is some kind of binary operation 
 * involving the destination 
 */
template<class T, class T1, class Op, class RHS>
//inline
void evaluate(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs,
	      const OrderedSubset& s)
{
//  cerr << "In evaluateOrderedSubset(olattice,oscalar)\n";

  const int istart = s.start() >> INNER_LEN;
  const int iend   = s.end()   >> INNER_LEN;

  for(int i=istart; i <= iend; ++i) 
  {
//    fprintf(stderr,"eval(olattice,oscalar): site %d\n",i);
    op(dest.elem(i), forEach(rhs, EvalLeaf1(0), OpCombine()));
  }
}

//! OLattice Op Scalar(Expression(source)) under the ALL subset
/*! 
 * OLattice Op Expression, where Op is some kind of binary operation 
 * involving the destination 
 */
template<class T, class T1, class Op, class RHS>
inline
void evaluate(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs)
{
//  cerr << "In evaluateAll(olattice,oscalar)\n";

  evaluate(dest, op, rhs, all);
}


//! OLattice Op OLattice(Expression(source)) under an UnorderedSubset
/*! 
 * OLattice Op Expression, where Op is some kind of binary operation 
 * involving the destination 
 */
template<class T, class T1, class Op, class RHS>
//inline
void evaluate(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs,
	      const UnorderedSubset& s)
{
//  cerr << "In evaluateUnorderedSubset(olattice,olattice)" << endl;

#if ! defined(QDP_NOT_IMPLEMENTED)
  // General form of loop structure
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
//    fprintf(stderr,"eval(olattice,olattice): site %d\n",i);
    op(dest.elem(i), forEach(rhs, EvalLeaf1(i), OpCombine()));
  }
#else
  QDP_error("evaluateUnorderedSubset not implemented");
#endif
}


//! OLattice Op OLattice(Expression(source)) under an OrderedSubset
/*! 
 * OLattice Op Expression, where Op is some kind of binary operation 
 * involving the destination 
 */
template<class T, class T1, class Op, class RHS>
//inline
void evaluate(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs,
	      const OrderedSubset& s)
{
//  cerr << "In evaluateOrderedSubset(olattice,olattice)" << endl;

  const int istart = s.start() >> INNER_LEN;
  const int iend   = s.end()   >> INNER_LEN;

  for(int i=istart; i <= iend; ++i) 
  {
//    fprintf(stderr,"eval(olattice,olattice): site %d\n",i);
    op(dest.elem(i), forEach(rhs, EvalLeaf1(i), OpCombine()));
  }
}


//! OLattice Op OLattice(Expression(source))
template<class T, class T1, class Op, class RHS>
inline
void evaluate(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs)
{
//  cerr << "In evaluateAll(olattice,olattice)\n";

  evaluate(dest, op, rhs, all);
}



//-----------------------------------------------------------------------------
//! dest = (mask) ? s1 : dest
template<class T1, class T2> 
void 
copymask(OSubLattice<T2,UnorderedSubset> d, const OLattice<T1>& mask, const OLattice<T2>& s1) 
{
  OLattice<T2>& dest = d.field();
  const UnorderedSubset& s = d.subset();

#if ! defined(QDP_NOT_IMPLEMENTED)
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    copymask(dest.elem(i), mask.elem(i), s1.elem(i));
  }
#else
  QDP_error("copymask_UnorderedSubset not implemented");
#endif
}

//! dest = (mask) ? s1 : dest
template<class T1, class T2> 
void 
copymask(OSubLattice<T2,OrderedSubset> d, const OLattice<T1>& mask, const OLattice<T2>& s1) 
{
  OLattice<T2>& dest = d.field();
  const OrderedSubset& s = d.subset();

  const int istart = s.start() >> INNER_LEN;
  const int iend   = s.end()   >> INNER_LEN;

  for(int i=istart; i <= iend; ++i) 
    copymask(dest.elem(i), mask.elem(i), s1.elem(i));
}

//! dest = (mask) ? s1 : dest
template<class T1, class T2> 
void 
copymask(OLattice<T2>& dest, const OLattice<T1>& mask, const OLattice<T2>& s1) 
{
  const int iend = Layout::outerSitesOnNode();
  for(int i=0; i < iend; ++i) 
    copymask(dest.elem(i), mask.elem(i), s1.elem(i));
}



//-----------------------------------------------------------------------------
// Random numbers
namespace RNG
{
  extern Seed ran_seed;
  extern Seed ran_mult;
  extern Seed ran_mult_n;
  extern LatticeSeed *lattice_ran_mult;
}


//! dest  = random  
/*! This implementation is correct for no inner grid */
template<class T>
void 
random(OScalar<T>& d)
{
  Seed seed = RNG::ran_seed;
  Seed skewed_seed = RNG::ran_seed * RNG::ran_mult;

  fill_random(d.elem(), seed, skewed_seed, RNG::ran_mult);

  RNG::ran_seed = seed;  // The seed from any site is the same as the new global seed
}


//! dest  = random    under a subset
template<class T>
void 
random(OLattice<T>& d, const UnorderedSubset& s)
{
  Seed seed;
  Seed skewed_seed;

#if ! defined(QDP_NOT_IMPLEMENTED)
#error "random(unorderedsubset) broken"
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    seed = RNG::ran_seed;
    skewed_seed.elem() = RNG::ran_seed.elem() * RNG::lattice_ran_mult->elem(i);
    fill_random(d.elem(i), seed, skewed_seed, RNG::ran_mult_n);
  }

  RNG::ran_seed = seed;  // The seed from any site is the same as the new global seed
#else
  QDP_error("random_UnorderedSubset not implemented");
#endif
}


//! dest  = random    under a subset
/*! This version uses an inner-grid */
template<class T>
void 
random(OLattice<T>& d, const OrderedSubset& s)
{
  Seed seed;
  ILatticeSeed skewed_seed;

  const int istart = s.start() >> INNER_LEN;
  const int iend   = s.end()   >> INNER_LEN;

  for(int i=istart; i <= iend; ++i) 
  {
    seed = RNG::ran_seed;
    skewed_seed.elem() = RNG::ran_seed.elem() * RNG::lattice_ran_mult->elem(i);
    fill_random(d.elem(i), seed, skewed_seed, RNG::ran_mult_n);
  }

  RNG::ran_seed = seed;  // The seed from any site is the same as the new global seed
}


//! dest  = random   under a subset
template<class T, class S>
void random(const OSubLattice<T,S>& dd)
{
  OLattice<T>& d = const_cast<OSubLattice<T,S>&>(dd).field();
  const S& s = dd.subset();

  random(d,s);
}


//! dest  = random  
template<class T>
void random(OLattice<T>& d)
{
  random(d,all);
}


//! dest  = gaussian   under a subset
template<class T>
void gaussian(OLattice<T>& d, const UnorderedSubset& s)
{
  OLattice<T>  r1, r2;

  random(r1,s);
  random(r2,s);

#if ! defined(QDP_NOT_IMPLEMENTED)
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    fill_gaussian(d.elem(i), r1.elem(i), r2.elem(i));
  }
#else
  QDP_error("gaussianUnorderedSubset not implemented");
#endif
}

//! dest  = gaussian   under a subset
template<class T>
void gaussian(OLattice<T>& d, const OrderedSubset& s)
{
  OLattice<T>  r1, r2;

  random(r1,s);
  random(r2,s);

  const int istart = s.start() >> INNER_LEN;
  const int iend   = s.end()   >> INNER_LEN;

  for(int i=istart; i <= iend; ++i) 
    fill_gaussian(d.elem(i), r1.elem(i), r2.elem(i));
}


//! dest  = gaussian   under a subset
template<class T, class S>
void gaussian(const OSubLattice<T,S>& dd)
{
  OLattice<T>& d = const_cast<OSubLattice<T,S>&>(dd).field();
  const S& s = dd.subset();

  gaussian(d,s);
}


//! dest  = gaussian
template<class T>
void gaussian(OLattice<T>& d)
{
  gaussian(d,all);
}



//-----------------------------------------------------------------------------
// Broadcast operations
//! dest  = 0 
template<class T> 
void zero_rep(OLattice<T>& dest, const UnorderedSubset& s) 
{
#if ! defined(QDP_NOT_IMPLEMENTED)
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    zero_rep(dest.elem(i));
  }
#else
  QDP_error("zero_rep_UnorderedSubset not implemented");
#endif
}

//! dest  = 0 
template<class T> 
void zero_rep(OLattice<T>& dest, const OrderedSubset& s) 
{
  const int istart = s.start() >> INNER_LEN;
  const int iend   = s.end()   >> INNER_LEN;

  for(int i=istart; i <= iend; ++i) 
    zero_rep(dest.elem(i));
}


//! dest  = 0 
template<class T, class S>
void zero_rep(OSubLattice<T,S> dd) 
{
  OLattice<T>& d = dd.field();
  const S& s = dd.subset();
  
  zero_rep(d,s);
}


//! dest  = 0 
template<class T> 
void zero_rep(OLattice<T>& dest) 
{
  const int iend = Layout::outerSitesOnNode();
  for(int i=0; i < iend; ++i) 
    zero_rep(dest.elem(i));
}



//-----------------------------------------------
// Global sums
//! OScalar = sum(OScalar) under an explicit subset
/*!
 * Allow a global sum that sums over the lattice, but returns an object
 * of the same primitive type. E.g., contract only over lattice indices
 */
template<class RHS, class T>
typename UnaryReturn<OScalar<T>, FnSum>::Type_t
sum(const QDPExpr<RHS,OScalar<T> >& s1, const Subset& s)
{
  typename UnaryReturn<OScalar<T>, FnSum>::Type_t  d;

  evaluate(d,OpAssign(),s1);
  return d;
}


//! OScalar = sum(OScalar)
/*!
 * Allow a global sum that sums over the lattice, but returns an object
 * of the same primitive type. E.g., contract only over lattice indices
 */
template<class RHS, class T>
typename UnaryReturn<OScalar<T>, FnSum>::Type_t
sum(const QDPExpr<RHS,OScalar<T> >& s1)
{
  typename UnaryReturn<OScalar<T>, FnSum>::Type_t  d;

  evaluate(d,OpAssign(),s1);
  return d;
}



//! OScalar = sum(OLattice)  under an explicit subset
/*!
 * Allow a global sum that sums over the lattice, but returns an object
 * of the same primitive type. E.g., contract only over lattice indices
 */
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnSum>::Type_t
sum(const QDPExpr<RHS,OLattice<T> >& s1, const Subset& s)
{
  typename UnaryReturn<OLattice<T>, FnSum>::Type_t  d;

  // Must initialize to zero since we do not know if the loop will be entered
  zero_rep(d.elem());

#if ! defined(QDP_NOT_IMPLEMENTED)
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    d.elem() += forEach(s1, EvalLeaf1(i), OpCombine());   // SINGLE NODE VERSION FOR NOW
  }
#else
  QDP_error("sum_UnorderedSubset not implemented");
#endif

  return d;
}


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
  OScalar<T> tmp;   // Note, expect to have ILattice inner grid

  // Loop always entered - could unroll
  zero_rep(d.elem());

  const int iend = Layout::outerSitesOnNode();
  for(int i=0; i < iend; ++i) 
  {
    tmp.elem() = forEach(s1, EvalLeaf1(i), OpCombine()); // Evaluate to ILattice part
    d.elem() += sum(tmp.elem());    // sum as well the ILattice part
  }

  return d;
}


//-----------------------------------------------------------------------------
// Multiple global sums 
//! multi1d<OScalar> dest  = sumMulti(OScalar,Set) 
/*!
 * Compute the global sum on multiple subsets specified by Set 
 *
 * This implementation is specific to a purely olattice like
 * types. The scalar input value is replicated to all the
 * slices
 */
template<class RHS, class T>
typename UnaryReturn<OScalar<T>, FnSum>::Type_t
sumMulti(const QDPExpr<RHS,OScalar<T> >& s1, const Set& ss)
{
  typename UnaryReturn<OScalar<T>, FnSumMulti>::Type_t  dest(ss.numSubsets());

  // lazy - evaluate repeatedly
  for(int i=0; i < ss.numSubsets(); ++i)
  {
    evaluate(dest[i],OpAssign(),s1);
  }

  return dest;
}


//! multi1d<OScalar> dest  = sumMulti(OLattice,Set) 
/*!
 * Compute the global sum on multiple subsets specified by Set 
 *
 * This is a very simple implementation. There is no need for
 * anything fancier unless global sums are just so extraordinarily
 * slow. Otherwise, generalized sums happen so infrequently the slow
 * version is fine.
 */
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t
sumMulti(const QDPExpr<RHS,OLattice<T> >& s1, const Set& ss)
{
  typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t  dest(ss.numSubsets());

  // Initialize result with zero
  for(int k=0; k < ss.numSubsets(); ++k)
    zero_rep(dest[k]);

  // Loop over all sites and accumulate based on the coloring 
  const multi1d<int>& lat_color =  ss.latticeColoring();

  const int iend = Layout::outerSitesOnNode();
  for(int i=0; i < iend; ++i) 
  {
    int j = lat_color[i];
    dest[j].elem() += forEach(s1, EvalLeaf1(i), OpCombine());   // SINGLE NODE VERSION FOR NOW
  }

  return dest;
}

//-----------------------------------------------------------------------------
// Peek and poke at individual sites. This is very architecture specific
// NOTE: these two routines assume there is no underlying inner grid

//! Extract site element
/*! @ingroup group1
  @param l  source to examine
  @param coord Nd lattice coordinates to examine
  @return single site object of the same primitive type
  @ingroup group1
  @relates QDPType */
template<class T1>
inline typename UnaryReturn<OScalar<T1>, FnPeekSite>::Type_t
peekSite(const OScalar<T1>& l, const multi1d<int>& coord)
{
  return l;
}


//! Extract site element
/*! @ingroup group1
  @param l  source to examine
  @param coord Nd lattice coordinates to examine
  @return single site object of the same primitive type
  @ingroup group1
  @relates QDPType */
template<class T1>
inline typename UnaryReturn<OLattice<T1>, FnPeekSite>::Type_t
peekSite(const OLattice<T1>& l, const multi1d<int>& coord)
{
  typename UnaryReturn<OLattice<T1>, FnPeekSite>::Type_t  dest;

  int i      = Layout::linearSiteIndex(coord);
  int iouter = i / INNER_LEN;
  int iinner = i % INNER_LEN;
  copy_site(dest.elem(), iinner, l.elem(iouter));
  return dest;
}


//! Insert site element
/*! @ingroup group1
  @param l  target to update
  @param r  source to insert
  @param coord Nd lattice coordinates where to insert
  @return object of the same primitive type but of promoted lattice type
  @ingroup group1
  @relates QDPType */
template<class T1, class T2>
inline OLattice<T1>&
pokeSite(OLattice<T1>& l, const OScalar<T2>& r, const multi1d<int>& coord)
{
  int i      = Layout::linearSiteIndex(coord);
  int iouter = i / INNER_LEN;
  int iinner = i % INNER_LEN;
  copy_site(l.elem(iouter), iinner, r.elem());
  return l;
}


//! Copy data values from field src to array dest
/*! @ingroup group1
  @param dest  target to update
  @param src   QDP source to insert
  @param s     subset
  @ingroup group1
  @relates QDPType */
template<class T>
inline void 
QDP_extract(multi1d<OScalar<T> >& dest, const OLattice<T>& src, const Subset& s)
{
#if ! defined(QDP_NOT_IMPLEMENTED)
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    dest[i].elem() = src.elem(i);
  }
#else
  QDP_error("QDP_extract_UnorderedSubset not implemented");
#endif
}

//! Inserts data values from site array src.
/*! @ingroup group1
  @param dest  QDP target to update
  @param src   source to insert
  @param s     subset
  @ingroup group1
  @relates QDPType */
template<class T>
inline void 
QDP_insert(OLattice<T>& dest, const multi1d<OScalar<T> >& src, const Subset& s)
{
#if ! defined(QDP_NOT_IMPLEMENTED)
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    dest.elem(i) = src[i].elem();
  }
#else
  QDP_error("QDP_insert_UnorderedSubset not implemented");
#endif
}


//-----------------------------------------------------------------------------
// Map
//
//! General permutation map class for communications
class Map
{
public:
  //! Constructor - does nothing really
  Map() {}

  //! Destructor
  ~Map() {}

  //! Constructor from a function object
  Map(const MapFunc& fn) {make(fn);}

  //! Actual constructor from a function object
  /*! The semantics are   source_site = func(dest_site,isign) */
  void make(const MapFunc& func);

  //! Function call operator for a shift
  /*! 
   * map(source)
   *
   * Implements:  dest(x) = s1(x+offsets)
   *
   * Shifts on a OLattice are non-trivial.
   *
   * Notice, this implementation supports an Inner grid
   */
  template<class T1>
  OLattice<T1>
  operator()(const OLattice<T1> & l)
    {
#if QDP_DEBUG >= 3
      QDP_info("Map()");
#endif

      OLattice<T1> d;
      int igather[1 << INNER_LEN];

      // *** THIS IS A TERRIBLE IMPLEMENTATION - I JUST WANT IT TO WORK FIRST!!!
      // For now, use the all subset
      // *** THIS IS WRONG ***
      for(int outersite=0; outersite < Layout::outerSitesOnNode(); ++outersite) 
      {
	for(int innersite=0; innersite < (1 << INNER_LEN); ++innersite) 
	{
	  int i = (outersite << INNER_LEN) + innersite;

#if QDP_DEBUG >= 3
	  QDP_info("Map(olattice[%d],olattice[%d])",i,goffsets[i]);
#endif
//	  d.elem(iouter) = l.elem(gouter,ginner);
	  zero_rep(d.elem(outersite));

#warning "Map is broken"
	}
      }

#if QDP_DEBUG >= 3
      QDP_info("exiting Map()");
#endif

      return d;
    }


  template<class T1>
  OScalar<T1>
  operator()(const OScalar<T1> & l)
    {
      return l;
    }

  template<class RHS, class T1>
  OScalar<T1>
  operator()(const QDPExpr<RHS,OScalar<T1> > & l)
    {
      // For now, simply evaluate the expression and then do the map
      typedef OScalar<T1> C1;

//    fprintf(stderr,"map(QDPExpr<OScalar>)\n");
      OScalar<T1> d = this->operator()(C1(l));

      return d;
    }

  template<class RHS, class T1>
  OLattice<T1>
  operator()(const QDPExpr<RHS,OLattice<T1> > & l)
    {
      // For now, simply evaluate the expression and then do the map
      typedef OLattice<T1> C1;

//    fprintf(stderr,"map(QDPExpr<OLattice>)\n");
      OLattice<T1> d = this->operator()(C1(l));

      return d;
    }


public:
  //! Accessor to offsets
  const multi1d<int>& goffset() const {return goffsets;}

private:
  //! Hide copy constructor
  Map(const Map&) {}

  //! Hide operator=
  void operator=(const Map&) {}

private:
  /*! 
   * The direction is in the sense of the Map or Shift functions from QDP.
   * goffsets(position) 
   */ 
  multi1d<int> goffsets;
};


//-----------------------------------------------------------------------------
//! Array of general permutation map class for communications
class ArrayMap
{
public:
  //! Constructor - does nothing really
  ArrayMap() {}

  //! Destructor
  ~ArrayMap() {}

  //! Constructor from a function object
  ArrayMap(const ArrayMapFunc& fn) {make(fn);}

  //! Actual constructor from a function object
  /*! The semantics are   source_site = func(dest_site,isign,dir) */
  void make(const ArrayMapFunc& func);

  //! Function call operator for a shift
  /*! 
   * map(source,dir)
   *
   * Implements:  dest(x) = source(map(x,dir))
   *
   * Shifts on a OLattice are non-trivial.
   * Notice, there may be an ILattice underneath which requires shift args.
   * This routine is very architecture dependent.
   */
  template<class T1>
  OLattice<T1>
  operator()(const OLattice<T1> & l, int dir)
    {
#if QDP_DEBUG >= 3
      QDP_info("ArrayMap(OLattice,%d)",dir);
#endif

      return mapsa[dir](l);
    }

  template<class T1>
  OScalar<T1>
  operator()(const OScalar<T1> & l, int dir)
    {
#if QDP_DEBUG >= 3
      QDP_info("ArrayMap(OScalar,%d)",dir);
#endif

      return mapsa[dir](l);
    }


  template<class RHS, class T1>
  OScalar<T1>
  operator()(const QDPExpr<RHS,OScalar<T1> > & l, int dir)
    {
//    fprintf(stderr,"ArrayMap(QDPExpr<OScalar>,%d)\n",dir);

      // For now, simply evaluate the expression and then do the map
      return mapsa[dir](l);
    }

  template<class RHS, class T1>
  OLattice<T1>
  operator()(const QDPExpr<RHS,OLattice<T1> > & l, int dir)
    {
//    fprintf(stderr,"ArrayMap(QDPExpr<OLattice>,%d)\n",dir);

      // For now, simply evaluate the expression and then do the map
      return mapsa[dir](l);
    }


private:
  //! Hide copy constructor
  ArrayMap(const ArrayMap&) {}

  //! Hide operator=
  void operator=(const ArrayMap&) {}

private:
  multi1d<Map> mapsa;
  
};

//-----------------------------------------------------------------------------
//! BiDirectional of general permutation map class for communications
class BiDirectionalMap
{
public:
  //! Constructor - does nothing really
  BiDirectionalMap() {}

  //! Destructor
  ~BiDirectionalMap() {}

  //! Constructor from a function object
  BiDirectionalMap(const MapFunc& fn) {make(fn);}

  //! Actual constructor from a function object
  /*! The semantics are   source_site = func(dest_site,isign) */
  void make(const MapFunc& func);

  //! Function call operator for a shift
  /*! 
   * map(source,isign)
   *
   * Implements:  dest(x) = source(map(x,isign))
   *
   * Shifts on a OLattice are non-trivial.
   * Notice, there may be an ILattice underneath which requires shift args.
   * This routine is very architecture dependent.
   */
  template<class T1>
  OLattice<T1>
  operator()(const OLattice<T1> & l, int isign)
    {
#if QDP_DEBUG >= 3
      QDP_info("BiDirectionalMap(OLattice,%d)",isign);
#endif

      return bimaps[(isign+1)>>1](l);
    }


  template<class T1>
  OScalar<T1>
  operator()(const OScalar<T1> & l, int isign)
    {
#if QDP_DEBUG >= 3
      QDP_info("BiDirectionalMap(OScalar,%d)",isign);
#endif

      return bimaps[(isign+1)>>1](l);
    }


  template<class RHS, class T1>
  OScalar<T1>
  operator()(const QDPExpr<RHS,OScalar<T1> > & l, int isign)
    {
//    fprintf(stderr,"BiDirectionalMap(QDPExpr<OScalar>,%d)\n",isign);

      // For now, simply evaluate the expression and then do the map
      return bimaps[(isign+1)>>1](l);
    }

  template<class RHS, class T1>
  OLattice<T1>
  operator()(const QDPExpr<RHS,OLattice<T1> > & l, int isign)
    {
//    fprintf(stderr,"BiDirectionalMap(QDPExpr<OLattice>,%d)\n",isign);

      // For now, simply evaluate the expression and then do the map
      return bimaps[(isign+1)>>1](l);
    }


private:
  //! Hide copy constructor
  BiDirectionalMap(const BiDirectionalMap&) {}

  //! Hide operator=
  void operator=(const BiDirectionalMap&) {}

private:
  multi1d<Map> bimaps;
  
};


//-----------------------------------------------------------------------------
//! ArrayBiDirectional of general permutation map class for communications
class ArrayBiDirectionalMap
{
public:
  //! Constructor - does nothing really
  ArrayBiDirectionalMap() {}

  //! Destructor
  ~ArrayBiDirectionalMap() {}

  //! Constructor from a function object
  ArrayBiDirectionalMap(const ArrayMapFunc& fn) {make(fn);}

  //! Actual constructor from a function object
  /*! The semantics are   source_site = func(dest_site,isign,dir) */
  void make(const ArrayMapFunc& func);

  //! Function call operator for a shift
  /*! 
   * Implements:  dest(x) = source(map(x,isign,dir))
   *
   * Syntax:
   * map(source,isign,dir)
   *
   * isign = parity of direction (+1 or -1)
   * dir   = array index (could be direction in range [0,...,Nd-1])
   *
   * Implements:  dest(x) = s1(x+isign*dir)
   * There are cpp macros called  FORWARD and BACKWARD that are +1,-1 resp.
   * that are often used as arguments
   *
   * Shifts on a OLattice are non-trivial.
   * Notice, there may be an ILattice underneath which requires shift args.
   * This routine is very architecture dependent.
   */
  template<class T1>
  OLattice<T1>
  operator()(const OLattice<T1> & l, int isign, int dir)
    {
#if QDP_DEBUG >= 3
      QDP_info("ArrayBiDirectionalMap(OLattice,%d,%d)",isign,dir);
#endif

      return bimapsa((isign+1)>>1,dir)(l);
    }

  template<class T1>
  OScalar<T1>
  operator()(const OScalar<T1> & l, int isign, int dir)
    {
#if QDP_DEBUG >= 3
      QDP_info("ArrayBiDirectionalMap(OScalar,%d,%d)",isign,dir);
#endif

      return bimapsa((isign+1)>>1,dir)(l);
    }


  template<class RHS, class T1>
  OScalar<T1>
  operator()(const QDPExpr<RHS,OScalar<T1> > & l, int isign, int dir)
    {
//    fprintf(stderr,"ArrayBiDirectionalMap(QDPExpr<OScalar>,%d,%d)\n",isign,dir);

      // For now, simply evaluate the expression and then do the map
      return bimapsa((isign+1)>>1,dir)(l);
    }

  template<class RHS, class T1>
  OLattice<T1>
  operator()(const QDPExpr<RHS,OLattice<T1> > & l, int isign, int dir)
    {
//    fprintf(stderr,"ArrayBiDirectionalMap(QDPExpr<OLattice>,%d,%d)\n",isign,dir);

      // For now, simply evaluate the expression and then do the map
      return bimapsa((isign+1)>>1,dir)(l);
    }


private:
  //! Hide copy constructor
  ArrayBiDirectionalMap(const ArrayBiDirectionalMap&) {}

  //! Hide operator=
  void operator=(const ArrayBiDirectionalMap&) {}

private:
  multi2d<Map> bimapsa;
  
};


//-----------------------------------------------------------------------------
// Input and output of various flavors that are architecture specific

//! Namelist output
template<class T>  
NmlWriter& operator<<(NmlWriter& nml, const OLattice<T>& d)
{
  nml.get() << "   [OUTER]" << endl;
  const int iend = Layout::outerSitesOnNode();
  for(int site=0; site < iend; ++site) 
  {
    int i = Layout::linearSiteIndex(site);
    nml.get() << "   Site =  " << site << "   = ";
    nml << d.elem(i);
    nml.get() << " ," << endl;
  }

//  int site = Layout::outerSitesOnNode()-1;
//  int i = Layout::linearSiteIndex(site);
//  nml << "   Site =  " << site << "   = " << d.elem(i) << ",\n";

  return nml;
}

//! XML output
template<class T>  
XMLWriter& operator<<(XMLWriter& xml, const OLattice<T>& d)
{
  xml.openTag("OLattice");

  XMLWriterAPI::AttributeList alist;

  const int iend = Layout::outerSitesOnNode();
  for(int site=0; site < iend; ++site) 
  {
    int i = Layout::linearSiteIndex(site);

    alist.clear();
    alist.push_back(XMLWriterAPI::Attribute("site", i));

    xml.openTag("elem", alist);
    xml << d.elem(i);
    xml.closeTag();
  }

  xml.closeTag(); // OLattice

  return xml;
}



//! Binary output
/*! Assumes no inner grid */
template<class T>
inline
void write(BinaryWriter& bin, const OScalar<T>& d)
{
  if (Layout::primaryNode()) 
    bin.writeArray((const char *)&(d.elem()), 
		   sizeof(typename WordType<T>::Type_t), 
		   sizeof(T) / sizeof(typename WordType<T>::Type_t));
}

//! Binary output
/*! Assumes no inner grid */
template<class T>  
void write(BinaryWriter& bin, const OLattice<T>& d)
{
  const int iend = Layout::outerSitesOnNode();
  for(int site=0; site < iend; ++site) 
  {
    int i = Layout::linearSiteIndex(site);
    bin.writeArray((const char*)&(d.elem(i)), 
		   sizeof(typename WordType<T>::Type_t), 
		   sizeof(T) / sizeof(typename WordType<T>::Type_t));
  }
}

//! Binary input
/*! Assumes no inner grid */
template<class T>
void read(BinaryReader& bin, OScalar<T>& d)
{
  bin.readArray((char*)&(d.elem()), 
		sizeof(typename WordType<T>::Type_t), 
		sizeof(T) / sizeof(typename WordType<T>::Type_t)); 
}

//! Binary input
/*! Assumes no inner grid */
template<class T>  
void read(BinaryReader& bin, OLattice<T>& d)
{
  const int iend = Layout::outerSitesOnNode();
  for(int site=0; site < iend; ++site) 
  {
    int i = Layout::linearSiteIndex(site);
    bin.readArray((char*)&(d.elem(i)), 
		  sizeof(typename WordType<T>::Type_t), 
		  sizeof(T) / sizeof(typename WordType<T>::Type_t));
  }
}

//! Text input
template<class T>
TextReader& operator>>(TextReader& txt, OScalar<T>& d)
{
  return txt >> d.elem();
}


QDP_END_NAMESPACE();
