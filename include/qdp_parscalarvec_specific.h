// -*- C++ -*-
// $Id: qdp_parscalarvec_specific.h,v 1.7 2003-09-03 19:53:32 edwards Exp $

/*! @file
 * @brief Outer/inner lattice routines specific to a parscalarvec platform 
 */

#include "qmp.h"

QDP_BEGIN_NAMESPACE(QDP);

//-----------------------------------------------------------------------------
// Layout stuff specific to a parscalarvec architecture
namespace Layout
{
  //! coord[mu]  <- mu  : fill with lattice coord in mu direction
  LatticeInteger latticeCoordinate(int mu);

  //! Number of sites in the outer grid
  int outerSitesOnNode() QDP_CONST;
}


//-----------------------------------------------------------------------------
// Internal ops with ties to QMP
namespace Internal
{
  //! Wait on send-receive
  void wait(int dir);

  //! Send to another node (wait)
  void sendToWait(void *send_buf, int dest_node, int count);

  //! Receive from another node (wait)
  void recvFromWait(void *recv_buf, int srce_node, int count);

  //! Via some mechanism, get the dest to node 0
  /*! Ultimately, I do not want to use point-to-point */
  template<class T>
  void sendToPrimaryNode(T& dest, int srcnode)
  {
    if (srcnode != 0)
    {
      if (Layout::primaryNode())
	recvFromWait((void *)&dest, srcnode, sizeof(T));

      if (Layout::nodeNumber() == srcnode)
	sendToWait((void *)&dest, 0, sizeof(T));
    }
  }

  //! Low level hook to QMP_global_sum
  inline void globalSumArray(int *dest, unsigned int len)
  {
    for(unsigned int i=0; i < len; i++, dest++)
      QMP_sum_int(dest);
  }

  //! Low level hook to QMP_global_sum
  inline void globalSumArray(float *dest, unsigned int len)
  {
    QMP_sum_float_array(dest, len);
  }

  //! Low level hook to QMP_global_sum
  inline void globalSumArray(double *dest, unsigned int len)
  {
    QMP_sum_double_array(dest, len);
  }

  //! Sum across all nodes
  template<class T>
  void globalSum(T& dest)
  {
    // The implementation here is relying on the structure being packed
    // tightly in memory - no padding
    typedef typename WordType<T>::Type_t  W;   // find the machine word type
    globalSumArray((W *)&dest, sizeof(T)/sizeof(W)); // call appropriate hook
  }

  //! Broadcast from primary node to all other nodes
  template<class T>
  void broadcast(T& dest)
  {
    QMP_broadcast((void *)&dest, sizeof(T));
  }

  //! Broadcast from primary node to all other nodes
  inline void broadcast(void* dest, unsigned int nbytes)
  {
    QMP_broadcast(dest, nbytes);
  }
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

  const int istart = s.start() >> INNER_LOG;
  const int iend   = s.end()   >> INNER_LOG;

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

  const int istart = s.start() >> INNER_LOG;
  const int iend   = s.end()   >> INNER_LOG;

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

  const int istart = s.start() >> INNER_LOG;
  const int iend   = s.end()   >> INNER_LOG;

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

  const int istart = s.start() >> INNER_LOG;
  const int iend   = s.end()   >> INNER_LOG;

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

  const int istart = s.start() >> INNER_LOG;
  const int iend   = s.end()   >> INNER_LOG;

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
  const int istart = s.start() >> INNER_LOG;
  const int iend   = s.end()   >> INNER_LOG;

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
 *
 * This will include a parent Subset and an UnorderedSubset.
 *
 * NOTE: if this implementation does not have  hasOrderedRep() == true,
 * then the implementation can be quite slow
 */
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnSum>::Type_t
sum(const QDPExpr<RHS,OLattice<T> >& s1, const Subset& s)
{
  typename UnaryReturn<OLattice<T>, FnSum>::Type_t  d;
  OScalar<T> tmp;   // Note, expect to have ILattice inner grid

  // Must initialize to zero since we do not know if the loop will be entered
  zero_rep(d.elem());

  if (s.hasOrderedRep())
  {
    const int istart = s.start() >> INNER_LOG;
    const int iend   = s.end()   >> INNER_LOG;

    for(int i=istart; i <= iend; ++i) 
    {
      tmp.elem() = forEach(s1, EvalLeaf1(i), OpCombine()); // Evaluate to ILattice part
      d.elem() += sum(tmp.elem());    // sum as well the ILattice part
    }
  }
  else
  {
    const int *tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); ++j) 
    {
      int i = tab[j];
      int outersite = i >> INNER_LOG;
      int innersite = i & (INNER_LEN-1);

      tmp.elem() = forEach(s1, EvalLeaf1(outersite), OpCombine()); // Evaluate to ILattice part
      d.elem() += getSite(tmp.elem(),innersite);    // wasteful - only extract a single site worth
    }
  }

  // Do a global sum on the result
  Internal::globalSum(d);
  
  return d;
}



//! OScalar = sum(OLattice) under an explicit OrderedSubset
/*!
 * Allow a global sum that sums over the lattice, but returns an object
 * of the same primitive type. E.g., contract only over lattice indices
 */
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnSum>::Type_t
sum(const QDPExpr<RHS,OLattice<T> >& s1, const OrderedSubset& s)
{
  typename UnaryReturn<OLattice<T>, FnSum>::Type_t  d;
  OScalar<T> tmp;   // Note, expect to have ILattice inner grid

  // Loop always entered - could unroll
  zero_rep(d.elem());

  const int istart = s.start() >> INNER_LOG;
  const int iend   = s.end()   >> INNER_LOG;

  for(int i=istart; i <= iend; ++i) 
  {
    tmp.elem() = forEach(s1, EvalLeaf1(i), OpCombine()); // Evaluate to ILattice part
    d.elem() += sum(tmp.elem());    // sum as well the ILattice part
  }

  // Do a global sum on the result
  Internal::globalSum(d);
  
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
  return sum(s1,all);
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
    dest[i] = sum(s1,ss[i]);
  
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

  // lazy - evaluate repeatedly
  for(int i=0; i < ss.numSubsets(); ++i)
    dest[i] = sum(s1,ss[i]);

  return dest;
}


//! multi1d<OScalar> dest  = sumMulti(OLattice,UnorderedSet) 
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
sumMulti(const QDPExpr<RHS,OLattice<T> >& s1, const OrderedSet& ss)
{
  typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t  dest(ss.numSubsets());

  // lazy - evaluate repeatedly
  for(int i=0; i < ss.numSubsets(); ++i)
    dest[i] = sum(s1,ss[i]);

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
  int nodenum = Layout::nodeNumber(coord);

  // Find the result somewhere within the machine.
  // Then we must get it to node zero so we can broadcast it
  // out to all nodes
  if (Layout::nodeNumber() == nodenum)
  {
    int i      = Layout::linearSiteIndex(coord);
    int iouter = i >> INNER_LOG;
    int iinner = i & ((1 << INNER_LOG)-1);
    dest.elem() = getSite(l.elem(iouter), iinner);
  }
  else
    zero_rep(dest.elem());

  // Send result to primary node via some mechanism
  Internal::sendToPrimaryNode(dest, nodenum);

  // Now broadcast back out to all nodes
  Internal::broadcast(dest);

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
  if (Layout::nodeNumber() == Layout::nodeNumber(coord))
  {
    int i      = Layout::linearSiteIndex(coord);
    int iouter = i >> INNER_LOG;
    int iinner = i & ((1 << INNER_LOG)-1);
    copy_site(l.elem(iouter), iinner, r.elem());
  }
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
QDP_extract(multi1d<OScalar<typename UnaryReturn<T, FnGetSite>::Type_t> >& dest, 
	    const OLattice<T>& src, const Subset& s)
{
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    int iouter = i >> INNER_LOG;
    int iinner = i & ((1 << INNER_LOG)-1);

    dest[i].elem() = getSite(src.elem(iouter),iinner);
  }
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
QDP_insert(OLattice<T>& dest, 
	   const multi1d<OScalar<typename UnaryReturn<T, FnGetSite>::Type_t> >& src, 
	   const Subset& s)
{
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    int iouter = i >> INNER_LOG;
    int iinner = i & ((1 << INNER_LOG)-1);
    copy_site(dest.elem(iouter), iinner, src[i].elem());
  }
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
   * Notice, this implementation does not allow an Inner grid
   */
  template<class T1>
  OLattice<T1>
  operator()(const OLattice<T1> & l)
    {
      OLattice<T1> d;

#if QDP_DEBUG >= 3
      QDP_info("Map()");
#endif

      typedef typename UnaryReturn<T1, FnGetSite>::Type_t  Site_t;  // strip-off inner-grid

      if (offnodeP)
      {
	// Off-node communications required
#if QDP_DEBUG >= 3
	QDP_info("Map: off-node communications required");
#endif

	// Eventually these declarations should move into d - the return object
	typedef Site_t * T1ptr;
	Site_t **dest = new T1ptr[Layout::sitesOnNode()];
	QMP_msgmem_t msg[2];
	QMP_msghandle_t mh_a[2], mh;

	//------------
	// yukky hack - transform all source to packed format
	Site_t *ll = new Site_t[Layout::sitesOnNode()]; 
	for(int i=0; i < Layout::sitesOnNode(); ++i) 
	{
	  int iouter = i >> INNER_LOG;
	  int iinner = i & (INNER_LEN - 1);

	  ll[i] = getSite(l.elem(iouter), iinner);
	}
	//-------------


	int dstnum = destnodes_num[0]*sizeof(Site_t);
	int srcnum = srcenodes_num[0]*sizeof(Site_t);
	Site_t *send_buf = (Site_t *)(QMP_allocate_aligned_memory(dstnum)); // packed data to send
	Site_t *recv_buf = (Site_t *)(QMP_allocate_aligned_memory(srcnum)); // packed receive data

	const int my_node = Layout::nodeNumber();

	// Gather the face of data to send
	// For now, use the all subset
	for(int si=0; si < soffsets.size(); ++si) 
	{
#if QDP_DEBUG >= 3
	  QDP_info("Map_scatter_send(buf[%d],olattice[%d])",si,soffsets[si]);
#endif

	  send_buf[si] = ll[soffsets[si]];
	}

	// Set the dest gather pointers
	// For now, use the all subset
	for(int i=0, ri=0; i < Layout::sitesOnNode(); ++i) 
	{
	  if (srcnode[i] != my_node)
	  {
#if QDP_DEBUG >= 3
	    QDP_info("Map_gather_recv(olattice[%d],recv[%d])",i,ri);
#endif

	    dest[i] = &(recv_buf[ri++]);
	  }
	  else
	  {
#if QDP_DEBUG >= 3
	    QDP_info("Map_gather_onnode(olattice[%d],olattice[%d])",i,goffsets[i]);
#endif

	    dest[i] = &(ll[goffsets[i]]);
	  }
	}

	QMP_status_t err;

#if QDP_DEBUG >= 3
	QDP_info("Map: send = 0x%x  recv = 0x%x",send_buf,recv_buf);
#endif

	msg[0]  = QMP_declare_msgmem(recv_buf, srcnum);
	msg[1]  = QMP_declare_msgmem(send_buf, dstnum);
	mh_a[0] = QMP_declare_receive_from(msg[0], srcenodes[0], 0);
	mh_a[1] = QMP_declare_send_to(msg[1], destnodes[0], 0);
	mh      = QMP_declare_multiple(mh_a, 2);

#if QDP_DEBUG >= 3
	QDP_info("Map: calling start send=%d recv=%d",destnodes[0],srcenodes[0]);
#endif

	// Launch the faces
	if ((err = QMP_start(mh)) != QMP_SUCCESS)
	  QMP_error_exit(QMP_error_string(err));

#if QDP_DEBUG >= 3
	QDP_info("Map: calling wait");
#endif

	// Wait on the faces
	if ((err = QMP_wait(mh)) != QMP_SUCCESS)
	  QMP_error_exit(QMP_error_string(err));

#if QDP_DEBUG >= 3
	QDP_info("Map: calling free msgs");
#endif

	QMP_free_msghandle(mh_a[1]);
	QMP_free_msghandle(mh_a[0]);
	QMP_free_msghandle(mh);
	QMP_free_msgmem(msg[1]);
	QMP_free_msgmem(msg[0]);

	// Scatter the data into the destination
	// Some of the data maybe in receive buffers
	// For now, use the all subset
	for(int i=0; i < Layout::sitesOnNode(); ++i) 
	{
#if QDP_DEBUG >= 3
	  QDP_info("Map_scatter(olattice[%d],olattice[0x%x])",i,dest[i]);
#endif
	  int iouter = i >> INNER_LOG;
	  int iinner = i & (INNER_LEN-1);

	  copy_site(d.elem(iouter), iinner, *(dest[i]));    // slow - should use gather_sites
	}

	// Cleanup
	QMP_free_aligned_memory(recv_buf);
	QMP_free_aligned_memory(send_buf);
	delete[] ll;
	delete[] dest;

#if QDP_DEBUG >= 3
	QDP_info("finished cleanup");
#endif
      }
      else 
      {
	// No off-node communications - copy on node
#if QDP_DEBUG >= 3
	QDP_info("Map: copy on node - no communications, try this");
#endif

	// *** SHOULD IMPROVE THIS - JUST GET IT TO WORK FIRST ***
	// For now, use the all subset
#if INNER_LOG == 2
	const int vvol = Layout::sitesOnNode();
	for(int i=0; i < vvol; i+= INNER_LEN) 
	{
	  int ii = i >> INNER_LOG;
	  int o0 = goffsets[i+0] >> INNER_LOG;
	  int i0 = goffsets[i+0] & (INNER_LEN - 1);

	  int o1 = goffsets[i+1] >> INNER_LOG;
	  int i1 = goffsets[i+1] & (INNER_LEN - 1);

	  int o2 = goffsets[i+2] >> INNER_LOG;
	  int i2 = goffsets[i+2] & (INNER_LEN - 1);

	  int o3 = goffsets[i+3] >> INNER_LOG;
	  int i3 = goffsets[i+3] & (INNER_LEN - 1);

#if QDP_DEBUG >= 3
	  QDP_info("Map(lattice[%d]=lattice([%d,%d],[%d,%d],[%d,%d],[%d,%d])",
		   ii,o0,i0,o1,i1,o2,i2,o3,i3);
#endif

	  // Gather 4 inner-grid sites together
	  gather_sites(d.elem(ii),
		       l.elem(o0),i0,
		       l.elem(o1),i1,
		       l.elem(o2),i2,
		       l.elem(o3),i3);
	}

#else
#error "Map: this inner grid length is not supported - easy to fix"
#endif

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
  const multi1d<int>& soffset() const {return soffsets;}

private:
  //! Hide copy constructor
  Map(const Map&) {}

  //! Hide operator=
  void operator=(const Map&) {}

private:
  //! Offset table used for communications. 
  /*! 
   * The direction is in the sense of the Map or Shift functions from QDP.
   * goffsets(position) 
   */ 
  multi1d<int> goffsets;
  multi1d<int> soffsets;
  multi1d<int> srcnode;
  multi1d<int> dstnode;

  multi1d<int> srcenodes;
  multi1d<int> destnodes;

  multi1d<int> srcenodes_num;
  multi1d<int> destnodes_num;

  // Indicate off-node communications is needed;
  bool offnodeP;
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

//! Binary output
/*! Assumes no inner grid */
template<class T>
inline
void write(BinaryWriter& bin, const OScalar<T>& d)
{
  bin.writeArray((const char *)&(d.elem()), 
		 sizeof(typename WordType<T>::Type_t), 
		 sizeof(T) / sizeof(typename WordType<T>::Type_t));
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



// There are 2 main classes of binary/nml/xml reader/writer methods.
// The first is a simple/portable but inefficient method of send/recv
// to/from the destination node.
// The second method (the else) is a more efficient roll-around method.
// However, this method more constrains the data layout - it must be
// close to the original lexicographic order.
// For now, use the direct send method

//! Decompose a lexicographic site into coordinates
multi1d<int> crtesn(int ipos, const multi1d<int>& latt_size);

//! Ascii output
/*! An inner grid is assumed */
template<class T>  
NmlWriter& operator<<(NmlWriter& nml, const OLattice<T>& d)
{
  typedef typename UnaryReturn<T, FnGetSite>::Type_t  Site_t;
  Site_t  recv_buf;

  if (Layout::primaryNode())
    nml.get() << "   [OUTER]" << endl;

  // Find the location of each site and send to primary node
  for(int site=0; site < Layout::vol(); ++site)
  {
    multi1d<int> coord = crtesn(site, Layout::lattSize());

    int node   = Layout::nodeNumber(coord);
    int linear = Layout::linearSiteIndex(coord);
    int outersite = linear >> INNER_LOG;
    int innersite = linear & ((1 << INNER_LOG)-1);

    // Copy to buffer: be really careful since max(linear) could vary among nodes
    if (Layout::nodeNumber() == node)
      recv_buf = getSite(d.elem(outersite),innersite);  // extract into conventional scalar form

    // Send result to primary node. Avoid sending prim-node sending to itself
    if (node != 0)
    {
      if (Layout::primaryNode())
	Internal::recvFromWait((void *)&recv_buf, node, sizeof(Site_t));

      if (Layout::nodeNumber() == node)
	Internal::sendToWait((void *)&recv_buf, 0, sizeof(Site_t));
    }

    if (Layout::primaryNode())
    {
      nml.get() << "   Site =  " << site << "   = ";
      nml << recv_buf;
      nml.get() << " ," << endl;
    }
  }

  return nml;
}

//! XML output
/*! An inner grid is assumed */
template<class T>  
XMLWriter& operator<<(XMLWriter& xml, const OLattice<T>& d)
{
  typedef typename UnaryReturn<T, FnGetSite>::Type_t  Site_t;
  Site_t  recv_buf;

  xml.openTag("OLattice");
  XMLWriterAPI::AttributeList alist;

  // Find the location of each site and send to primary node
  for(int site=0; site < Layout::vol(); ++site)
  {
    multi1d<int> coord = crtesn(site, Layout::lattSize());

    int node   = Layout::nodeNumber(coord);
    int linear = Layout::linearSiteIndex(coord);
    int outersite = linear >> INNER_LOG;
    int innersite = linear & ((1 << INNER_LOG)-1);

    // Copy to buffer: be really careful since max(linear) could vary among nodes
    if (Layout::nodeNumber() == node)
      recv_buf = getSite(d.elem(outersite),innersite);  // extract into conventional scalar form

    // Send result to primary node. Avoid sending prim-node sending to itself
    if (node != 0)
    {
      if (Layout::primaryNode())
	Internal::recvFromWait((void *)&recv_buf, node, sizeof(Site_t));

      if (Layout::nodeNumber() == node)
	Internal::sendToWait((void *)&recv_buf, 0, sizeof(Site_t));
    }

    if (Layout::primaryNode())
    {
      alist.clear();
      alist.push_back(XMLWriterAPI::Attribute("site", site));

      xml.openTag("elem", alist);
      xml << recv_buf;
      xml.closeTag();
    }
  }

  xml.closeTag(); // OLattice
  return xml;
}


//! Binary output
/*! An inner grid is assumed */
template<class T>
void write(BinaryWriter& bin, const OLattice<T>& d)
{
  typedef typename UnaryReturn<T, FnGetSite>::Type_t  Site_t;
  Site_t  recv_buf;

  // Find the location of each site and send to primary node
  for(int site=0; site < Layout::vol(); ++site)
  {
    multi1d<int> coord = crtesn(site, Layout::lattSize());

    int node   = Layout::nodeNumber(coord);
    int linear = Layout::linearSiteIndex(coord);
    int outersite = linear >> INNER_LOG;
    int innersite = linear & ((1 << INNER_LOG)-1);

    // Copy to buffer: be really careful since max(linear) could vary among nodes
    if (Layout::nodeNumber() == node)
      recv_buf = getSite(d.elem(outersite),innersite);  // extract into conventional scalar form

    // Send result to primary node. Avoid sending prim-node sending to itself
    if (node != 0)
    {
      if (Layout::primaryNode())
	Internal::recvFromWait((void *)&recv_buf, node, sizeof(Site_t));

      if (Layout::nodeNumber() == node)
	Internal::sendToWait((void *)&recv_buf, 0, sizeof(Site_t));
    }

    if (Layout::primaryNode())
      bin.writeArray((char *)&recv_buf,
		     sizeof(typename WordType<Site_t>::Type_t), 
		     sizeof(Site_t) / sizeof(typename WordType<Site_t>::Type_t));
  }
}


//! Binary input
/*! An inner grid is assumed */
template<class T>
void read(BinaryReader& bin, OLattice<T>& d)
{
  typedef typename UnaryReturn<T, FnGetSite>::Type_t  Site_t;
  Site_t  recv_buf;

  // Find the location of each site and send to primary node
  for(int site=0; site < Layout::vol(); ++site)
  {
    multi1d<int> coord = crtesn(site, Layout::lattSize());

    int node   = Layout::nodeNumber(coord);
    int linear = Layout::linearSiteIndex(coord);
    int outersite = linear >> INNER_LOG;
    int innersite = linear & ((1 << INNER_LOG)-1);

    // Only on primary node read the data
    bin.readArrayPrimaryNode((char *)&recv_buf,
			     sizeof(typename WordType<Site_t>::Type_t), 
			     sizeof(Site_t) / sizeof(typename WordType<Site_t>::Type_t));

    // Send result to destination node. Avoid sending prim-node sending to itself
    if (node != 0)
    {
      if (Layout::primaryNode())
	Internal::sendToWait((void *)&recv_buf, node, sizeof(Site_t));

      if (Layout::nodeNumber() == node)
	Internal::recvFromWait((void *)&recv_buf, 0, sizeof(Site_t));
    }

    if (Layout::nodeNumber() == node)
      copy_site(d.elem(outersite), innersite, recv_buf);// insert into conventional scalar form
  }
}


QDP_END_NAMESPACE();
