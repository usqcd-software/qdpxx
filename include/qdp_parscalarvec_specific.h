// -*- C++ -*-

/*! @file
 * @brief Outer/inner lattice routines specific to a parscalarvec platform 
 */

#ifndef QDP_PARSCALARVEC_SPECIFIC_H
#define QDP_PARSCALARVEC_SPECIFIC_H

#include "qmp.h"

namespace QDP {

//-----------------------------------------------------------------------------
// Layout stuff specific to a parscalarvec architecture
namespace Layout
{
  //! coord[mu]  <- mu  : fill with lattice coord in mu direction
  LatticeInteger latticeCoordinate(int mu);
}


//-----------------------------------------------------------------------------
// Internal ops with ties to QMP
namespace QDPInternal
{
  //! Route to another node (blocking)
  void route(void *send_buf, int srce_node, int dest_node, int count);

  //! Wait on send-receive
  void wait(int dir);

  //! Send to another node (wait)
  void sendToWait(void *send_buf, int dest_node, int count);

  //! Receive from another node (wait)
  void recvFromWait(void *recv_buf, int srce_node, int count);

  //! Send a clear-to-send
  void clearToSend(void *buffer, int count, int node);

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

  //! Unsigned accumulate
  inline void sumAnUnsigned(void* inout, void* in)
  {
    *(unsigned int*)inout += *(unsigned int*)in;
  }

  //! Wrapper to get a functional unsigned global sum
  inline void globalSumArray(unsigned int *dest, int len)
  {
    for(int i=0; i < len; i++, dest++)
      QMP_binary_reduction(dest, sizeof(unsigned int), sumAnUnsigned);
  }

  //! Low level hook to QMP_global_sum
  inline void globalSumArray(int *dest, int len)
  {
    for(unsigned int i=0; i < len; i++, dest++)
      QMP_sum_int(dest);
  }

  //! Low level hook to QMP_global_sum
  inline void globalSumArray(float *dest, int len)
  {
    QMP_sum_float_array(dest, len);
  }

  //! Low level hook to QMP_global_sum
  inline void globalSumArray(double *dest, int len)
  {
    QMP_sum_double_array(dest, len);
  }

  //! Sum across all nodes
  template<class T>
  inline void globalSum(T& dest)
  {
    // The implementation here is relying on the structure being packed
    // tightly in memory - no padding
    typedef typename WordType<T>::Type_t  W;   // find the machine word type
    globalSumArray((W *)&dest, sizeof(T)/sizeof(W)); // call appropriate hook
  }

  //! Broadcast from primary node to all other nodes
  template<class T>
  inline void broadcast(T& dest)
  {
    QMP_broadcast((void *)&dest, sizeof(T));
  }

  //! Broadcast a string from primary node to all other nodes
  void broadcast_str(std::string& dest);

  //! Broadcast from primary node to all other nodes
  inline void broadcast(void* dest, size_t nbytes)
  {
    QMP_broadcast(dest, nbytes);
  }

  //! Broadcast a string from primary node to all other nodes
  template<>
  inline void broadcast(std::string& dest)
  {
    broadcast_str(dest);
  }

}

  // #define QDP_NOT_IMPLEMENTED

//-----------------------------------------------------------------------------
//! OLattice Op Scalar(Expression(source)) under an Subset
/*! 
 * OLattice Op Expression, where Op is some kind of binary operation 
 * involving the destination 
 */
template<class T, class T1, class Op, class RHS>
//inline
void evaluate(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs,
	      const Subset& s)
{
//  cerr << "In evaluateUnorderedSubet(olattice,oscalar)\n";
  int num_threads = 1;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest, op, rhs);
  prof.time -= getClockTime();
#endif

#if ! defined(QDP_NOT_IMPLEMENTED)
  if (s.hasOrderedRep()) {
    const int istart = s.start() >> INNER_LOG;
    const int iend   = s.end()   >> INNER_LOG;

#pragma omp parallel shared(istart, iend, num_threads) default(shared)
    {
      num_threads = omp_get_num_threads ();
      int myId = omp_get_thread_num();
      int low = istart + (iend - istart + 1) * myId/num_threads;
      int high = istart + (iend - istart + 1) * (myId + 1)/num_threads;

      for(int i= low; i < high; ++i) {
	//    fprintf(stderr,"eval(olattice,oscalar): site %d\n",i);
	op(dest.elem(i), forEach(rhs, EvalLeaf3(0, low, high - 1), OpCombine()));
      }
    }
  }
  else {
    // this part of code has never been tested: need more study
    const int *tab = s.siteTable().slice();

#pragma omp parallel shared(tab, num_threads) default(shared)
    {
      num_threads = omp_get_num_threads ();
      int myId = omp_get_thread_num();
      int low = s.numSiteTable() * myId/num_threads;
      int high = s.numSiteTable() * (myId + 1)/num_threads;
      
      for(int j = low; j < high; ++j) {
	int i = tab[j];
	int outersite = i >> INNER_LOG;
	int innersite = i & ((1 << INNER_LOG)-1);
	//    fprintf(stderr,"eval(olattice,oscalar): site %d\n",i);
	op(dest.elem(outersite), forEach(rhs, EvalLeaf3(0, low >> INNER_LOG, high >> INNER_LOG), OpCombine()));
      }
    }
  }
#else
  QDP_error("evaluateSubset not implemented");
#endif

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif
}


//! OLattice Op OLattice(Expression(source)) under an Subset
/*! 
 * OLattice Op Expression, where Op is some kind of binary operation 
 * involving the destination 
 */
template<class T, class T1, class Op, class RHS>
//inline
void evaluate(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs,
	      const Subset& s)
{
  int num_threads = 1;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest, op, rhs);
  prof.time -= getClockTime();
#endif

#if ! defined(QDP_NOT_IMPLEMENTED)
  // General form of loop structure
  if (s.hasOrderedRep()) {
    const int istart = s.start() >> INNER_LOG;
    const int iend   = s.end()   >> INNER_LOG;

#pragma omp parallel shared(istart, iend, num_threads) default(shared)
    {
      num_threads = omp_get_num_threads (); 
      int myId = omp_get_thread_num();
      int low = istart + (iend - istart + 1) * myId/num_threads;
      int high = istart + (iend - istart + 1) * (myId + 1)/num_threads;

      for(int i = low; i < high; ++i) {
	// QDP_info("Thread %d: eval(olattice,olattice): site %d low %d high %d\n", myId, i, low, high);
	op(dest.elem(i), forEach(rhs, EvalLeaf3(i, low, high - 1), OpCombine()));
      }
    }
  }
  else {
    // this part of code need to test
    const int *tab = s.siteTable().slice();

#pragma omp parallel shared(tab, num_threads) default(shared) 
    {
      num_threads = omp_get_num_threads ();
      int myId = omp_get_thread_num();
      int low = s.numSiteTable() * myId/num_threads;
      int high = s.numSiteTable() * (myId + 1)/num_threads;
      
      for(int j = low; j < high; ++j) {
	int i = tab[j];
	int outersite = i >> INNER_LOG;
	int innersite = i & ((1 << INNER_LOG)-1);

	//    fprintf(stderr,"eval(olattice,olattice): site %d\n",i);
	op(dest.elem(outersite), forEach(rhs,
					 EvalLeaf3(outersite, low >> INNER_LOG, high >> INNER_LOG),
					 OpCombine()));
      }
    }
  }  
#else
  QDP_error("evaluateSubset not implemented");
#endif

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif
}


//-----------------------------------------------------------------------------
//! dest = (mask) ? s1 : dest
template<class T1, class T2> 
void 
copymask(OSubLattice<T2> d, const OLattice<T1>& mask, const OLattice<T2>& s1) 
{
  OLattice<T2>& dest = d.field();
  const Subset& s = d.subset();

#if ! defined(QDP_NOT_IMPLEMENTED)
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    copymask(dest.elem(i), mask.elem(i), s1.elem(i));
  }
#else
  QDP_error("copymask_Subset not implemented");
#endif
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
random(OLattice<T>& d, const Subset& s)
{
  Seed seed;
#if 0
  Seed skewed_seed;
#endif
  ILatticeSeed skewed_seed;

#if ! defined(QDP_NOT_IMPLEMENTED)
#warning "random(unorderedsubset) broken"
  if (s.hasOrderedRep()) {
    const int istart = s.start() >> INNER_LOG;
    const int iend   = s.end()   >> INNER_LOG;

    for(int i=istart; i <= iend; ++i) {
      seed = RNG::ran_seed;
      // Jie Chen: error here. IScalar = ILattice (?)
      skewed_seed.elem() = RNG::ran_seed.elem() * RNG::lattice_ran_mult->elem(i);
      fill_random(d.elem(i), seed, skewed_seed, RNG::ran_mult_n);
    }
  }
  else {
   const int *tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); ++j) {
      int i = tab[j];
      int outersite = i >> INNER_LOG;
      int innersite = i & ((1 << INNER_LOG)-1);

      seed = RNG::ran_seed;
      // Jie Chen: error here. IScalar = ILattice (?)
      skewed_seed.elem() = RNG::ran_seed.elem() * RNG::lattice_ran_mult->elem(outersite);
      fill_random(d.elem(outersite), seed, skewed_seed, RNG::ran_mult_n);
    }

  }

  RNG::ran_seed = seed;  // The seed from any site is the same as the new global seed
#else
  QDP_error("random_Subset not implemented");
#endif
}


//! dest  = random   under a subset
template<class T, class S>
void random(const OSubLattice<T>& dd)
{
  OLattice<T>& d = const_cast<OSubLattice<T>&>(dd).field();
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
void gaussian(OLattice<T>& d, const Subset& s)
{
  OLattice<T>  r1, r2;

  random(r1,s);
  random(r2,s);

#if ! defined(QDP_NOT_IMPLEMENTED)
  if (s.hasOrderedRep()) {
    const int istart = s.start() >> INNER_LOG;
    const int iend   = s.end()   >> INNER_LOG;

    for(int i=istart; i <= iend; ++i) 
      fill_gaussian(d.elem(i), r1.elem(i), r2.elem(i));
  }
  else {
    const int *tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); ++j) {
      int i = tab[j];
      int outersite = i >> INNER_LOG;
      int innersite = i & ((1 << INNER_LOG)-1);

      fill_gaussian(d.elem(outersite), r1.elem(outersite), r2.elem(outersite));
    }
  }
#else
  QDP_error("gaussianSubset not implemented");
#endif
}



//! dest  = gaussian   under a subset
template<class T, class S>
void gaussian(const OSubLattice<T>& dd)
{
  OLattice<T>& d = const_cast<OSubLattice<T>&>(dd).field();
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
void zero_rep(OLattice<T>& dest, const Subset& s) 
{
#if ! defined(QDP_NOT_IMPLEMENTED)
  if (s.hasOrderedRep()) {
    const int istart = s.start() >> INNER_LOG;
    const int iend   = s.end()   >> INNER_LOG;

    for(int i=istart; i <= iend; ++i)
      zero_rep(dest.elem(i));
  }
  else {
    const int *tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); ++j) 
      {
	int i = tab[j];
	int outersite = i >> INNER_LOG;
	zero_rep(dest.elem(outersite));
      }
  }
#else
  QDP_error("zero_rep_Subset not implemented");
#endif
}



//! dest  = 0 
template<class T>
void zero_rep(OSubLattice<T> dd) 
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

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  evaluate(d,OpAssign(),s1,all);

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

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

#if defined(QDP_USE_PROFILING)  
  static QDPProfile_t prof(d, OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  evaluate(d,OpAssign(),s1,all);   // since OScalar, no global sum needed

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}



//! OScalar = sum(OLattice)  under an explicit subset
/*!
 * Allow a global sum that sums over the lattice, but returns an object
 * of the same primitive type. E.g., contract only over lattice indices
 *
 * This will include a parent Subset and an Subset.
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
  int num_threads = 1;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // Must initialize to zero since we do not know if the loop will be entered
  zero_rep(d.elem());

  if (s.hasOrderedRep())
  {
    const int istart = s.start() >> INNER_LOG;
    const int iend   = s.end()   >> INNER_LOG;

    typename UnaryReturn<OLattice<T>, FnSum>::Type_t  *pd = new  typename UnaryReturn<OLattice<T>, FnSum>::Type_t[qdpNumThreads()]; 

#pragma omp parallel shared(pd,istart,iend,num_threads) private(tmp) default(shared)
    {
      num_threads = omp_get_num_threads();
      int myId = omp_get_thread_num();
      int low = istart + (iend - istart + 1) * myId/num_threads;
      int high = istart + (iend - istart + 1) * (myId + 1)/num_threads;
      // zero the partial result
      zero_rep(pd[myId].elem());

      for(int i=low; i < high; ++i) {
	tmp.elem() = forEach(s1, EvalLeaf1(i), OpCombine()); // Evaluate to ILattice part
	pd[myId].elem() += sum(tmp.elem()); // sum as well the ILattice part
      }
    }
    // Now combine all together
    for (int j = 0; j < num_threads; j++) 
      d.elem() += sum(pd[j].elem());

    delete []pd;
  }
  else
  {
    typename UnaryReturn<OLattice<T>, FnSum>::Type_t  *pd = new  typename UnaryReturn<OLattice<T>, FnSum>::Type_t[qdpNumThreads()]; 
    const int *tab = s.siteTable().slice();

#pragma omp parallel shared(pd,num_threads) private(tmp) default(shared) 
    {
      num_threads = omp_get_num_threads();
      int myId = omp_get_thread_num();
      int low = s.numSiteTable() * myId/num_threads;
      int high = s.numSiteTable() * (myId + 1)/num_threads;

      // zero the partial result
      zero_rep(pd[myId].elem());

      for(int j=low; j < high; ++j) {
	int i = tab[j];
	int outersite = i >> INNER_LOG;
	int innersite = i & (INNER_LEN-1);

	tmp.elem() = forEach(s1, EvalLeaf1(outersite), OpCombine()); // Evaluate to ILattice part
	pd[myId].elem() += getSite(tmp.elem(),innersite);    // wasteful - only extract a single site worth
      }
    }
    // Now combine all together
    for (int k = 0; k < num_threads; k++) 
      d.elem() += sum(pd[k].elem());

    delete []pd;
  }

  // Do a global sum on the result
  QDPInternal::globalSum(d);
  
#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
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

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest[0], OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // lazy - evaluate repeatedly
  for(int i=0; i < ss.numSubsets(); ++i)
    dest[i] = sum(s1,ss[i]);
  
#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

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

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest[0], OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // lazy - evaluate repeatedly
  for(int i=0; i < ss.numSubsets(); ++i)
    dest[i] = sum(s1,ss[i]);

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return dest;
}


//-----------------------------------------------------------------------------
// Multiple global sums 
//! multi2d<OScalar> dest  = sumMulti(multi1d<OScalar>,Set) 
/*!
 * Compute the global sum on multiple subsets specified by Set 
 *
 * This implementation is specific to a purely olattice like
 * types. The scalar input value is replicated to all the
 * slices
 */
template<class T>
multi2d<typename UnaryReturn<OScalar<T>, FnSum>::Type_t>
sumMulti(const multi1d< OScalar<T> >& s1, const Set& ss)
{
  multi2d<typename UnaryReturn<OScalar<T>, FnSum>::Type_t>  dest(s1.size(), ss.numSubsets());

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest(0,0), OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // lazy - evaluate repeatedly
  for(int i=0; i < dest.size1(); ++i)
    for(int j=0; j < dest.size2(); ++j)
      dest(j,i) = s1[j];

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return dest;
}


//! multi2d<OScalar> dest  = sumMulti(multi1d<OLattice>,Set) 
/*!
 * Compute the global sum on multiple subsets specified by Set 
 *
 * This is a very simple implementation. There is no need for
 * anything fancier unless global sums are just so extraordinarily
 * slow. Otherwise, generalized sums happen so infrequently the slow
 * version is fine.
 */
template<class T>
multi2d<typename UnaryReturn<OLattice<T>, FnSum>::Type_t>
sumMulti(const multi1d< OLattice<T> >& s1, const Set& ss)
{
  multi2d<typename UnaryReturn<OLattice<T>, FnSum>::Type_t>  dest(s1.size(),ss.numSubsets());

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest(0,0), OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // lazy - evaluate repeatedly
  for(int k=0; k < s1.size(); ++k)
    for(int i=0; i < ss.numSubsets(); ++i)
      dest(k,i) = sum(s1[k],ss[i]);

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return dest;
}


//-----------------------------------------------------------------------------
//! OScalar = norm2(trace(adj(multi1d<source>)*multi1d<source>))
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T>
inline typename UnaryReturn<OScalar<T>, FnNorm2>::Type_t
norm2(const multi1d< OScalar<T> >& s1)
{
  typename UnaryReturn<OScalar<T>, FnNorm2>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnNorm2(), s1[0]);
  prof.time -= getClockTime();
#endif

  // Possibly loop entered
  zero_rep(d.elem());

  for(int n=0; n < s1.size(); ++n)
  {
    OScalar<T>& ss1 = s1[n];
    d.elem() += localNorm2(ss1.elem());
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}

//! OScalar = sum(OScalar)  under an explicit subset
/*! Discards subset */
template<class T>
inline typename UnaryReturn<OScalar<T>, FnNorm2>::Type_t
norm2(const multi1d< OScalar<T> >& s1, const Subset& s)
{
  return norm2(s1);
}


//! OScalar = norm2(multi1d<OLattice>) under an explicit subset
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T>
inline typename UnaryReturn<OLattice<T>, FnNorm2>::Type_t
norm2(const multi1d< OLattice<T> >& s1, const Subset& s)
{
  typename UnaryReturn<OLattice<T>, FnNorm2>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnNorm2(), s1[0]);
  prof.time -= getClockTime();
#endif

  // Possibly loop entered
  zero_rep(d.elem());

#if ! defined(QDP_NOT_IMPLEMENTED)
  if (s.hasOrderedRep()) {
    const int istart = s.start() >> INNER_LOG;
    const int iend   = s.end()   >> INNER_LOG;

    for(int n=0; n < s1.size(); ++n) {
      const OLattice<T>& ss1 = s1[n];

      for (int i = istart; i <= iend; ++i) 
	d.elem() += localNorm2(ss1.elem(i));
    }
  }
  else {
    const int *tab = s.siteTable().slice();
    for(int n=0; n < s1.size(); ++n)
      {
	const OLattice<T>& ss1 = s1[n];
	for(int j=0; j < s.numSiteTable(); ++j) {
	  int i = tab[j];
	  int outersite = i >> INNER_LOG;
	  int innersite = i & ((1 << INNER_LOG)-1);
	  d.elem() += localNorm2(ss1.elem(outersite));
	}
      }
  }
#else
  QDP_error_exit("norm2-Subset not implemented");
#endif

  // Do a global sum on the result
  QDPInternal::globalSum(d);
  
#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}



//! OScalar = norm2(multi1d<OLattice>)
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T>
inline typename UnaryReturn<OLattice<T>, FnNorm2>::Type_t
norm2(const multi1d< OLattice<T> >& s1)
{
  return norm2(s1,all);
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
template<class RHS, class T1>
inline OScalar<T1>
peekSite(const QDPExpr<RHS,OScalar<T1> > & l, const multi1d<int>& coord)
{
  // For now, simply evaluate the expression and then call the function
  typedef OScalar<T1> C1;
  
  return peekSite(C1(l), coord);
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
  QDPInternal::sendToPrimaryNode(dest, nodenum);

  // Now broadcast back out to all nodes
  QDPInternal::broadcast(dest);

  return dest;
}

//! Extract site element
/*! @ingroup group1
  @param l  source to examine
  @param coord Nd lattice coordinates to examine
  @return single site object of the same primitive type
  @ingroup group1
  @relates QDPType */
template<class RHS, class T1>
inline OScalar<T1>
peekSite(const QDPExpr<RHS,OLattice<T1> > & l, const multi1d<int>& coord)
{
  // For now, simply evaluate the expression and then call the function
  typedef OLattice<T1> C1;
  
  return peekSite(C1(l), coord);
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
//-----------------------------------------------------------------------------
// Empty map

// Forward Decleration of FnMap
struct FnMap;


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
   * Notice, this implementation supports an inner grid
   */
  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPType<T1,C1> & l)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(this),
	CreateLeaf<QDPType<T1,C1> >::make(l)));
    }


  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPExpr<T1,C1> & l)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(this),
	CreateLeaf<QDPExpr<T1,C1> >::make(l)));
    }

public:
  //! Accessor to offsets
  const multi1d<int>& goffset() const {return goffsets;}
  const multi1d<int>& soffset() const {return soffsets;}
  // offnode
  bool offnodeComm (void) const {return offnodeP;}
  // number of destination sites
  int numDestSites (void) const {return destnodes_num[0];}
  // number of source sites
  int numSrcSites (void) const {return srcenodes_num[0];}
  // source nodes
  int qmpsrcNode() const {return srcenodes[0];}
  // dest nodes
  int qmpdestNode() const {return destnodes[0];}

  // check whether a site is a recieving surface site
  int isReceivingSite (int rsite) const 
  {
    const int my_node = Layout::nodeNumber();
    if (srcnode[rsite] != my_node)
      return 1;

    return 0;
  }

  // check whether a site is on a surface
  // this site may not send anything, but we pack data anyway
  int isSendingSite (int ssite) const 
  {
    if (send_pos[ssite] != -1)
      return 1;

    return 0;
  }

  // return a right receiving position for a surface site
  int receivingPosition (int rsite) const
  {
    return recv_pos[rsite];
  }

  // return a right sending position inside send buffer for a surface site
  int sendingPosition (int ssite) const
  {
    return send_pos[ssite];
  }

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

  // a surface receiving site position to receive data
  multi1d<int> recv_pos;

  // a surface sending site position inside the send buffer
  multi1d<int> send_pos;

  // Indicate off-node communications is needed;
  bool offnodeP;
};


struct FnMap
{
  PETE_EMPTY_CONSTRUCTORS(FnMap)
  
  // Internel pointer for a real map
  const Map *map;
  
  // shift direction and sign
  int isign, dir;

  // dumb constructor for fnmap: make compiler happy
  FnMap (const Map* m)
    :map (m), isign (1), dir (0)
  {
    // empty
  }

  FnMap (const Map* m, int sign, int d)
    : map (m), isign (sign), dir (d)
  {
    // empty
  }

  template<class T>
  inline typename UnaryReturn<T, FnMap>::Type_t
  operator()(const T &a) const
  {
    return (a);
  }
};

#if defined(QDP_USE_PROFILING)   
template <>
struct TagVisitor<FnMap, PrintTag> : public ParenPrinter<FnMap>
{ 
  static void visit(FnMap op, PrintTag t) 
    { t.os_m << "shift"; }
};
#endif


//----------------------------------------------------------------------------
// Leaf functor containing an array of offsets, map and send recv buffers
// First 3 integers are inherited from EvalLeaf3
//----------------------------------------------------------------------------
struct EvalLeaf5Comm
{
  int i1_m, i2_m, i3_m, i4_m, i5_m;
  const Map *map_;

  inline EvalLeaf5Comm (int i1, int i2, int i3, 
			int i4, int i5,
			const Map* m)
    : i1_m(i1), i2_m(i2), i3_m(i3), i4_m(i4), i5_m(i5),
      map_(m) { }
  inline int val1() const { return i1_m; }
  inline int val2() const { return i2_m; }
  inline int val3() const { return i3_m; }
  inline int val4() const { return i4_m; }
  inline int val5() const { return i5_m; }
  inline const Map* map() const { return map_; }
};


//-----------------------------------------------------------------------------
// Specialization of LeafFunctor class for applying the EvalLeaf5Comm
// tag to a QDPType. The apply method simply returns the array
// evaluated at the point.
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Currently, x-direction is the direction that is vectorized. Any shift
// along the other directions can be easilly done by using nearest neighbor
// vectorized data. For shifting along x direction, we need to reconstruct
// a new element containing inner-data
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//      Gather data for x direction
//-----------------------------------------------------------------------------
template <class T, class C>
void gather_along_x (const QDPType<T, C>& a, int old_inner_site, 
		     const int* offsets,
		     T& ret)
{
  int i = old_inner_site;
#if INNER_LOG == 2
  int o0 = offsets[i+0] >> INNER_LOG;
  int i0 = offsets[i+0] & (INNER_LEN - 1);

  int o1 = offsets[i+1] >> INNER_LOG;
  int i1 = offsets[i+1] & (INNER_LEN - 1);
    
  int o2 = offsets[i+2] >> INNER_LOG;
  int i2 = offsets[i+2] & (INNER_LEN - 1);

  int o3 = offsets[i+3] >> INNER_LOG;
  int i3 = offsets[i+3] & (INNER_LEN - 1);

  // Gather 4 inner-grid sites together
  gather_sites(ret,
	       a.elem(o0),i0,
	       a.elem(o1),i1,
	       a.elem(o2),i2,
	       a.elem(o3),i3);

#elif INNER_LOG == 3
  int o0 = offsets[i+0] >> INNER_LOG;
  int i0 = offsets[i+0] & (INNER_LEN - 1);
  
  int o1 = offsets[i+1] >> INNER_LOG;
  int i1 = offsets[i+1] & (INNER_LEN - 1);
  
  int o2 = offsets[i+2] >> INNER_LOG;
  int i2 = offsets[i+2] & (INNER_LEN - 1);
  
  int o3 = offsets[i+3] >> INNER_LOG;
  int i3 = offsets[i+3] & (INNER_LEN - 1);

  int o4 = offsets[i+4] >> INNER_LOG;
  int i4 = offsets[i+4] & (INNER_LEN - 1);
  
  int o5 = offsets[i+5] >> INNER_LOG;
  int i5 = offsets[i+5] & (INNER_LEN - 1);
      
  int o6 = offsets[i+6] >> INNER_LOG;
  int i6 = offsets[i+6] & (INNER_LEN - 1);

  int o7 = offsets[i+7] >> INNER_LOG;
  int i7 = offsets[i+7] & (INNER_LEN - 1);
  gather_sites (ret,
		a.elem(o0),i0,
		a.elem(o1),i1,
		a.elem(o2),i2,
		a.elem(o3),i3,
		a.elem(o4),i4,
		a.elem(o5),i5,
		a.elem(o6),i6,
		a.elem(o7),i7);

#elif INNER_LOG == 4
  int o0 = offsets[i+0] >> INNER_LOG;
  int i0 = offsets[i+0] & (INNER_LEN - 1);

  int o1 = offsets[i+1] >> INNER_LOG;
  int i1 = offsets[i+1] & (INNER_LEN - 1);
      
  int o2 = offsets[i+2] >> INNER_LOG;
  int i2 = offsets[i+2] & (INNER_LEN - 1);
    
  int o3 = offsets[i+3] >> INNER_LOG;
  int i3 = offsets[i+3] & (INNER_LEN - 1);

  int o4 = offsets[i+4] >> INNER_LOG;
  int i4 = offsets[i+4] & (INNER_LEN - 1);

  int o5 = offsets[i+5] >> INNER_LOG;
  int i5 = offsets[i+5] & (INNER_LEN - 1);

  int o6 = offsets[i+6] >> INNER_LOG;
  int i6 = offsets[i+6] & (INNER_LEN - 1);

  int o7 = offsets[i+7] >> INNER_LOG;
  int i7 = offsets[i+7] & (INNER_LEN - 1);

  int o8 = offsets[i+8] >> INNER_LOG;
  int i8 = offsets[i+8] & (INNER_LEN - 1);
      
  int o9 = offsets[i+9] >> INNER_LOG;
  int i9 = offsets[i+9] & (INNER_LEN - 1);

  int o10 = offsets[i+10] >> INNER_LOG;
  int i10 = offsets[i+10] & (INNER_LEN - 1);

  int o11 = offsets[i+11] >> INNER_LOG;
  int i11 = offsets[i+11] & (INNER_LEN - 1);

  int o12 = offsets[i+12] >> INNER_LOG;
  int i12 = offsets[i+12] & (INNER_LEN - 1);

  int o13 = offsets[i+13] >> INNER_LOG;
  int i13 = offsets[i+13] & (INNER_LEN - 1);

  int o14 = offsets[i+14] >> INNER_LOG;
  int i14 = offsets[i+14] & (INNER_LEN - 1);

  int o15 = offsets[i+15] >> INNER_LOG;
  int i15 = offsets[i+15] & (INNER_LEN - 1);

  // Gather 16 inner-grid sites together
  gather_sites(ret,
	       a.elem(o0),i0,
	       a.elem(o1),i1,
	       a.elem(o2),i2,
	       a.elem(o3),i3,
	       a.elem(o4),i4,
	       a.elem(o5),i5,
	       a.elem(o6),i6,
	       a.elem(o7),i7,
	       a.elem(o8),i8,
	       a.elem(o9),i9,
	       a.elem(o10),i10,
	       a.elem(o11),i11,
	       a.elem(o12),i12,
	       a.elem(o13),i13,
	       a.elem(o14),i14,
	       a.elem(o15),i15);
#else
#error "Map: shift positive this inner grid length is not supported - easy to fix"
#endif
}

//-----------------------------------------------------------------------------
//  Intra-node shift for olattice 
//-----------------------------------------------------------------------------
template<class T, class C>
T intra_shift (const QDPType<T, C>& a, const EvalLeaf5Comm& f)
{
  const int *goffsets = f.map()->goffset().slice();

  if (f.val5() != 0) { // not along x direction
    // f.val1() is an old outer site, convert to new neighbor real site
    int oldrealsite = f.val1() << INNER_LOG;
    int newrealsite = goffsets[oldrealsite];
    // Y, Z, T neighbors are boundled together
    return a.elem(newrealsite >> INNER_LOG);
  }
  else {
    T ret;
    // old real site 
    int i = f.val1() << INNER_LOG;

    gather_along_x (a, i, goffsets, ret);

    return ret;
  }
}

template<class T>
T intra_shift (const OLattice<T>& a, const EvalLeaf5Comm& f)
{
  return intra_shift<T, OLattice<T> > ((const QDPType<T, OLattice<T> >)a, f);
}

//-----------------------------------------------------------------------------
//  Inter-node shift for olattice for a single site
//-----------------------------------------------------------------------------
template<class T, class C>
T inter_shift (const QDPType<T, C>& a, const EvalLeaf5Comm& f)
{
  const Map *map = f.map();
  const int *goffsets = map->goffset().slice();
  const int *soffsets = map->soffset().slice();

  /* figure out which part of send buffer this thread is working on */
  int tid = omp_get_thread_num ();
  int num_threads = omp_get_num_threads ();

  // send and receiving buffer of surface
  T *send_buf = 0;
  T *recv_buf = 0;

  QMP_msgmem_t msg[2];
  QMP_msghandle_t mh_a[2], mh;
  QMP_status_t err;

  // send and recv buffer now is associated with olattice<T>
  QMP_mem_t* send_buf_mem = a.sendBufMem(f.val5(), f.val4());
  QMP_mem_t* recv_buf_mem = a.recvBufMem(f.val5(), f.val4());
  int recv_buf_size = a.recvBufMemSize(f.val5(), f.val4());
  int send_buf_size = a.sendBufMemSize(f.val5(), f.val4());

  /**
   * Build up send buffer by those sites that are on surface sending
   * surface data
   */
  send_buf=(T *)QMP_get_memory_pointer(send_buf_mem);
  if( send_buf == 0x0 ) { 
    QDP_error_exit("QMP_get_memory_pointer returned NULL pointer from non NULL QMP_mem_t (send_buf)\n");
  }	


  recv_buf=(T *)QMP_get_memory_pointer(recv_buf_mem);
  if (recv_buf == 0x0) { 
    QDP_error_exit("QMP_get_memory_pointer returned NULL pointer from non NULL QMP_mem_t (recv_buf)\n");
  }


  if (f.val5() != 0) { 
    // not along x direction. data are packed togethr
    if (f.val1() == f.val2()) {
      // since we are dealing with a thread, so we fill the send buffer
      // in the beginning of the iteration in the thread
      // we are working with outer sites
      int size = map->soffset().size() >> INNER_LOG;
      int low, high;
      low = size * tid /num_threads;
      high = size * (tid + 1) /num_threads;

      for (int si = low; si < high; si++) {
	// original site
	int srcsite = soffsets[si << INNER_LOG];
	int pos = map->sendingPosition(srcsite) >> INNER_LOG;
	send_buf[pos] = a.elem (srcsite >> INNER_LOG);
      }
    }
  }
  else { // along x direction so we have more sites
    if (f.val1() == f.val2()) {
      // we are working with real sites
      int size = map->soffset().size();
      int low, high;
      low = size * tid /num_threads;
      high = size * (tid + 1)/num_threads;

      for (int si = low; si < high; si++) {
	// original site
	int srcsite = soffsets[si];
	int pos = map->sendingPosition(srcsite) >> INNER_LOG;
	send_buf[pos] = a.elem (srcsite >> INNER_LOG);
      }
    }
  }
    
  /**
   * We have to wait for every thread to finish populate the send buffer
   * We do this on the last iteration of this thread
   */
  if (f.val1() == f.val2()) {
#pragma omp barrier
    ;
  }

  if (tid == 0 && f.val1() == f.val2() ) { // send and receive surface
    // do this for the first site
    // f.val2() is starting point of the lattice site within this thread

    // now start sending and receiving data
#if QDP_DEBUG >= 3
    QDP_info("Map: send = 0x%x  recv = 0x%x",send_buf,recv_buf);
#endif

    msg[0] = QMP_declare_msgmem(recv_buf, recv_buf_size);
    if( msg[0] == (QMP_msgmem_t)NULL ) { 
      QDP_error_exit("QMP_declare_msgmem for msg[0] failed in Map::operator()\n");
    }
    msg[1]  = QMP_declare_msgmem(send_buf, send_buf_size);
    if( msg[1] == (QMP_msgmem_t)NULL ) {
      QDP_error_exit("QMP_declare_msgmem for msg[1] failed in Map::operator()\n");
    }

    mh_a[0] = QMP_declare_receive_from(msg[0], map->qmpsrcNode(), 0);
    if( mh_a[0] == (QMP_msghandle_t)NULL ) { 
      QDP_error_exit("QMP_declare_receive_from for mh_a[0] failed in Map::operator()\n");
    }

    mh_a[1] = QMP_declare_send_to(msg[1], map->qmpdestNode(), 0);
    if( mh_a[1] == (QMP_msghandle_t)NULL ) {
      QDP_error_exit("QMP_declare_send_to for mh_a[1] failed in Map::operator()\n");
    }

    mh = QMP_declare_multiple(mh_a, 2);
    if( mh == (QMP_msghandle_t)NULL ) { 
      QDP_error_exit("QMP_declare_multiple for mh failed in Map::operator()\n");
    }


#if 0
    // Launch the faces
    QDP::StopWatch swatch;

    swatch.reset();
    swatch.start();
#endif
    if ((err = QMP_start(mh)) != QMP_SUCCESS)
      QDP_error_exit(QMP_error_string(err));

#if 0
    swatch.stop();
    QDP_info ("Thread %d qmp_start takes %lf us\n", tid, swatch.getTimeInMicroseconds());
#endif
  }

  // get the site number we are working on 
  // f.val1() is an old outer site
  int oldrealsite = f.val1() << INNER_LOG;
  T ret;

  if (!(f.map()->isReceivingSite(oldrealsite))) {
    if (f.val5() != 0) { // not along x direction
      int newrealsite = goffsets[oldrealsite];
      // Y, Z, T neighbors are boundled together
      ret = a.elem(newrealsite >> INNER_LOG);
    }
    else {
      // old real site 
      gather_along_x (a, oldrealsite, goffsets, ret);
    }
  }

  if (tid == 0 && f.val1() == f.val2()) {
#if QDP_DEBUG >= 3
    QDP_info("Map: calling wait");
#endif

#if 0
    // Wait on the faces
    QDP::StopWatch swatch;

    swatch.reset();
    swatch.start();
#endif

    if ((err = QMP_wait(mh)) != QMP_SUCCESS)
      QDP_error_exit(QMP_error_string(err));

#if 0
    swatch.stop();
    QDP_info ("Thread %d qmp_wait takes %lf us\n", tid, swatch.getTimeInMicroseconds());
#endif

#if QDP_DEBUG >= 3
    QDP_info("Map: calling free msgs");
#endif
  
    QMP_free_msghandle(mh);
    QMP_free_msgmem(msg[1]);
    QMP_free_msgmem(msg[0]);
  }


  if (f.val1() == f.val2() ) {
#pragma omp barrier
    ;
  }

  // now surface sites get data from receiving buffer
  if ((f.map()->isReceivingSite(oldrealsite))) {
    // data receiving buffer
    recv_buf=(T *)QMP_get_memory_pointer(recv_buf_mem);
    if (recv_buf == 0x0) { 
      QDP_error_exit("QMP_get_memory_pointer returned NULL pointer from non NULL QMP_mem_t (recv_buf)\n");
    }

    if (f.val5() != 0) { // not along x direction
      // get receiving position
      int recv_pos = f.map()->receivingPosition (oldrealsite);

      //      QDP_info ("thread %d: receiving surface site = %d recvelem = (%d)\n", 		tid, oldrealsite, recv_pos >> INNER_LOG);

      // data are sorted according to receiving site
      ret = recv_buf[recv_pos >> INNER_LOG];

    }
    else { // along x direction: unlikely
      ret = recv_buf[oldrealsite];
    }
  }
  return ret;
}

template<class T>
T inter_shift (const OLattice<T>& a, const EvalLeaf5Comm& f)
{
  return inter_shift<T, OLattice<T> > ((const QDPType<T, OLattice<T> >)a, f);
}

template<class T, class C>
struct LeafFunctor<QDPType<T,C>, EvalLeaf5Comm>
{
  typedef Reference<T> Type_t;
//  typedef T Type_t;
  inline static Type_t apply(const QDPType<T,C> &a, const EvalLeaf5Comm &f)
  { 
    const Map *map = f.map();

    if (map->offnodeComm() ) {
      return (Type_t)(inter_shift(a, f));
    }
    else 
      return (Type_t)(intra_shift(a, f));
  }
};

template<class T>
struct LeafFunctor<OScalar<T>, EvalLeaf5Comm>
{
//  typedef T Type_t;
  typedef Reference<T> Type_t;
  inline static Type_t apply(const OScalar<T> &a, const EvalLeaf5Comm &f)
    {return Type_t(a.elem());}
};

template<class T>
struct LeafFunctor<OLattice<T>, EvalLeaf5Comm>
{
//  typedef T Type_t;
  typedef Reference<T> Type_t;

  inline static Type_t apply(const OLattice<T> &a, const EvalLeaf5Comm &f)
  {
    const Map *map = f.map();

    if (map->offnodeComm() )
      return (Type_t)(inter_shift(a, f));
    else 
      return (Type_t)(intra_shift(a, f));
  }
};

// Specialization of ForEach deals with maps. 
template<class A, class CTag>
struct ForEach<UnaryNode<FnMap, A>, EvalLeaf3, CTag>
{
  typedef typename ForEach<A, EvalLeaf5Comm, CTag>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnMap, CTag>::Type_t Type_t;
  inline static
  Type_t apply(const UnaryNode<FnMap, A> &expr, const EvalLeaf3 &f, 
    const CTag &c) 
  {
    // expr.operation() is fnmap which contains offsets, sign and direction
    // f.val1() is pointing to outersites: check evaluate function above:
    // --Jie Chen
    int isign = expr.operation().isign;
    int dir = expr.operation().dir;

    // passing offsets pointer as well
    EvalLeaf5Comm ff(f.val1(), f.val2(), f.val3(), 
		     isign, dir, expr.operation().map);
    
    //    fprintf(stderr,"ForEach<Unary<FnMap>>: outersite = %d, sign = %d dir = %d\n", ff.val1(), ff.val4(), ff.val5());
    //    fprintf(stderr, "%s\n", __PRETTY_FUNCTION__);
    return Combine1<TypeA_t, FnMap, CTag>::
      combine(ForEach<A, EvalLeaf5Comm, CTag>::apply(expr.child(), ff, c),
              expr.operation(), c);
  }
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
  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPType<T1,C1> & l, int dir)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(&(mapsa[dir])),
	CreateLeaf<QDPType<T1,C1> >::make(l)));
    }


  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPExpr<T1,C1> & l, int dir)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(&(mapsa[dir])),
	CreateLeaf<QDPExpr<T1,C1> >::make(l)));
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
  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPType<T1,C1> & l, int isign)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(&(bimaps[(isign+1)>>1])),
	     CreateLeaf<QDPType<T1,C1> >::make(l)));
    }


  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPExpr<T1,C1> & l, int isign)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(&(bimaps[(isign+1)>>1])),
	CreateLeaf<QDPExpr<T1,C1> >::make(l)));
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
#if 1
  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPType<T1,C1> & l, int isign, int dir)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(&(bimapsa((isign+1)>>1,dir)), isign, dir),
	CreateLeaf<QDPType<T1,C1> >::make(l)));
    }

  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPExpr<T1,C1> & l, int isign, int dir)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(&(bimapsa((isign+1)>>1,dir)), isign, dir),
	CreateLeaf<QDPExpr<T1,C1> >::make(l)));
    }
#endif


#if 0
  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPType<T1,C1> & l, int isign, int dir)
    {
      Map *map = &(bimapsa((isign+1)>>1,dir));
      SurfaceCommBuf* commBuf = 0;
      
      if (map->offnodeComm()) {
	commBuf = new SurfaceCommBuf();
      }

      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(&(bimapsa((isign+1)>>1,dir)), isign, dir, commBuf),
	CreateLeaf<QDPType<T1,C1> >::make(l)));
    }


  template<class T1>
  OScalar<T1>
  operator()(const OScalar<T1> & l, int isign, int dir)
  {
    return l;
  }

  template<class RHS,class T1>
  inline typename MakeReturn<UnaryNode<FnMap,
  typename CreateLeaf<OLattice<T1> >::Leaf_t>, T1>::Expression_t
  operator()(const QDPExpr<RHS, OScalar<T1> > & l, int isign, int dir)
  {
    typedef OScalar<T1> C1;
      
    return C1(l);
  }

  template<class RHS,class T1>
  inline typename MakeReturn<UnaryNode<FnMap,
  typename CreateLeaf<OLattice<T1> >::Leaf_t>, OLattice<T1> >::Expression_t
  operator()(const QDPExpr<RHS,OLattice<T1> > & l, int isign, int dir)
  {
    Map *map = &(bimapsa((isign+1)>>1,dir));
    SurfaceCommBuf* commBuf = 0;

    if (map->offnodeComm()) {
      commBuf = new SurfaceCommBuf();
    }

    typedef UnaryNode<FnMap,
      typename CreateLeaf<OLattice<T1> >::Leaf_t> Tree_t;
    return MakeReturn<Tree_t,OLattice<T1> >::make(Tree_t(FnMap(&(bimapsa((isign+1)>>1,dir)), isign, dir, commBuf),
	 CreateLeaf<OLattice<T1> >::make( *tmp )));
  }
#endif

  /**
   * Check whether there are communication needed for direction and sign
   * @param dir: 0, 1, 2, 3
   * @param isign: +1, -1
   */
  bool offnodeComm (int dir, int isign) const
  {
    return bimapsa((isign + 1) >> 1, dir).offnodeComm();
  }

  /**
   * Return number of sending sites along a direction
   * with either positive or negative sending or receiving
   * @param dir: 0, 1, 2, 3
   * @param isign: +1, -1
   */
  int numberSendingSites (int dir, int isign)
  {
    return bimapsa((isign + 1) >> 1, dir).numDestSites ();
  }

  /**
   * Return number of receiving sites along a direction
   * with either positive or negative sending or receiving
   * @param dir: 0, 1, 2, 3
   * @param isign: +1, -1
   */
  int numberRecvingSites (int dir, int isign)
  {
    return bimapsa((isign + 1) >> 1, dir).numSrcSites ();
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



// There are 2 main classes of binary/xml reader/writer methods.
// The first is a simple/portable but inefficient method of send/recv
// to/from the destination node.
// The second method (the else) is a more efficient roll-around method.
// However, this method more constrains the data layout - it must be
// close to the original lexicographic order.
// For now, use the direct send method

//! Decompose a lexicographic site into coordinates
multi1d<int> crtesn(int ipos, const multi1d<int>& latt_size);

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
#if 1
      // All nodes participate
      QDPInternal::route((void *)&recv_buf, node, 0, sizeof(Site_t));
#else
      if (Layout::primaryNode())
	QDPInternal::recvFromWait((void *)&recv_buf, node, sizeof(Site_t));

      if (Layout::nodeNumber() == node)
	QDPInternal::sendToWait((void *)&recv_buf, 0, sizeof(Site_t));
#endif
    }

    if (Layout::primaryNode())
    {
      std::ostringstream os;
      os << coord[0];
      for(int i=1; i < coord.size(); ++i)
	os << " " << coord[i];

      alist.clear();
      alist.push_back(XMLWriterAPI::Attribute("site", site));
      alist.push_back(XMLWriterAPI::Attribute("coord", os.str()));

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
#if 1
      // All nodes participate
      QDPInternal::route((void *)&recv_buf, node, 0, sizeof(Site_t));
#else
      if (Layout::primaryNode())
	QDPInternal::recvFromWait((void *)&recv_buf, node, sizeof(Site_t));

      if (Layout::nodeNumber() == node)
	QDPInternal::sendToWait((void *)&recv_buf, 0, sizeof(Site_t));
#endif
    }

    if (Layout::primaryNode())
      bin.writeArray((char *)&recv_buf,
		     sizeof(typename WordType<Site_t>::Type_t), 
		     sizeof(Site_t) / sizeof(typename WordType<Site_t>::Type_t));
  }
}

//! Write a single site from coord
/*! An inner grid is assumed */
template<class T>
void write(BinaryWriter& bin, const OLattice<T>& d, const multi1d<int>& coord)
{
  typedef typename UnaryReturn<T, FnGetSite>::Type_t  Site_t;
  Site_t  recv_buf;

  // Find the location of each site and send to primary node
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
#if 1
      // All nodes participate
      QDPInternal::route((void *)&recv_buf, node, 0, sizeof(Site_t));
#else
    if (Layout::primaryNode())
      QDPInternal::recvFromWait((void *)&recv_buf, node, sizeof(Site_t));

    if (Layout::nodeNumber() == node)
      QDPInternal::sendToWait((void *)&recv_buf, 0, sizeof(Site_t));
#endif
  }

  if (Layout::primaryNode())
    bin.writeArray((char *)&recv_buf,
		   sizeof(typename WordType<Site_t>::Type_t), 
		   sizeof(Site_t) / sizeof(typename WordType<Site_t>::Type_t));
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
#if 1
      // All nodes participate
      QDPInternal::route((void *)&recv_buf, 0, node, sizeof(Site_t));
#else
      if (Layout::primaryNode())
	QDPInternal::sendToWait((void *)&recv_buf, node, sizeof(Site_t));

      if (Layout::nodeNumber() == node)
	QDPInternal::recvFromWait((void *)&recv_buf, 0, sizeof(Site_t));
#endif
    }

    if (Layout::nodeNumber() == node)
      copy_site(d.elem(outersite), innersite, recv_buf);// insert into conventional scalar form
  }
}

//! Read a single lattice site worth of data
/*! An inner grid is assumed */
template<class T>
void read(BinaryReader& bin, OLattice<T>& d, const multi1d<int>& coord)
{
  typedef typename UnaryReturn<T, FnGetSite>::Type_t  Site_t;
  Site_t  recv_buf;

  // Find the location of each site and send to primary node
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
#if 1
      // All nodes participate
      QDPInternal::route((void *)&recv_buf, 0, node, sizeof(Site_t));
#else
    if (Layout::primaryNode())
      QDPInternal::sendToWait((void *)&recv_buf, node, sizeof(Site_t));

    if (Layout::nodeNumber() == node)
      QDPInternal::recvFromWait((void *)&recv_buf, 0, sizeof(Site_t));
#endif
  }
  
  if (Layout::nodeNumber() == node)
    copy_site(d.elem(outersite), innersite, recv_buf);// insert into conventional scalar form
}


// **************************************************************
// Special support for slices of a lattice
// **************************************************************

// --JIe Chen: need to implement
namespace LatticeTimeSliceIO 
{
  //! Lattice time slice reader

  template<class T>
  void readSlice(BinaryReader& bin, OLattice<T>& data, 
		 int start_lexico, int stop_lexico)
  {
    // check whether dimention t is multiple of INNER_LEN
    int tDir = Nd-1;

    if (Layout::lattSize()[tDir] % INNER_LEN != 0)
      QDP_error_exit ("Size of time dimension %d is not multiple of vector len %d\n", Layout::lattSize()[tDir], INNER_LEN);

    const int xinc = Layout::subgridLattSize()[0];

    if ((stop_lexico % xinc) != 0) {
      QDPIO::cerr << __func__ << ": erorr: stop_lexico= " << stop_lexico << "  xinc= " << xinc << std::endl;
      QDP_abort(1);
    }

    // memory block size of vectorized type
    size_t nmemb = sizeof(T) / sizeof(typename WordType<T>::Type_t);
    // Now memory block size is the same as non-vectorized type
    nmemb = (nmemb >> INNER_LOG); 

    // individual element size
    size_t size = sizeof(typename WordType<T>::Type_t);

    // one element memory size in bytes
    size_t sizemem = size * nmemb;
    // one time slice memory size in bytes
    size_t tot_size = sizemem * xinc;
    // allocate memory buffer
    char *recv_buf = new(nothrow) char[tot_size];
    if( recv_buf == 0x0 ) { 
      QDP_error_exit("Unable to allocate recv_buf\n");
    }
    
    // Find the location of each site and send to primary node
    for (int site=start_lexico; site < stop_lexico; site += xinc){
      // first site in each segment uniquely identifies the node
      int node = Layout::nodeNumber(crtesn(site, Layout::lattSize()));

      // Only on primary node read the data
      bin.readArrayPrimaryNode(recv_buf, size, nmemb * xinc);

      // Send result to destination node. Avoid sending prim-node sending to itself
      if (node != 0) {
#if 1
	// All nodes participate
	QDPInternal::route((void *)recv_buf, 0, node, tot_size);
#else
	if (Layout::primaryNode())
	  QDPInternal::sendToWait((void *)recv_buf, node, tot_size);
	if (Layout::nodeNumber() == node)
	  QDPInternal::recvFromWait((void *)recv_buf, 0, tot_size);
#endif
      }

      if (Layout::nodeNumber() == node) {
	for(int i=0; i < xinc; i++) {
	  int linear = Layout::linearSiteIndex(crtesn(site+i, Layout::lattSize()));
	  int outersite = linear >> INNER_LOG;
	  int innersite =  linear & ((1 << INNER_LOG)-1);
	  
	  // read data back in piece-by-piece and insert into data
	  typedef typename UnaryReturn<T, FnGetSite>::Type_t  Site_t;
	  Site_t  this_site;

	  memcpy (&this_site, recv_buf + i * sizemem, sizemem);

	  copy_site (data.elem(outersite), innersite, this_site);
	}
      }
    }
    delete[] recv_buf;
  }

  template<class T>
  void writeSlice(BinaryWriter& bin, OLattice<T>& data, 
		  int start_lexico, int stop_lexico)
  {
    // check whether dimention t is multiple of INNER_LEN
    int tDir = Nd-1;

    if (Layout::lattSize()[tDir] % INNER_LEN != 0)
      QDP_error_exit("Size of time dimension %d is not multiple of vector len %d\n", Layout::lattSize()[tDir], INNER_LEN);

    const int xinc = Layout::subgridLattSize()[0];

    if ((stop_lexico % xinc) != 0) {
      QDPIO::cerr << __func__ << ": erorr: stop_lexico= " << stop_lexico << "  xinc= " << xinc << std::endl;
      QDP_abort(1);
    }

    // memory block size of vectorized type
    size_t nmemb = sizeof(T) / sizeof(typename WordType<T>::Type_t);
    // Now memory block size is the same as non-vectorized type
    nmemb = (nmemb >> INNER_LOG); 

    // individual element size
    size_t size = sizeof(typename WordType<T>::Type_t);

    // one element memory size in bytes
    size_t sizemem = size * nmemb;
    // one time slice memory size in bytes
    size_t tot_size = sizemem * xinc;
    // allocate memory buffer
    char *recv_buf = new(nothrow) char[tot_size];
    if( recv_buf == 0x0 ) { 
      QDP_error_exit("Unable to allocate recv_buf\n");
    }

    // Find the location of each site and send to primary node
    int old_node = 0;

    for (int site=start_lexico; site < stop_lexico; site += xinc) {
      // first site in each segment uniquely identifies the node
      int node = Layout::nodeNumber(crtesn(site, Layout::lattSize()));

      // Send nodes must wait for a ready signal from the master node
      // to prevent message pileups on the master node
      if (node != old_node){
	// On non-grid machines, use a clear-to-send like protocol
	QDPInternal::clearToSend(recv_buf,sizeof(int),node);
	old_node = node;
      }
    
      // Copy to buffer: be really careful since max(linear) could vary among nodes
      if (Layout::nodeNumber() == node){
	for(int i=0; i < xinc; i++) {
	  int linear = Layout::linearSiteIndex(crtesn(site+i,Layout::lattSize()));
	  int outersite = linear >> INNER_LOG;
	  int innersite =  linear & ((1 << INNER_LOG)-1);

	  // write data piece-by-piece from each innersite
	  typedef typename UnaryReturn<T, FnGetSite>::Type_t  Site_t;
	  Site_t  this_site = getSite(data.elem(outersite),innersite);

	  memcpy(recv_buf+ i * sizemem, &this_site, sizemem);
	}
      }

      // Send result to primary node. Avoid sending prim-node sending to itself
      if (node != 0) {
#if 1
	// All nodes participate
	QDPInternal::route((void *)recv_buf, node, 0, tot_size);
#else
	if (Layout::primaryNode())
	  QDPInternal::recvFromWait((void *)recv_buf, node, tot_size);
	if (Layout::nodeNumber() == node)
	  QDPInternal::sendToWait((void *)recv_buf, 0, tot_size);
#endif
      }

      bin.writeArrayPrimaryNode(recv_buf, size, nmemb*xinc);
    }
    delete[] recv_buf;
  }
  
} // namespace LatticeTimeSliceIO

} // namespace QDP

#endif
