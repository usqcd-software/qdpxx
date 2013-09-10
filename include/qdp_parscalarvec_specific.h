// -*- C++ -*-

/*! @file
 * @brief Outer/inner lattice routines specific to a parscalarvec platform 
 */

#ifndef QDP_PARSCALARVEC_SPECIFIC_H
#define QDP_PARSCALARVEC_SPECIFIC_H

#include "qdp_config.h"
#include "qmp.h"
#include "qdp_mapresource.h"
#include "qdp_pshift.h"
#include "qdp_mastermap.h"

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

  if (s.hasOrderedRep()) {
    const int istart = s.start() >> INNER_LOG;
    const int iend   = s.end()   >> INNER_LOG;
    // #pragma omp parallel shared(istart, iend, num_threads) default(shared) 
#pragma omp parallel shared(num_threads) default(shared) 
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
    // this part of code is not correct because non-contiguous memory layout
    // need gather and scatter code for Olattice
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
	op(dest.elem(outersite), forEach(rhs, EvalLeaf3(0, low >> INNER_LOG, high >> INNER_LOG), OpCombine()));
      }
    }
  }
  
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
//  cerr << "In evaluateSubset(olattice,olattice)" << endl;
  int num_threads = 1;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest, op, rhs);
  prof.time -= getClockTime();
#endif

  /**
   * walk through expression tree to find out whether  there is 
   * fnmap operator or shift involved
   *
   * send buffer will be constructed and data should be sent out
   * single thread here
   */
  ShiftPhase1 phase1;
  int maps_involved = forEach(rhs, phase1 , BitOrCombine());
  
  if (maps_involved > 0) {

    // there are shifts involved and data have been sent out
    const multi1d<int>& innerSites = MasterMap::Instance().getInnerSites(maps_involved);
    const multi1d<int>& faceSites = MasterMap::Instance().getFaceSites(maps_involved);

    if (s.hasOrderedRep()) {
#pragma omp parallel for
      // inner sites are aligned. For shift along x direction,
      // all sites that are within INNER_LEN of a surface sites
      // are treated as non-interior sites : fat surface
      for (int i = 0; i < innerSites.size(); i += INNER_LEN) {
	if (s.isElement(innerSites[i])) 
	  op(dest.elem(innerSites[i] >> INNER_LOG), forEach(rhs, EvalLeaf1(innerSites(i) >> INNER_LOG), OpCombine()));
      }

      // now handle surface receiving data
      ShiftPhase2 phase2;
      forEach(rhs, phase2 , NullCombine());

      // data should be received by now
#pragma omp parallel for
      // now processing surface sites
      for (int i = 0; i < faceSites.size(); i += INNER_LEN) {
	if (s.isElement(faceSites[i])) {
	  op(dest.elem(faceSites[i] >> INNER_LOG), forEach(rhs,  EvalLeaf1(faceSites(i) >> INNER_LOG), OpCombine()));
	}
      }
    }
    else {
      QDP_error_exit ("Not implemented for non-OrderedRep() map operations\n");
    }
  }
  else {
    if (s.hasOrderedRep()) {
      const int istart = s.start() >> INNER_LOG;
      const int iend   = s.end()   >> INNER_LOG;

      // #pragma omp parallel shared(istart, iend, num_threads) default(shared)
#pragma omp parallel shared(num_threads) default(shared)
      {
	num_threads = omp_get_num_threads (); 
	int myId = omp_get_thread_num();
	int low = istart + (iend - istart + 1) * myId/num_threads;
	int high = istart + (iend - istart + 1) * (myId + 1)/num_threads;
	
	for(int i = low; i < high; ++i) {
	  // QDP_info("Thread %d: eval(olattice,olattice): site %d low %d high %d\n", myId, i, low, high);
	  op(dest.elem(i), forEach(rhs, EvalLeaf1(i), OpCombine()));
	}
      }
    }
    else {
      QDP_error_exit ("Not implemented for non-OrderedRep()\n");
    }
  }

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

  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    copymask(dest.elem(i), mask.elem(i), s1.elem(i));
  }
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
  if (s.hasOrderedRep()) {
    const int istart = s.start() >> INNER_LOG;
    const int iend   = s.end()   >> INNER_LOG;

#pragma omp parallel for
    for(int i=istart; i <= iend; ++i)
      zero_rep(dest.elem(i));
  }
  else {
    const int *tab = s.siteTable().slice();
#pragma omp parallel for
    for(int j=0; j < s.numSiteTable(); ++j) 
      {
	int i = tab[j];
	int outersite = i >> INNER_LOG;
	zero_rep(dest.elem(outersite));
      }
  }
}



//! dest  = 0 
template<class T>
void zero_rep(OSubLattice<T> dd) 
{
  OLattice<T>& d = dd.field();
  const Subset& s = dd.subset();
  
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
#if 0
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
#endif

#if 1
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

  /**
   * walk through expression tree to find out whether  there is 
   * fnmap operator or shift involved
   *
   * send buffer will be constructed and data should be sent out
   * single thread here
   */
  ShiftPhase1 phase1;
  int maps_involved = forEach(s1, phase1 , BitOrCombine());

  if (maps_involved > 0) {
    // there are shifts involved and data have been sent out
    const multi1d<int>& innerSites = MasterMap::Instance().getInnerSites(maps_involved);
    const multi1d<int>& faceSites = MasterMap::Instance().getFaceSites(maps_involved);

    if (s.hasOrderedRep()) {
      typename UnaryReturn<OLattice<T>, FnSum>::Type_t  *pd = new typename UnaryReturn<OLattice<T>, FnSum>::Type_t[omp_get_max_threads()];

      int segsize = 1;

      if ((innerSites.size()/qdpNumThreads()) & (INNER_LEN - 1) == 0)
	segsize = innerSites.size()/qdpNumThreads();
      else
	segsize = (innerSites.size()/qdpNumThreads() + INNER_LEN - 1)/INNER_LEN * INNER_LEN;

      // the following omp parallel block may not be efficient
      // since some of threads may execute nothing due to
      // some of innersites are not in the subset
#pragma omp parallel shared(pd,segsize,num_threads) private(tmp) default(shared)
      {
	num_threads = omp_get_num_threads();
	int myId = omp_get_thread_num();
	int low = segsize * myId;
	int high = segsize * (myId + 1);
	if (low >= innerSites.size())
	  low = high = innerSites.size();
	if (high >= innerSites.size())
	  high = innerSites.size();

	// zero the partial result
	zero_rep(pd[myId].elem());

	for(int i = low; i < high; i += INNER_LEN) {
	  if (s.isElement(innerSites[i])) {
	    tmp.elem() = forEach(s1, EvalLeaf1(innerSites[i] >> INNER_LOG), OpCombine()); // Evaluate to ILattice part
	    pd[myId].elem() += sum(tmp.elem()); // sum as well the ILattice part
	  }
	}
      }

      // now handle surface receiving data
      ShiftPhase2 phase2;
      forEach(s1, phase2 , NullCombine());

      // data should be received by now
      // handle face sites
      // the following omp parallel block may not be efficient
      // since some of threads may execute nothing due to
      // some of innersites are not in the subset
      if ((faceSites.size()/qdpNumThreads()) & (INNER_LEN - 1) == 0)
	segsize = faceSites.size()/qdpNumThreads();
      else
	segsize = (faceSites.size()/qdpNumThreads() + INNER_LEN - 1)/INNER_LEN * INNER_LEN;

#pragma omp parallel shared(pd,segsize,num_threads) private(tmp) default(shared)
      {
	num_threads = omp_get_num_threads();
	int myId = omp_get_thread_num();
	int low = segsize * myId;
	int high = segsize * (myId + 1);
	if (low >= faceSites.size())
	  low = high = faceSites.size();
	if (high >= faceSites.size())
	  high = faceSites.size();

	int isite = 0;
	for(int i = low; i < high; i += INNER_LEN) {
	  if (s.isElement(faceSites[i])) {
	    tmp.elem() = forEach(s1, EvalLeaf1(faceSites[i] >> INNER_LOG), OpCombine()); // Evaluate to ILattice part
	    pd[myId].elem() += sum(tmp.elem()); // sum as well the ILattice part
	  }
	}
      }
      
      // Now combine values from each thread
      for (int j = 0; j < num_threads; j++) 
	d.elem() += sum(pd[j].elem());

      delete []pd;
    }
    else {

            QDP_error_exit ("SUM has not been implemented for non-OrderedRep() operations\n");
    }



  }
  else {
    // no shift is involved
    if (s.hasOrderedRep()) {
      const int istart = s.start() >> INNER_LOG;
      const int iend   = s.end()   >> INNER_LOG;

      typename UnaryReturn<OLattice<T>, FnSum>::Type_t  *pd = new  typename UnaryReturn<OLattice<T>, FnSum>::Type_t[ omp_get_max_threads() ]; 

      // #pragma omp parallel shared(pd,istart,iend,num_threads) private(tmp) default(shared)
 #pragma omp parallel shared(pd,num_threads) private(tmp) default(shared)
      {
	num_threads = omp_get_num_threads();
	int myId = omp_get_thread_num();
	int low = istart + (iend - istart + 1) * myId/num_threads;
	int high = istart + (iend - istart + 1) * (myId + 1)/num_threads;
	// zero the partial result
	zero_rep(pd[myId].elem());

	for(int i = low; i < high; ++i) {
	  tmp.elem() = forEach(s1, EvalLeaf1(i), OpCombine()); // Evaluate to ILattice part
	  pd[myId].elem() += sum(tmp.elem()); // sum as well the ILattice part
	}
      }
      // Now combine all together
      for (int j = 0; j < num_threads; j++) {
	d.elem() += sum(pd[j].elem());
      }

      delete []pd;
    }
    else {
      // OK No Ordered Rep. 
      // This is a dumb way to do it.  No vectorization. Repeated applications.
      // I introduce the concept of sum over mask.
      // FIXME: Come back and fix the loop.

      // One partial some per thread.
      typename UnaryReturn<OLattice<T>, FnSum>::Type_t  *pd = new  typename UnaryReturn<OLattice<T>, FnSum>::Type_t[ omp_get_max_threads()]; 

      // Zero result
      for(int j=0; j < omp_get_max_threads(); j++) {
	zero_rep(pd[j].elem());
      }
      
      // Get the site table
      const int *tab = s.siteTable().slice();


      // Parallel loop through the sites
#pragma omp parallel for shared(tab,pd) private(tmp)
      for(int site=0; site < s.numSiteTable(); site++) { 
	int myId = omp_get_thread_num();
	int fullsite = tab[site];
	int osite = fullsite >> INNER_LOG;
	int isite = fullsite & (INNER_LEN - 1);
	
	bool mask[INNER_LEN]; 
	for(int maskbit=0; maskbit < INNER_LEN; maskbit++) mask[maskbit] = false;
	mask[isite] = true; // Mask the bit we are currently working on
	
	// Evaluate the full inner vector
	// Now it is entirely possible that another thread is re-doing this vector,
	// or that this thread will stay within this vector for the next operation.
	//
	// Hopefully the mask will ensure no dublicates.
	// 
	// NB: This is really dumb way of doing it. We should have tables for outer-blocks
	// and masks for each set and divide those between the threads. 
	tmp.elem() = forEach(s1, EvalLeaf1(osite), OpCombine());
	
	// Sum the ILattice part under a mask
	pd[myId].elem() += sum(tmp.elem(), mask);
      }// End for
      
      // Now combine it all together
      for (int j = 0; j < omp_get_max_threads(); j++) {
	d.elem() += sum(pd[j].elem());
      }
      
      delete []pd;

    } // ! has Oredered rep    
  } // ! has Shift part


  // Do a global sum on the result
  QDPInternal::globalSum(d);
  
#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}
#endif

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


// ! Oscalar = innerProductReal( multi1d<OScalar<T> >, multi1d<OScalar<T> > )
template<class T>
inline typename BinaryReturn<OScalar<T>, OScalar<T>, FnInnerProductReal>::Type_t
innerProductReal(const multi1d< OScalar<T> >& s1, 
		 const multi1d< OScalar<T> >& s2 )
{
  typename BinaryReturn<OScalar<T>, OScalar<T>, FnInnerProductReal>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnInnerProductReal(), s1[0],s2[0]);
  prof.time -= getClockTime();
#endif

  // Possibly loop entered
  zero_rep(d.elem());

  // Loop over the multi1d size and accumulate
  for(int n=0; n < s1.size(); ++n)
  {
    OScalar<T>& ss1 = s1[n];
    OScalar<T>& ss2 = s2[n];

    d.elem() += localInnerProductReal(ss1.elem(), ss2.elem());
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
inline typename BinaryReturn<OScalar<T>, 
			     OScalar<T>,
			     FnInnerProductReal >::Type_t
innerProductReal(const multi1d< OScalar<T> >& s1, 
		 const multi1d< OScalar<T> >& s2,
		 const Subset& s)
{
  // NB: This is on scalars. So subset doesn't matter.
  // In fact I question the validity of providing this override.
  // In any case, since subset doesn't matter just feed it through
  // to the global one.
  return innerProductReal(s1,s2);
}


// ! OScalar = innerProductReal( mulit1d<OLattice>, multi1d<OLattice> ) 
template<class T>
inline typename BinaryReturn< OLattice<T>, OLattice<T>, FnInnerProductReal>::Type_t
innerProductReal(const multi1d< OLattice<T> >& s1, 
		 const multi1d< OLattice<T> >& s2, 
		 const Subset& s)
{
  typename BinaryReturn< OLattice<T>, OLattice<T>, FnInnerProductReal>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnInnerProductReal(), s1[0], s2[0]);
  prof.time -= getClockTime();
#endif

  // Possibly loop entered
  zero_rep(d.elem());

  if (s.hasOrderedRep()) {
    const int istart = s.start() >> INNER_LOG;
    const int iend   = s.end()   >> INNER_LOG;

    // Loop over the multi1d array index and accumulate locally
    for(int n=0; n < s1.size(); ++n) {
      const OLattice<T>& ss1 = s1[n];
      const OLattice<T>& ss2 = s2[n];

      for (int i = istart; i <= iend; ++i)  {
	d.elem() += localInnerProductReal(ss1.elem(i), ss2.elem(i));
      }
    }
  }
  else {
    // Unordered, so loop over sites
    const int *tab = s.siteTable().slice();
    for(int n=0; n < s1.size(); ++n)
      {
	const OLattice<T>& ss1 = s1[n];
	const OLattice<T>& ss2 = s2[n];

	for(int j=0; j < s.numSiteTable(); ++j) {
	  int i = tab[j];
	  int outersite = i >> INNER_LOG;
	  int innersite = i & ((1 << INNER_LOG)-1);
	  d.elem() += localInnerProductReal(ss1.elem(outersite),
					    ss2.elem(outersite));
	}
      }
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

// ! OScalar = innerProductReal( mulit1d<OLattice>, multi1d<OLattice> ) 
template<class T>
inline typename BinaryReturn< OLattice<T>, OLattice<T>, FnInnerProductReal>::Type_t
innerProductReal(const multi1d< OLattice<T> >& s1, 
		 const multi1d< OLattice<T> >& s2)

{
  return innerProductReal(s1,s2,all);
}


// ! Oscalar = innerProduct( multi1d<OScalar<T> >, multi1d<OScalar<T> > )
template<class T>
inline typename BinaryReturn<OScalar<T>, OScalar<T>, FnInnerProduct>::Type_t
innerProduct(const multi1d< OScalar<T> >& s1, 
	     const multi1d< OScalar<T> >& s2 )
{
  typename BinaryReturn<OScalar<T>, OScalar<T>, FnInnerProduct>::Type_t  d;
  
#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnInnerProduct(), s1[0],s2[0]);
  prof.time -= getClockTime();
#endif
  
  // Possibly loop entered
  zero_rep(d.elem());
  
  // Loop over the multi1d size and accumulate
  for(int n=0; n < s1.size(); ++n) {
    OScalar<T>& ss1 = s1[n];
    OScalar<T>& ss2 = s2[n];
      
    d.elem() += localInnerProduct(ss1.elem(), ss2.elem());
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
inline typename BinaryReturn<OScalar<T>, 
			     OScalar<T>,
			     FnInnerProduct >::Type_t
innerProduct(const multi1d< OScalar<T> >& s1, 
	     const multi1d< OScalar<T> >& s2,
	     const Subset& s)
{
  // NB: This is on scalars. So subset doesn't matter.
  // In fact I question the validity of providing this override.
  // In any case, since subset doesn't matter just feed it through
  // to the global one.
  return innerProduct(s1,s2);
}



// ! OScalar = innerProductReal( mulit1d<OLattice>, multi1d<OLattice> ) 
template<class T>
inline typename BinaryReturn< OLattice<T>, OLattice<T>, FnInnerProduct>::Type_t
innerProduct(const multi1d< OLattice<T> >& s1, 
	     const multi1d< OLattice<T> >& s2, 
	     const Subset& s)
{
  typename BinaryReturn< OLattice<T>, OLattice<T>, FnInnerProduct>::Type_t  d;
  
#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnInnerProduct(), s1[0], s2[0]);
  prof.time -= getClockTime();
#endif
  
  zero_rep(d.elem());
  
  if (s.hasOrderedRep()) {
    const int istart = s.start() >> INNER_LOG;
    const int iend   = s.end()   >> INNER_LOG;
    
    // Loop over the multi1d array index and accumulate locally
    for(int n=0; n < s1.size(); ++n) {
      const OLattice<T>& ss1 = s1[n];
      const OLattice<T>& ss2 = s2[n];
      
      for (int i = istart; i <= iend; ++i)  {
	d.elem() += localInnerProduct(ss1.elem(i), ss2.elem(i));
      }
    }
  }
  else {
    // Unordered, so loop over sites
    const int *tab = s.siteTable().slice();
    for(int n=0; n < s1.size(); ++n) {
      
      const OLattice<T>& ss1 = s1[n];
      const OLattice<T>& ss2 = s2[n];
      
      for(int j=0; j < s.numSiteTable(); ++j) {
	int i = tab[j];
	int outersite = i >> INNER_LOG;
	int innersite = i & ((1 << INNER_LOG)-1);
	d.elem() += localInnerProduct(ss1.elem(outersite),
				      ss2.elem(outersite));
      }
    }
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

// ! OScalar = innerProductReal( mulit1d<OLattice>, multi1d<OLattice> ) 
template<class T>
inline typename BinaryReturn< OLattice<T>, OLattice<T>, FnInnerProduct>::Type_t
innerProduct(const multi1d< OLattice<T> >& s1, 
	     const multi1d< OLattice<T> >& s2)
  
{
  return innerProduct(s1,s2,all);
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

/**
 * A leaf functor for shift
 * It contains an original site (outer) index and a map reference
 */
struct EvalLeaf1Map
{
  int i1_m;
  const Map& map_;

  inline EvalLeaf1Map (int i1, const Map& m)
    :i1_m(i1), map_(m) 
  {
    // empty
  }

  inline int val1() const { return i1_m; }
  inline const Map& map() const { return map_; }
};

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
volatile
void gather_along_x (const QDPType<T, C>& a, int old_inner_site, 
		     const int* offsets,
		     T& ret)
{
  int i = old_inner_site;
  
#if INNER_LOG == 1
  // VECLEN=2
  int o1 = offsets[i+1] >> INNER_LOG;
  int i1 = offsets[i+1] & (INNER_LEN - 1);
  gather_sites(ret,
	       a.elem(o0),i0,
	       a.elem(o1),i1);

#elif INNER_LOG == 2
  // VECLEN = 4
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
  // VECLEN =8 
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
  // VECLEN=16
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
T intra_shift (const QDPType<T, C>& a, const EvalLeaf1Map& f)
{
  const Map& map = f.map();
  const int *goffsets = map.goffset().slice();

  if (map.mapDir() != 0) { // not along x direction
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
T intra_shift (const OLattice<T>& a, const EvalLeaf1Map& f)
{
  return intra_shift<T, OLattice<T> > ((const QDPType<T, OLattice<T> >)a, f);
}


/**
 * Handle shift for inner sites or handle for the case of
 * no communication required
 */
template<class T, class C>
struct LeafFunctor<QDPType<T,C>, EvalLeaf1Map>
{
  typedef Reference<T> Type_t;
//  typedef T Type_t;
  inline static Type_t apply(const QDPType<T,C> &a, const EvalLeaf1Map &f)
  { 
    return (Type_t)(intra_shift(a, f));
  }
};

template<int N> 
struct LeafFunctor<GammaType<N>, EvalLeaf1Map>
{
  typedef GammaType<N> Type_t;

  inline static Type_t apply(const GammaType<N> &a, const EvalLeaf1Map &f)
  {
    // No need to intra shift it cos it is not a lattice thingie?
    return a;
  }
};

template<class T>
struct LeafFunctor<OScalar<T>, EvalLeaf1Map>
{
//  typedef T Type_t;
  typedef Reference<T> Type_t;
  inline static Type_t apply(const OScalar<T> &a, const EvalLeaf1Map &f)
  {
    return Type_t(a.elem());
  }
};

template<class T>
struct LeafFunctor<OLattice<T>, EvalLeaf1Map>
{
  //  typedef T Type_t;
  typedef Reference<T> Type_t;

  inline static Type_t apply(const OLattice<T> &a, const EvalLeaf1Map &f)
  {
    return (Type_t)(intra_shift(a, f));
  }
};

/**
 * Handle FnMap for ShiftPhase1. Other expressions returns 0
 * as defined inside LeafFunctor for ShiftPhase1.
 * Type is also defined inside LeafFunctor as integer
 * BitOrCombine is therefore used to determine whether we 
 * need shift or not
 */
template<typename A>
struct ForEach<UnaryNode<FnMap, A>, ShiftPhase1 , BitOrCombine>
{
  typedef typename ForEach<A, EvalLeaf1, OpCombine>::Type_t InnerTypeA_t;
  typedef typename Combine1<InnerTypeA_t, FnMap, OpCombine>::Type_t InnerType_t;
  typedef int Type_t;
  typedef QDPExpr<A,OLattice<InnerType_t> > Expr;

  inline static
  Type_t apply(const UnaryNode<FnMap, A> &expr, const ShiftPhase1 &f, 
	       const BitOrCombine &c)
  {
    const Map& map = expr.operation().map;
    FnMap& fnmap = const_cast<FnMap&>(expr.operation());

    const int nodeSites = Layout::sitesOnNode();
    int returnVal=0;

    if (map.offnodeP) {
#if QDP_DEBUG >= 3
      QDP_info("Map: off-node communications required");
#endif

      int dstnum = 0;
      int srcnum = 0;
      if (map.mapDir() != 0) {
	dstnum = (map.destnodes_num[0] >> INNER_LOG) * sizeof(InnerType_t);
	srcnum = (map.srcenodes_num[0] >> INNER_LOG) * sizeof(InnerType_t);
      }
      else {
	dstnum = map.destnodes_num[0]*sizeof(InnerType_t);
	srcnum = map.srcenodes_num[0]*sizeof(InnerType_t);
      }

      const FnMapRsrc& rRSrc = fnmap.getResource(srcnum,dstnum);
      const InnerType_t *send_buf_c = rRSrc.getSendBufPtr<InnerType_t>();

      InnerType_t* send_buf = const_cast<InnerType_t*>(send_buf_c);
      if (send_buf == 0x0) {
	QDP_error_exit("QMP_get_memory_pointer returned NULL pointer from non NULL QMP_mem_t (send_buf)\n");
      }
      
      const int my_node = Layout::nodeNumber();
      Expr subexpr(expr.child());

      // Make sure the inner expression's map function
      // send and receive before recursing down
      int maps_involved = forEach(subexpr, f , BitOrCombine());
      if (maps_involved > 0) {
	ShiftPhase2 phase2;
	forEach(subexpr, phase2 , NullCombine());
      }

      // Gather the face of data to send
      // For now, use the all subset
      if (map.mapDir() != 0) {
	// not along x direction of this shift, neighbors are packed together
	// need to do openmp here
	// calclulation is done before data are sent out ?
#pragma omp parallel for
	for (int si = 0; si < map.soffsets.size(); si += INNER_LEN)
	  send_buf[si >> INNER_LOG] = forEach( subexpr , 
				   EvalLeaf1(map.soffsets[si] >> INNER_LOG) ,
				   OpCombine() );
      }
      else {
	// along x direction, more sites are needed
	// even though map.soffsets[si] may not be aligned on INNER_LEN.
	// but soffsets[i] >> INNER_LOG is good for whole vector len of data
#pragma omp parallel for
	for (int si = 0; si < map.soffsets.size(); si ++) {
	  send_buf[si] = forEach( subexpr , 
				  EvalLeaf1(map.soffsets[si] >> INNER_LOG) ,
				  OpCombine() );
	}
      }
      
      // sending data out
      rRSrc.send_receive();

      // if we are shifting along x direction, we need to puplate the fat receiving
      // sites as well except the real face sites
      if (map.mapDir() == 0) {
	const InnerType_t *shadow_buf_c = rRSrc.getShadowBufPtr<InnerType_t>();
	InnerType_t* shadow_buf = const_cast<InnerType_t*>(shadow_buf_c);

#pragma omp parallel for	
	for (int gi = 0; gi < map.goffsets.size(); gi++) {
	  if (map.goffsets[gi] < 0) { // receiving sites
	    int bufpos = -map.goffsets[gi]-1;
	    if (map.mapSign() < 0) {
	      // shift right
	      shadow_buf[bufpos] = forEach( subexpr , 
					    EvalLeaf1( map.goffsets[gi + 1] >> INNER_LOG) ,
					    OpCombine() );
	    }
	    else {
	      // shift to left
	      shadow_buf[bufpos] = forEach( subexpr , 
					    EvalLeaf1(map.goffsets[gi - 1] >> INNER_LOG) ,
					    OpCombine() );
	    }

	  }
	}
      }
      returnVal = maps_involved | map.getId();
    } 
    else 
      returnVal = ForEach<A, ShiftPhase1, BitOrCombine>::apply(expr.child(), f, c);

    return returnVal;
  }
};

template<class A, class CTag>
struct ForEach<UnaryNode<FnMap, A>, ShiftPhase2 , CTag>
{
  // typedef typename ForEach<A, EvalLeaf1, OpCombine>::Type_t TypeA_t;
  // typedef typename Combine1<TypeA_t, FnMap, OpCombine>::Type_t Type_t;
  // typedef QDPExpr<A,OLattice<Type_t> > Expr;
  typedef int Type_t;
  inline static
  Type_t apply(const UnaryNode<FnMap, A> &expr, const ShiftPhase2 &f, const CTag &c)
  {
    const Map& map = expr.operation().map;
    FnMap& fnmap = const_cast<FnMap&>(expr.operation());
    if (map.offnodeP) {
      const FnMapRsrc& rRSrc = fnmap.getCached();
      rRSrc.qmp_wait();
    }
    return ForEach<A, ShiftPhase2, CTag>::apply(expr.child(), f, c);
  }
};


/**
 * This part of code works for ordered layout only. Especially for x direction
 * shift
 */
template<class A, class CTag>
struct ForEach<UnaryNode<FnMap, A>, EvalLeaf1, CTag>
{
  // note: f.val1() is the an outersite
  typedef typename ForEach<A, EvalLeaf1, CTag>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnMap, CTag>::Type_t Type_t;
  inline static
  Type_t apply(const UnaryNode<FnMap, A> &expr, const EvalLeaf1 &f, const CTag &c)
  {
    const Map& map = expr.operation().map;
    FnMap& fnmap = const_cast<FnMap&>(expr.operation());

    // the original site not the outer site
    int osite = f.val1() << INNER_LOG;
    if (map.offnodeP) {
      if (map.mapDir() != 0 && map.goffsets[osite] < 0) {
	// face site, ready to get data from receiving buffer
	const FnMapRsrc& rRSrc = fnmap.getCached();
	const Type_t *recv_buf_c = rRSrc.getRecvBufPtr<Type_t>();
	Type_t* recv_buf = const_cast<Type_t*>(recv_buf_c);

#if QDP_DEBUG >= 3
	if ( recv_buf == 0x0 ) { 
	  QDP_error_exit("QMP_get_memory_pointer returned NULL pointer from non NULL QMP_mem_t (recv_buf). Do you use shifts of shifts?"); 
	}
#endif
	int gpos = -map.goffsets[osite]-1;
	return recv_buf[gpos >> INNER_LOG];
      }
      else if (map.mapDir() == 0 && (map.goffsets[osite] < 0 || map.goffsets[osite + INNER_LEN - 1] < 0)) {
	// face site, ready to get data from receiving buffer
	const FnMapRsrc& rRSrc = fnmap.getCached();
	const Type_t *recv_buf_c = rRSrc.getRecvBufPtr<Type_t>();
	Type_t* recv_buf = const_cast<Type_t*>(recv_buf_c);

	const Type_t *shadow_buf_c = rRSrc.getShadowBufPtr<Type_t>();
	Type_t* shadow_buf = const_cast<Type_t*>(shadow_buf_c);
#if QDP_DEBUG >= 3
	if ( recv_buf == 0x0 ) { 
	  QDP_error_exit("QMP_get_memory_pointer returned NULL pointer from non NULL QMP_mem_t (recv_buf). Do you use shifts of shifts?"); 
	}
#endif
	int gpos = 0; 
	// shift alon x direction. Data contains non-surface data
	if (map.mapSign () < 0) {
	  gpos = -map.goffsets[osite]-1;
	  // shift to right, only the first element of the vec-length
	  // receives real data over network. The others are from shadow buffer
	  // pick out each inner site that are in the veclength

	  // move the last element inside the buffer to the beginning of the buffer
	  copy_site (recv_buf[gpos], 0,
		     getSite (recv_buf[gpos], INNER_LEN - 1));
	  for (int i = 1; i < INNER_LEN; i++) {
	    copy_site (recv_buf[gpos], i,
		       getSite (shadow_buf[gpos], i - 1));
	  }
	}
	else {
	  // shift to left, only the last element of the vec-length
	  // receives data over network. The others are from shadow buffer.
	  // pick out each inner site that are in the veclength 
	  gpos = -map.goffsets[osite + INNER_LEN - 1]-1;

	  // move data from the beginning of the buffer to the end
	  copy_site (recv_buf[gpos], INNER_LEN - 1,
		     getSite (recv_buf[gpos], 0));
		     
	  // shift the shadow buffer now
	  for (int i = 0; i < INNER_LEN - 1; i++) {
	    copy_site (recv_buf[gpos], i,
		       getSite (shadow_buf[gpos], i + 1));	    
	  }
	}
	return recv_buf[gpos];
      }
      else {
	// inner sites: direction is important 
	typedef typename ForEach<A, EvalLeaf1Map, CTag>::Type_t TypeA_Map_t;
	typedef typename Combine1<TypeA_Map_t, FnMap, CTag>::Type_t Type_Map_t;

	EvalLeaf1Map fm (f.val1(), map);

	return Combine1<TypeA_Map_t, FnMap, CTag>::combine(ForEach<A, EvalLeaf1Map, CTag>::apply(expr.child(), fm, c),expr.operation(), c);
      }
    }
    else {
      // no communication needed
      typedef typename ForEach<A, EvalLeaf1Map, CTag>::Type_t TypeA_Map_t;
      typedef typename Combine1<TypeA_Map_t, FnMap, CTag>::Type_t Type_Map_t;

      EvalLeaf1Map fm (f.val1(), map);
      
      return Combine1<TypeA_Map_t, FnMap, CTag>::combine(ForEach<A, EvalLeaf1Map, CTag>::apply(expr.child(), fm, c),expr.operation(), c);
    }
  }
};


#if 0
template<class A, class CTag>
struct ForEach<UnaryNode<FnMap, A>, EvalInnerLeaf1, CTag>
{
  typedef typename ForEach<A, EvalInnerLeaf1, CTag>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnMap, CTag>::Type_t Type_t;
  inline static
  Type_t apply(const UnaryNode<FnMap, A> &expr, const EvalInnerLeaf1 &f, const CTag &c)
  {
    const Map& map = expr.operation().map;
    
    EvalInnerLeaf1 ff( map.goffsets[f.val1()]);
    return Combine1<TypeA_t, FnMap, CTag>::combine(ForEach<A, EvalInnerLeaf1, CTag>::apply(expr.child(), ff, c),expr.operation(), c);
  }
};
#endif

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
