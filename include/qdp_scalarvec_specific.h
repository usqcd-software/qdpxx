// -*- C++ -*-

/*! @file
 * @brief Outer/inner lattice routines specific to a scalarvec platform 
 */

#ifndef QDP_SCALARVEC_SPECIFIC_H
#define QDP_SCALARVEC_SPECIFIC_H

namespace QDP {

//-----------------------------------------------------------------------------
// Layout stuff specific to a scalarvec architecture
namespace Layout
{
  //! coord[mu]  <- mu  : fill with lattice coord in mu direction
  LatticeInteger latticeCoordinate(int mu);
}


//-----------------------------------------------------------------------------
// Internal ops designed to look like those in parscalar
// These dummy routines exist just to make code more portable
namespace QDPInternal
{
  //! Dummy array sum accross all nodes
  template<class T>
  inline void globalSumArray(T* dest, int n) {}

  //! Dummy global sum on a multi1d
  template<class T>
  inline void globalSumArray(multi1d<T>& dest) {}

  //! Dummy global sum on a multi2d
  template<class T>
  inline void globalSumArray(multi2d<T>& dest) {}

  //! Dummy sum across all nodes
  template<class T>
  inline void globalSum(T& dest) {}

  //! Dummy broadcast from primary node to all other nodes
  template<class T>
  inline void broadcast(T& dest) {}

  template<>
  inline void broadcast(std::string& dest) {}

  //! Dummy broadcast a string from primary node to all other nodes
  inline void broadcast_str(std::string& dest) {}

  //! Dummy broadcast from primary node to all other nodes
  inline void broadcast(void* dest, size_t nbytes) {}
}

//#define QDP_NOT_IMPLEMENTED

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
  // cerr << "In evaluateUnorderedSubet(olattice,oscalar)\n";

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest, op, rhs);
  prof.time -= getClockTime();
#endif

#ifndef QDP_NOT_IMPLEMENTED
  if (s.hasOrderedRep()) {
    const int istart = s.start() >> INNER_LOG;
    const int iend   = s.end()   >> INNER_LOG;
    int i = 0;

#pragma omp parallel for
    for(i=istart; i <= iend; ++i) {
      //    fprintf(stderr,"eval(olattice,oscalar): site %d\n",i);
      op(dest.elem(i), forEach(rhs, EvalLeaf1(0), OpCombine()));
    }
  }
  else {
    const int *tab = s.siteTable().slice();
    int j = 0;

#pragma omp parallel for
    for(j=0; j < s.numSiteTable(); ++j) {
      int i = tab[j];
      int outersite = i >> INNER_LOG;
      int innersite = i & ((1 << INNER_LOG)-1);

      //    fprintf(stderr,"eval(olattice,oscalar): site %d\n",i);
      op(dest.elem(i), forEach(rhs, EvalLeaf1(0), OpCombine()));
    }
  }    
#else
  QDP_error_exit("evaluateSubset not implemented, really?");
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
  // cerr << "In evaluateSubset(olattice,olattice)" << endl;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest, op, rhs);
  prof.time -= getClockTime();
#endif

#ifndef QDP_NOT_IMPLEMENTED
  // General form of loop structure
  if (s.hasOrderedRep()) {
    const int istart = s.start() >> INNER_LOG;
    const int iend   = s.end()   >> INNER_LOG;
    int i = 0;

#pragma omp parallel for
    for(i=istart; i <= iend; ++i) {
      //    fprintf(stderr,"eval(olattice,olattice): site %d\n",i);
      op(dest.elem(i), forEach(rhs, EvalLeaf1(i), OpCombine()));
    }
  }
  else {
    // this part of code have never been tested --Jie Chen
    const int *tab = s.siteTable().slice();
    int j;

    // list keeps track of outer sites
    multi1d<int> osites(s.numSiteTable() >> INNER_LOG);
    // make every element to be -1
    osites = -1;

#pragma omp parallel for
    for(j=0; j < s.numSiteTable(); ++j) {
      int i = tab[j];
      int outersite = i >> INNER_LOG;
      int innersite = i & ((1 << INNER_LOG)-1);
      //    fprintf(stderr,"eval(olattice,olattice): site %d\n",i);

      // check whether this outer sites has been done
      if (osites(outersite) == -1) {
	op(dest.elem(i), forEach(rhs, EvalLeaf1(i), OpCombine()));
	osites(outersite) = outersite;
      }
    }
  }    
#else
  QDP_error_exit("evaluateSubset not implemented, damn!");
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
  QDP_error_exit("copymask_Subset not implemented");
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
  //#error "random(unorderedsubset) broken"
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
  // The seed from any site is the same as the new global seed
  RNG::ran_seed = seed;  
#else
  QDP_error_exit("random_Subset not implemented");
#endif
}

//! dest  = random   under a subset
template<class T>
void random(const OSubLattice<T>& dd)
{
  OLattice<T>& d = const_cast<OSubLattice<T>&>(dd).field();
  const Subset& s = dd.subset();

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
  QDP_error_exit("gaussianSubset not implemented");
#endif
}


//! dest  = gaussian   under a subset
template<class T>
void gaussian(const OSubLattice<T>& dd)
{
  OLattice<T>& d = const_cast<OSubLattice<T>&>(dd).field();
  const Subset& s = dd.subset();

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
  QDP_error_exit("zero_rep_Subset not implemented");
#endif
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

  evaluate(d,OpAssign(),s1,all);   // since OScalar, no global sum needed

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

  evaluate(d,OpAssign(),s1,all);

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

      for (int i = low; i < high; i++) {
	tmp.elem() = forEach(s1, EvalLeaf1(i), OpCombine()); // Evaluate to ILattice part
	pd[myId].elem() += sum(tmp.elem());
      }
    }
    
    // Now combine all together
    for (int j = 0; j < num_threads; j++) 
      d.elem() += sum(pd[j].elem());

    delete []pd;

#if 0	
      // old code
    for(int i=istart; i <= iend; ++i) 
    {
      tmp.elem() = forEach(s1, EvalLeaf1(i), OpCombine()); // Evaluate to ILattice part
      d.elem() += sum(tmp.elem());    // sum as well the ILattice part
    }
#endif
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
	int innersite = i & ((1 << INNER_LOG)-1);

	tmp.elem() = forEach(s1, EvalLeaf1(outersite), OpCombine()); // Evaluate to ILattice part
	pd[myId].elem() += getSite(tmp.elem(),innersite);    // wasteful - only extract a single site worth
      }
    }

    // Now combine all together
    for (int k = 0; k < num_threads; k++) 
      d.elem() += sum(pd[k].elem());

    delete []pd;
  }

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
//! multi2d<OScalar> dest  = sumMulti(multi1d<OScalar>,Set) 
/*!
 * Compute the global sum on multiple subsets specified by Set 
 *
 * This implementation is specific to a purely olattice like
 * types. The scalar input value is replicated to all the
 * slices
 */
template<class RHS, class T>
multi1d<typename UnaryReturn<OScalar<T>, FnSum>::Type_t>
sumMulti(const QDPExpr<RHS,OScalar<T> >& s1, const Set& ss)
{
  multi2d<typename UnaryReturn<OScalar<T>, FnSum>::Type_t>  dest(s1.size(), ss.numSubsets());

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest(0,0), OpAssign(), FnSum(), s1);
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
    for(int n=0; n < s1.size(); ++n) {
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

  int i      = Layout::linearSiteIndex(coord);
  int iouter = i >> INNER_LOG;
  int iinner = i & ((1 << INNER_LOG)-1);

  dest.elem() = getSite(l.elem(iouter), iinner);
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
  int i      = Layout::linearSiteIndex(coord);
  int iouter = i >> INNER_LOG;
  int iinner = i & ((1 << INNER_LOG)-1);
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
// Empty map
struct FnMap;


//----------------------------------------------------------------------------
// Leaf functor containing an array of offsets
//----------------------------------------------------------------------------
struct EvalLeaf3Array
{
  int i1_m, i2_m, i3_m;
  const int *goffsets;
  inline EvalLeaf3Array (int i1, int i2, int i3, const int* offsets) 
    : i1_m(i1), i2_m(i2), i3_m(i3), goffsets(offsets) { }
  inline int val1() const { return i1_m; }
  inline int val2() const { return i2_m; }
  inline int val3() const { return i3_m; }
  inline const int* offsets() const { return goffsets; }
};


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
  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPType<T1,C1> & l)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(goffsets.slice()),
	CreateLeaf<QDPType<T1,C1> >::make(l)));
    }


  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPExpr<T1,C1> & l)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(goffsets.slice()),
	CreateLeaf<QDPExpr<T1,C1> >::make(l)));
    }

public:
  //! Accessor to offsets
  const multi1d<int>& Offsets() const {return goffsets;}

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
};//! General permutation map class for communications

struct FnMap
{
  PETE_EMPTY_CONSTRUCTORS(FnMap)

  const int *goff;
  int isign, dir;

  FnMap(const int *goffsets): goff(goffsets), isign(0), dir(0)
  {
    //    fprintf(stderr,"FnMap(): goff=0x%x\n",goff);
  }

  FnMap(const int *goffsets, int sign, int d): 
    goff(goffsets), isign(sign), dir(d)
  {
    // fprintf(stderr,"FnMap(): goff=0x%x\n",goff);
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


//-----------------------------------------------------------------------------
// Specialization of LeafFunctor class for applying the EvalLeaf3Array
// tag to a QDPType. The apply method simply returns the array
// evaluated at the point.
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Currently, x-direction is the direction that is vectorized. Any shift
// along the other directions can be easilly done by using nearest neighbor
// vectorized data. For shifting along x direction, we need to reconstruct
// a new element containing inner-data
//-----------------------------------------------------------------------------
template<class T, class C>
struct LeafFunctor<QDPType<T,C>, EvalLeaf3Array>
{
  typedef Reference<T> Type_t;
//  typedef T Type_t;
  inline static Type_t apply(const QDPType<T,C> &a, const EvalLeaf3Array &f)
  { 
    if (f.val3() != 0) { // not along x direction
      // f.val1() is an old outer site, convert to new neighbor real site
      int oldrealsite = f.val1() << INNER_LOG;
      int newrealsite = f.offsets()[oldrealsite];
      // Y, Z, T neighbors are boundled together
      return Type_t(a.elem(newrealsite >> INNER_LOG));
    }
    else {
      T ret;
      const int *goffsets = f.offsets();
      // old real site 
      int i = f.val1() << INNER_LOG;
      
#if INNER_LOG == 2
      int o0 = goffsets[i+0] >> INNER_LOG;
      int i0 = goffsets[i+0] & (INNER_LEN - 1);

      int o1 = goffsets[i+1] >> INNER_LOG;
      int i1 = goffsets[i+1] & (INNER_LEN - 1);

      int o2 = goffsets[i+2] >> INNER_LOG;
      int i2 = goffsets[i+2] & (INNER_LEN - 1);

      int o3 = goffsets[i+3] >> INNER_LOG;
      int i3 = goffsets[i+3] & (INNER_LEN - 1);

      // Gather 4 inner-grid sites together
      gather_sites(ret,
		   a.elem(o0),i0,
		   a.elem(o1),i1,
		   a.elem(o2),i2,
		   a.elem(o3),i3);

#elif INNER_LOG == 3
      int o0 = goffsets[i+0] >> INNER_LOG;
      int i0 = goffsets[i+0] & (INNER_LEN - 1);

      int o1 = goffsets[i+1] >> INNER_LOG;
      int i1 = goffsets[i+1] & (INNER_LEN - 1);

      int o2 = goffsets[i+2] >> INNER_LOG;
      int i2 = goffsets[i+2] & (INNER_LEN - 1);

      int o3 = goffsets[i+3] >> INNER_LOG;
      int i3 = goffsets[i+3] & (INNER_LEN - 1);

      int o4 = goffsets[i+4] >> INNER_LOG;
      int i4 = goffsets[i+4] & (INNER_LEN - 1);
      
      int o5 = goffsets[i+5] >> INNER_LOG;
      int i5 = goffsets[i+5] & (INNER_LEN - 1);
      
      int o6 = goffsets[i+6] >> INNER_LOG;
      int i6 = goffsets[i+6] & (INNER_LEN - 1);

      int o7 = goffsets[i+7] >> INNER_LOG;
      int i7 = goffsets[i+7] & (INNER_LEN - 1);
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
      int o0 = goffsets[i+0] >> INNER_LOG;
      int i0 = goffsets[i+0] & (INNER_LEN - 1);

      int o1 = goffsets[i+1] >> INNER_LOG;
      int i1 = goffsets[i+1] & (INNER_LEN - 1);
      
      int o2 = goffsets[i+2] >> INNER_LOG;
      int i2 = goffsets[i+2] & (INNER_LEN - 1);

      int o3 = goffsets[i+3] >> INNER_LOG;
      int i3 = goffsets[i+3] & (INNER_LEN - 1);

      int o4 = goffsets[i+4] >> INNER_LOG;
      int i4 = goffsets[i+4] & (INNER_LEN - 1);

      int o5 = goffsets[i+5] >> INNER_LOG;
      int i5 = goffsets[i+5] & (INNER_LEN - 1);

      int o6 = goffsets[i+6] >> INNER_LOG;
      int i6 = goffsets[i+6] & (INNER_LEN - 1);

      int o7 = goffsets[i+7] >> INNER_LOG;
      int i7 = goffsets[i+7] & (INNER_LEN - 1);

      int o8 = goffsets[i+8] >> INNER_LOG;
      int i8 = goffsets[i+8] & (INNER_LEN - 1);

      int o9 = goffsets[i+9] >> INNER_LOG;
      int i9 = goffsets[i+9] & (INNER_LEN - 1);

      int o10 = goffsets[i+10] >> INNER_LOG;
      int i10 = goffsets[i+10] & (INNER_LEN - 1);

      int o11 = goffsets[i+11] >> INNER_LOG;
      int i11 = goffsets[i+11] & (INNER_LEN - 1);

      int o12 = goffsets[i+12] >> INNER_LOG;
      int i12 = goffsets[i+12] & (INNER_LEN - 1);

      int o13 = goffsets[i+13] >> INNER_LOG;
      int i13 = goffsets[i+13] & (INNER_LEN - 1);

      int o14 = goffsets[i+14] >> INNER_LOG;
      int i14 = goffsets[i+14] & (INNER_LEN - 1);

      int o15 = goffsets[i+15] >> INNER_LOG;
      int i15 = goffsets[i+15] & (INNER_LEN - 1);

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
      return Type_t(ret);	
    }
  }
};


template<class T>
struct LeafFunctor<OScalar<T>, EvalLeaf3Array>
{
//  typedef T Type_t;
  typedef Reference<T> Type_t;
  inline static Type_t apply(const OScalar<T> &a, const EvalLeaf3Array &f)
    {return Type_t(a.elem());}
};

template<class T>
struct LeafFunctor<OLattice<T>, EvalLeaf3Array>
{
//  typedef T Type_t;
  typedef Reference<T> Type_t;

  inline static Type_t apply(const OLattice<T> &a, const EvalLeaf3Array &f)
  {
    if (f.val3() != 0) { // not along x direction
      // f.val1() is an old outer site, convert to new neighbor real site
      int oldrealsite = f.val1() << INNER_LOG;
      int newrealsite = f.offsets()[oldrealsite];
      // Y, Z, T neighbors are boundled together
      return Type_t(a.elem(newrealsite >> INNER_LOG));
    }
    else {
      T ret;
      const int *goffsets = f.offsets();
      // old real site 
      int i = f.val1() << INNER_LOG;
      
#if INNER_LOG == 2
      int o0 = goffsets[i+0] >> INNER_LOG;
      int i0 = goffsets[i+0] & (INNER_LEN - 1);

      int o1 = goffsets[i+1] >> INNER_LOG;
      int i1 = goffsets[i+1] & (INNER_LEN - 1);

      int o2 = goffsets[i+2] >> INNER_LOG;
      int i2 = goffsets[i+2] & (INNER_LEN - 1);

      int o3 = goffsets[i+3] >> INNER_LOG;
      int i3 = goffsets[i+3] & (INNER_LEN - 1);

      // Gather 4 inner-grid sites together
      gather_sites(ret,
		   a.elem(o0),i0,
		   a.elem(o1),i1,
		   a.elem(o2),i2,
		   a.elem(o3),i3);

#elif INNER_LOG == 3
      int o0 = goffsets[i+0] >> INNER_LOG;
      int i0 = goffsets[i+0] & (INNER_LEN - 1);

      int o1 = goffsets[i+1] >> INNER_LOG;
      int i1 = goffsets[i+1] & (INNER_LEN - 1);

      int o2 = goffsets[i+2] >> INNER_LOG;
      int i2 = goffsets[i+2] & (INNER_LEN - 1);

      int o3 = goffsets[i+3] >> INNER_LOG;
      int i3 = goffsets[i+3] & (INNER_LEN - 1);

      int o4 = goffsets[i+4] >> INNER_LOG;
      int i4 = goffsets[i+4] & (INNER_LEN - 1);
      
      int o5 = goffsets[i+5] >> INNER_LOG;
      int i5 = goffsets[i+5] & (INNER_LEN - 1);
      
      int o6 = goffsets[i+6] >> INNER_LOG;
      int i6 = goffsets[i+6] & (INNER_LEN - 1);

      int o7 = goffsets[i+7] >> INNER_LOG;
      int i7 = goffsets[i+7] & (INNER_LEN - 1);
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
      int o0 = goffsets[i+0] >> INNER_LOG;
      int i0 = goffsets[i+0] & (INNER_LEN - 1);

      int o1 = goffsets[i+1] >> INNER_LOG;
      int i1 = goffsets[i+1] & (INNER_LEN - 1);
      
      int o2 = goffsets[i+2] >> INNER_LOG;
      int i2 = goffsets[i+2] & (INNER_LEN - 1);

      int o3 = goffsets[i+3] >> INNER_LOG;
      int i3 = goffsets[i+3] & (INNER_LEN - 1);

      int o4 = goffsets[i+4] >> INNER_LOG;
      int i4 = goffsets[i+4] & (INNER_LEN - 1);

      int o5 = goffsets[i+5] >> INNER_LOG;
      int i5 = goffsets[i+5] & (INNER_LEN - 1);

      int o6 = goffsets[i+6] >> INNER_LOG;
      int i6 = goffsets[i+6] & (INNER_LEN - 1);

      int o7 = goffsets[i+7] >> INNER_LOG;
      int i7 = goffsets[i+7] & (INNER_LEN - 1);

      int o8 = goffsets[i+8] >> INNER_LOG;
      int i8 = goffsets[i+8] & (INNER_LEN - 1);

      int o9 = goffsets[i+9] >> INNER_LOG;
      int i9 = goffsets[i+9] & (INNER_LEN - 1);

      int o10 = goffsets[i+10] >> INNER_LOG;
      int i10 = goffsets[i+10] & (INNER_LEN - 1);

      int o11 = goffsets[i+11] >> INNER_LOG;
      int i11 = goffsets[i+11] & (INNER_LEN - 1);

      int o12 = goffsets[i+12] >> INNER_LOG;
      int i12 = goffsets[i+12] & (INNER_LEN - 1);

      int o13 = goffsets[i+13] >> INNER_LOG;
      int i13 = goffsets[i+13] & (INNER_LEN - 1);

      int o14 = goffsets[i+14] >> INNER_LOG;
      int i14 = goffsets[i+14] & (INNER_LEN - 1);

      int o15 = goffsets[i+15] >> INNER_LOG;
      int i15 = goffsets[i+15] & (INNER_LEN - 1);

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
      return Type_t(ret);	
    }
  }
};

// Specialization of ForEach deals with maps. 
template<class A, class CTag>
struct ForEach<UnaryNode<FnMap, A>, EvalLeaf1, CTag>
{
  typedef typename ForEach<A, EvalLeaf3Array, CTag>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnMap, CTag>::Type_t Type_t;
  inline static
  Type_t apply(const UnaryNode<FnMap, A> &expr, const EvalLeaf1 &f, 
    const CTag &c) 
  {
    // expr.operation() is fnmap which contains offsets, sign and direction
    // f.val1() is pointing to outersites: check evaluate function above:
    // --Jie Chen
    int isign = expr.operation().isign;
    int dir = expr.operation().dir;

    // passing offsets pointer as well
    EvalLeaf3Array ff(f.val1(),isign, dir, expr.operation().goff);
    
    // fprintf(stderr,"ForEach<Unary<FnMap>>: outersite = %d, sign = %d dir = %d offsets = %p\n", ff.val1(), ff.val1(), ff.val2(), ff.offsets() );

    return Combine1<TypeA_t, FnMap, CTag>::
      combine(ForEach<A, EvalLeaf3Array, CTag>::apply(expr.child(), ff, c),
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
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(mapsa[dir].Offsets().slice()),
	CreateLeaf<QDPType<T1,C1> >::make(l)));
    }


  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPExpr<T1,C1> & l, int dir)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(mapsa[dir].Offsets().slice()),
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
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(bimaps[(isign+1)>>1].Offsets().slice()),
	CreateLeaf<QDPType<T1,C1> >::make(l)));
    }


  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPExpr<T1,C1> & l, int isign)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(bimaps[(isign+1)>>1].Offsets().slice()),
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
  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPType<T1,C1> & l, int isign, int dir)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(bimapsa((isign+1)>>1,dir).Offsets().slice(), isign, dir),
	CreateLeaf<QDPType<T1,C1> >::make(l)));
    }


  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPExpr<T1,C1> & l, int isign, int dir)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(bimapsa((isign+1)>>1,dir).Offsets().slice(), isign, dir),
	CreateLeaf<QDPExpr<T1,C1> >::make(l)));
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

//! Decompose a lexicographic site into coordinates
multi1d<int> crtesn(int ipos, const multi1d<int>& latt_size);

//! XML output
template<class T>  
XMLWriter& operator<<(XMLWriter& xml, const OLattice<T>& d)
{
  xml.openTag("OLattice");

  XMLWriterAPI::AttributeList alist;

  const int iend = Layout::vol();
  for(int site=0; site < iend; ++site) 
  {
    multi1d<int> coord = crtesn(site, Layout::lattSize());
    std::ostringstream os;
    os << coord[0];
    for(int i=1; i < coord.size(); ++i)
      os << " " << coord[i];

    int i = Layout::linearSiteIndex(site);
    int outersite = i >> INNER_LOG;
    int innersite = i & ((1 << INNER_LOG)-1);

    alist.clear();
    alist.push_back(XMLWriterAPI::Attribute("site", site));
    alist.push_back(XMLWriterAPI::Attribute("coord", os.str()));

    xml.openTag("elem", alist);
    xml << getSite(d.elem(outersite),innersite);
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
  bin.writeArray((const char *)&(d.elem()), 
		 sizeof(typename WordType<T>::Type_t), 
		 sizeof(T) / sizeof(typename WordType<T>::Type_t));
}

//! Binary output
/*! An inner grid is assumed */
template<class T>  
void write(BinaryWriter& bin, const OLattice<T>& d)
{
  const int iend = Layout::vol();
  for(int site=0; site < iend; ++site) 
  {
    int i = Layout::linearSiteIndex(site);
    int outersite = i >> INNER_LOG;
    int innersite = i & ((1 << INNER_LOG)-1);

    typedef typename UnaryReturn<T, FnGetSite>::Type_t  Site_t;
    Site_t  this_site = getSite(d.elem(outersite),innersite);

    bin.writeArray((const char*)&this_site,
		   sizeof(typename WordType<Site_t>::Type_t), 
		   sizeof(Site_t) / sizeof(typename WordType<Site_t>::Type_t));
  }
}

//! Write a single site from coord
/*! An inner grid is assumed */
template<class T>  
void write(BinaryWriter& bin, const OLattice<T>& d, const multi1d<int>& coord)
{
  int i = Layout::linearSiteIndex(coord);
  int outersite = i >> INNER_LOG;
  int innersite = i & ((1 << INNER_LOG)-1);

  typedef typename UnaryReturn<T, FnGetSite>::Type_t  Site_t;
  Site_t  this_site = getSite(d.elem(outersite),innersite);

  bin.writeArray((const char*)&this_site,
		 sizeof(typename WordType<Site_t>::Type_t), 
		 sizeof(Site_t) / sizeof(typename WordType<Site_t>::Type_t));
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
/*! An inner grid is assumed */
template<class T>  
void read(BinaryReader& bin, OLattice<T>& d)
{
  const int iend = Layout::vol();
  for(int site=0; site < iend; ++site) 
  {
    int i = Layout::linearSiteIndex(site);
    int outersite = i >> INNER_LOG;
    int innersite = i & ((1 << INNER_LOG)-1);

    typedef typename UnaryReturn<T, FnGetSite>::Type_t  Site_t;
    Site_t  this_site;

    bin.readArray((char*)&this_site,
		  sizeof(typename WordType<Site_t>::Type_t), 
		  sizeof(Site_t) / sizeof(typename WordType<Site_t>::Type_t));

    copy_site(d.elem(outersite), innersite, this_site);
  }
}

//! Read a single site and place it at coord
/*! An inner grid is assumed */
template<class T>  
void read(BinaryReader& bin, OLattice<T>& d, const multi1d<int>& coord)
{
  int i = Layout::linearSiteIndex(coord);
  int outersite = i >> INNER_LOG;
  int innersite = i & ((1 << INNER_LOG)-1);

  typedef typename UnaryReturn<T, FnGetSite>::Type_t  Site_t;
  Site_t  this_site;

  bin.readArray((char*)&this_site,
		sizeof(typename WordType<Site_t>::Type_t), 
		sizeof(Site_t) / sizeof(typename WordType<Site_t>::Type_t));

  copy_site(d.elem(outersite), innersite, this_site);
}

// **************************************************************
// Special support for slices of a lattice
// **************************************************************
namespace LatticeTimeSliceIO 
{
  template<class T>
  void readSlice(BinaryReader& bin, OLattice<T>& data, 
		 int start_lexico, int stop_lexico)
  {
    // check whether dimention t is multiple of INNER_LEN
    int tDir = Nd-1;

    if (Layout::lattSize()[tDir] % INNER_LEN != 0)
      QDP_error_exit ("Size of time dimension %d is not multiple of vector len %d\n", Layout::lattSize()[tDir], INNER_LEN);


    for(int site=start_lexico; site < stop_lexico; ++site) 
    {
      int i = Layout::linearSiteIndex(site);
      int outersite = i >> INNER_LOG;
      int innersite = i & ((1 << INNER_LOG)-1);

      typedef typename UnaryReturn<T, FnGetSite>::Type_t  Site_t;
      Site_t  this_site;

      bin.readArray((char*)&this_site,
		    sizeof(typename WordType<Site_t>::Type_t), 
		    sizeof(Site_t) / sizeof(typename WordType<Site_t>::Type_t));

      copy_site(data.elem(outersite), innersite, this_site);
    }
  }

  template<class T>
  void writeSlice(BinaryWriter& bin, OLattice<T>& data, 
		  int start_lexico, int stop_lexico)
  {
    // check whether dimention t is multiple of INNER_LEN
    int tDir = Nd-1;

    if (Layout::lattSize()[tDir] % INNER_LEN != 0)
      QDP_error_exit("Size of time dimension %d is not multiple of vector len %d\n", Layout::lattSize()[tDir], INNER_LEN);

    for(int site=start_lexico; site < stop_lexico; ++site) 
    {
      int i = Layout::linearSiteIndex(site);
      int outersite = i >> INNER_LOG;
      int innersite = i & ((1 << INNER_LOG)-1);

      typedef typename UnaryReturn<T, FnGetSite>::Type_t  Site_t;
      Site_t  this_site = getSite(data.elem(outersite),innersite);

      bin.writeArray((const char*)&this_site,
		     sizeof(typename WordType<Site_t>::Type_t), 
		     sizeof(Site_t) / sizeof(typename WordType<Site_t>::Type_t));
    }
  }

} // namespace LatticeTimeSliceIO

} // namespace QDP

#endif
