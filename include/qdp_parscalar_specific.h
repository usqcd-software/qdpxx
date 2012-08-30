// -*- C++ -*-

/*! @file
 * @brief Outer lattice routines specific to a parallel platform with scalar layout
 */

#ifndef QDP_PARSCALAR_SPECIFIC_H
#define QDP_PARSCALAR_SPECIFIC_H

#include "qmp.h"

namespace QDP {


// Use separate defs here. This will cause subroutine calls under g++

//-----------------------------------------------------------------------------
// Layout stuff specific to a parallel architecture
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
  /*! All nodes participate */
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
    for(int i=0; i < len; i++, dest++)
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

  //! Global sum on a multi1d
  template<class T>
  inline void globalSumArray(multi1d<T>& dest)
  {
    // The implementation here is relying on the structure being packed
    // tightly in memory - no padding
    typedef typename WordType<T>::Type_t  W;   // find the machine word type

#if 0
    QDPIO::cout << "sizeof(T) = " << sizeof(T) << endl;
    QDPIO::cout << "sizeof(W) = " << sizeof(W) << endl;
    QDPIO::cout << "Calling multi1d global sum array with length " << dest.size()*sizeof(T)/sizeof(W) << endl;
#endif
    globalSumArray((W *)dest.slice(), dest.size()*sizeof(T)/sizeof(W)); // call appropriate hook
  }

  //! Global sum on a multi2d
  template<class T>
  inline void globalSumArray(multi2d<T>& dest)
  {
    // The implementation here is relying on the structure being packed
    // tightly in memory - no padding
    typedef typename WordType<T>::Type_t  W;   // find the machine word type

#if 0
    QDPIO::cout << "sizeof(T) = " << sizeof(T) << endl;
    QDPIO::cout << "sizeof(W) = " << sizeof(W) << endl;
    QDPIO::cout << "Calling multi2d global sum array with length " << dest.size1()*dest.size2()*sizeof(T)/sizeof(W) << endl;
#endif
    // call appropriate hook
    globalSumArray((W *)dest.slice(0), dest.size1()*dest.size2()*sizeof(T)/sizeof(W));
  }

  //! Sum across all nodes
  template<class T>
  inline void globalSum(T& dest)
  {
    // The implementation here is relying on the structure being packed
    // tightly in memory - no padding
    typedef typename WordType<T>::Type_t  W;   // find the machine word type

#if 0 
    QDPIO::cout << "sizeof(T) = " << sizeof(T) << endl;
    QDPIO::cout << "sizeof(W) = " << sizeof(W) << endl;
    QDPIO::cout << "Calling global sum array with length " << sizeof(T)/sizeof(W) << endl;
#endif
    globalSumArray((W *)&dest, int(sizeof(T)/sizeof(W))); // call appropriate hook
  }

  template<>
  inline void globalSum(double& dest)
  {
#if 0 
    QDPIO::cout << "Using simple sum_double" << endl;
#endif
    QMP_sum_double(&dest);
  }


  //! Low level hook to QMP_max_double
  inline void globalMaxValue(float* dest)
  {
    QMP_max_float(dest);
  }

  //! Low level hook to QMP_max_double
  inline void globalMaxValue(double* dest)
  {
    QMP_max_double(dest);
  }

  //! Global max across all nodes
  template<class T>
  inline void globalMax(T& dest)
  {
    typedef typename WordType<T>::Type_t  W;   // find the machine word type

    globalMaxValue((W *)&dest);
  }


  //! Low level hook to QMP_min_float
  inline void globalMinValue(float* dest)
  {
    QMP_min_float(dest);
  }

  //! Low level hook to QMP_min_double
  inline void globalMinValue(double* dest)
  {
    QMP_min_double(dest);
  }

  //! Global min across all nodes
  template<class T>
  inline void globalMin(T& dest)
  {
    typedef typename WordType<T>::Type_t  W;   // find the machine word type

    globalMinValue((W *)&dest);
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

/////////////////////////////////////////////////////////
// Threading evaluate with openmp and qmt implementation
//
// by Xu Guo, EPCC, 16 June 2008
/////////////////////////////////////////////////////////

//! user argument for the evaluate function:
// "OLattice Op Scalar(Expression(source)) under an Subset"
//
template<class T, class T1, class Op, class RHS>
struct u_arg{
    u_arg(
        OLattice<T>& d_,
        const QDPExpr<RHS,OScalar<T1> >& r_,
        const Op& op_,
        const int *tab_
    ) : d(d_), r(r_), op(op_), tab(tab_) {}
    
    OLattice<T>& d;
    const QDPExpr<RHS,OScalar<T1> >& r;
    const Op& op;
    const int *tab;
   };

//! user function for the evaluate function:
// "OLattice Op Scalar(Expression(source)) under an Subset"
//
template<class T, class T1, class Op, class RHS>
void ev_userfunc(int lo, int hi, int myId, u_arg<T,T1,Op,RHS> *a)
{
   OLattice<T>& dest = a->d;
   const QDPExpr<RHS,OScalar<T1> >&rhs = a->r;
   const int* tab = a->tab;
   const Op& op= a->op;

      
   for(int j=lo; j < hi; ++j)
   {
     int i = tab[j];
     op(dest.elem(i), forEach(rhs, EvalLeaf1(0), OpCombine()));
   }
}

//! user argument for the evaluate function:
// "OLattice Op OLattice(Expression(source)) under an Subset"
//
template<class T, class T1, class Op, class RHS>
struct user_arg{
    user_arg(
        OLattice<T>& d_,
        const QDPExpr<RHS,OLattice<T1> >& r_,
        const Op& op_,
        const int *tab_ ) : d(d_), r(r_), op(op_), tab(tab_) {}

        OLattice<T>& d;
        const QDPExpr<RHS,OLattice<T1> >& r;
        const Op& op;
        const int *tab;
   };

//! user function for the evaluate function:
// "OLattice Op OLattice(Expression(source)) under an Subset"
//
template<class T, class T1, class Op, class RHS>
void evaluate_userfunc(int lo, int hi, int myId, user_arg<T,T1,Op,RHS> *a)
{

   OLattice<T>& dest = a->d;
   const QDPExpr<RHS,OLattice<T1> >&rhs = a->r;
   const int* tab = a->tab;
   const Op& op= a->op;

      
   for(int j=lo; j < hi; ++j)
   {
     int i = tab[j];
     op(dest.elem(i), forEach(rhs, EvalLeaf1(i), OpCombine()));
   }
}


template<class T, class T1, class Op, class RHS>
struct user_arg_mask{
    user_arg_mask(
        OLattice<T>& d_,
        const QDPExpr<RHS,OLattice<T1> >& r_,
        const Op& op_,
        const int *tab_ ,
	const bool *mask_ ) : d(d_), r(r_), op(op_), tab(tab_), mask(mask_) {}

        OLattice<T>& d;
        const QDPExpr<RHS,OLattice<T1> >& r;
        const Op& op;
        const int *tab;
        const bool *mask;
   };


template<class T, class T1, class Op, class RHS>
void evaluate_userfunc_mask(int lo, int hi, int myId, user_arg_mask<T,T1,Op,RHS> *a)
{

   OLattice<T>& dest = a->d;
   const QDPExpr<RHS,OLattice<T1> >&rhs = a->r;
   const int* tab = a->tab;
   const bool* mask = a->mask;
   const Op& op= a->op;
      
   for(int j=lo; j < hi; ++j)
   {
     if (mask[ tab[j] ])
       op(dest.elem(tab[j]), forEach(rhs, EvalLeaf1(tab[j]), OpCombine()));
   }
}



template<class T, class T1, class Op, class RHS>
struct user_arg_face{
    user_arg_face(
        T*& d_,
	const int* tab_ ,
        const QDPExpr<RHS,OLattice<T1> >& r_,
        const Op& op_ ) : d(d_), r(r_), op(op_), tab(tab_) {}

        T*& d;
        const QDPExpr<RHS,OLattice<T1> >& r;
        const int *tab;
        const Op& op;
   };

//! user function for the evaluate function:
// "OLattice Op OLattice(Expression(source)) under an Subset"
//
template<class T, class T1, class Op, class RHS>
void evaluate_userfunc_face(int lo, int hi, int myId, user_arg_face<T,T1,Op,RHS> *a)
{

   T*& dest = a->d;
   const QDPExpr<RHS,OLattice<T1> >&rhs = a->r;
   const Op& op= a->op;
   const int* tab = a->tab;
      
   for(int j=lo; j < hi; ++j)
   {
     int i = tab[j];
     op(dest[j], forEach(rhs, EvalLeaf1(i), OpCombine()));
   }
}




//! include the header file for dispatch
#include "qdp_dispatch.h"



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
  //  cout << __PRETTY_FUNCTION__ << "\n";

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest, op, rhs);
  prof.time -= getClockTime();
#endif

  int numSiteTable = s.numSiteTable();
  u_arg<T,T1,Op,RHS> a(dest, rhs, op, s.siteTable().slice());
  dispatch_to_threads< u_arg<T,T1,Op,RHS> >(numSiteTable, a, ev_userfunc);

  ///////////////////
  // Original code
  //////////////////

  //const int *tab = s.siteTable().slice();
  //for(int j=0; j < s.numSiteTable(); ++j) 
  //{
  //int i = tab[j];
//    fprintf(stderr,"eval(olattice,oscalar): site %d\n",i);
//    op(dest.elem(i), forEach(rhs, ElemLeaf(), OpCombine()));
  //op(dest.elem(i), forEach(rhs, EvalLeaf1(0), OpCombine()));
  //}

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif
}




struct ShiftPhase1
{
};

struct ShiftPhase2
{
};


// template<class T, class C>
// struct LeafFunctor<QDPType<T,C>, ShiftPhase1>
// {
//   typedef int Type_t;
//   static int apply(const QDPType<T,C> &s, const ShiftPhase1 &f)
//     {
//       return 0;
//     }
// };

// template<class T>
// struct LeafFunctor<OScalar<T>, ShiftPhase1>
// {
//   typedef int Type_t;
//   static int apply(const OScalar<T> &s, const ShiftPhase1 &f)
//     { 
//       return 0;
//     }
// };


// template<class A, class B, class Op>
// struct Combine2<A,B,Op,NullCombine>
// {
//   typedef int Type_t;
//   static Type_t combine(const A &a,const B &b, const Op &, const NullCombine &) { return 0; }
// };







template<class T>
struct LeafFunctor<OScalar<T>, ShiftPhase1>
{
  typedef int Type_t;
  inline static Type_t apply(const OScalar<T> &a, const ShiftPhase1 &f) {
    return 0;
  }
};

template<class T>
struct LeafFunctor<OLattice<T>, ShiftPhase1>
{
  typedef int Type_t;
  inline static Type_t apply(const OLattice<T> &a, const ShiftPhase1 &f) {
    return 0;
  }
};

template<class T, class C>
struct LeafFunctor<QDPType<T,C>, ShiftPhase1>
{
  typedef int Type_t;
  static int apply(const QDPType<T,C> &s, const ShiftPhase1 &f) {
    return 0;
  }
};


template<class T>
struct LeafFunctor<OScalar<T>, ShiftPhase2>
{
  typedef int Type_t;
  inline static Type_t apply(const OScalar<T> &a, const ShiftPhase2 &f) {
    return 0;
  }
};

template<class T>
struct LeafFunctor<OLattice<T>, ShiftPhase2>
{
  typedef int Type_t;
  inline static Type_t apply(const OLattice<T> &a, const ShiftPhase2 &f) {
    return 0;
  }
};

template<class T, class C>
struct LeafFunctor<QDPType<T,C>, ShiftPhase2>
{
  typedef int Type_t;
  static int apply(const QDPType<T,C> &s, const ShiftPhase2 &f) {
    return 0;
  }
};




// template<class T, class C>
// struct LeafFunctor<QDPType<T,C>, ShiftPhase2>
// {
//   typedef int Type_t;
//   static int apply(const QDPType<T,C> &s, const ShiftPhase2 &f)
//     {
//       return 0;
//     }
// };

// template<class T>
// struct LeafFunctor<OScalar<T>, ShiftPhase2>
// {
//   typedef int Type_t;
//   static int apply(const OScalar<T> &s, const ShiftPhase2 &f)
//     { 
//       return 0;
//     }
// };

template<int N>
struct LeafFunctor<GammaType<N>, ShiftPhase1>
{
  typedef int Type_t;
  static int apply(const GammaType<N> &s, const ShiftPhase1 &f) { return 0; }
};

template<int N, int m>
struct LeafFunctor<GammaConst<N,m>, ShiftPhase1>
{
  typedef int Type_t;
  static int apply(const GammaConst<N,m> &s, const ShiftPhase1 &f) { return 0; }
};

template<int N>
struct LeafFunctor<GammaType<N>, ShiftPhase2>
{
  typedef int Type_t;
  static int apply(const GammaType<N> &s, const ShiftPhase2 &f) { return 0; }
};

template<int N, int m>
struct LeafFunctor<GammaConst<N,m>, ShiftPhase2>
{
  typedef int Type_t;
  static int apply(const GammaConst<N,m> &s, const ShiftPhase2 &f) { return 0; }
};


template<class Op, class A, class B, class FTag>
struct ForEach<BinaryNode<Op, A, B>, FTag, NullCombine >
{
  typedef int Type_t;
  inline static
  Type_t apply(const BinaryNode<Op, A, B> &expr, const FTag &f,
	       const NullCombine &c)
  {
    ForEach<A, FTag, NullCombine>::apply(expr.left(), f, c);
    ForEach<B, FTag, NullCombine>::apply(expr.right(), f, c);
    return 0;
  }
};






// template<class T>
// struct LeafFunctor<T, ShiftPhase1>
// {
//   typedef int Type_t;
//   template<class C>
//   static int apply(const QDPType<T,C> &s, const ShiftPhase1 &f)
//   { 
//     return 0;
//     //return LeafFunctor<C,PrintTag>::apply(static_cast<const C&>(s),f);
//   }
// };



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
  // cout << __PRETTY_FUNCTION__ << "\n";

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest, op, rhs);
  prof.time -= getClockTime();
#endif

#if 1
  ShiftPhase1 phase1;
  int maps_involved = forEach(rhs, phase1 , BitOrCombine());
#endif
  //int maps_involved = 0;


  //QDP_info("eval(Lat) maps_involved = %d",maps_involved);
  
  if (maps_involved > 0) {

    //QDPIO::cout << "eval maps="<<maps_involved<<__PRETTY_FUNCTION__<<"\n";
#if 1
    const multi1d<int>& innerSites = MasterMap::Instance().getInnerSites(maps_involved);
    const multi1d<int>& faceSites = MasterMap::Instance().getFaceSites(maps_involved);

#if 0
    user_arg_mask<T,T1,Op,RHS> a0(dest, rhs, op, innerSites.slice() , s.getIsElement().slice() );
    dispatch_to_threads< user_arg_mask<T,T1,Op,RHS> >(innerSites.size(), a0, evaluate_userfunc_mask );
#else
    for (int j=0 ; j<innerSites.size() ; j++ ) {
      if (s.isElement(innerSites[j])) {
    	//QDP_info("inner site %d is element",innerSites[j]);
    	op(dest.elem(innerSites[j]), forEach(rhs, EvalInnerLeaf1(innerSites[j]), OpCombine()) );
      } // else QDP_info("inner site %d is not element",innerSites[j]);
    }
#endif
#endif

#if 1
    ShiftPhase2 phase2;
    forEach(rhs, phase2 , NullCombine());
#endif

    //QDP_info("face sites total = %d",faceSites.size());

#if 1
#if 1
    user_arg_mask<T,T1,Op,RHS> a1(dest, rhs, op, faceSites.slice() , s.getIsElement().slice() );
    dispatch_to_threads< user_arg_mask<T,T1,Op,RHS> >(faceSites.size(), a1, evaluate_userfunc_mask );
#else
    for (int j=0 ; j<faceSites.size() ; j++ ) {
      if (s.isElement(faceSites[j])) {
    	//QDP_info("face site %d is element",faceSites[j]);
    	op(dest.elem(faceSites[j]), forEach(rhs, EvalLeaf1(faceSites[j]), OpCombine()) );
      } // else QDP_info("face site %d is not element",faceSites[j]);
    }
#endif
#endif



  } else {

    // if (Layout::primaryNode())
    //   QDP_info("eval(Lat)");

    //QDPIO::cout << "eval maps=0 \n";
#if 1
#if 0
    int numSiteTable = s.numSiteTable();
    user_arg<T,T1,Op,RHS> a(dest, rhs, op, s.siteTable().slice());
    dispatch_to_threads< user_arg<T,T1,Op,RHS> >(numSiteTable, a, evaluate_userfunc);
#else
    const int *tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); ++j) 
      {
    	int i = tab[j];
    	op(dest.elem(i), forEach(rhs, EvalLeaf1(i), OpCombine()) );
      }
#endif
#endif
  }


  ////////////////////
  // Original code
  ///////////////////

  // General form of loop structure
  //const int *tab = s.siteTable().slice();
  //for(int j=0; j < s.numSiteTable(); ++j) 
  //{
  //int i = tab[j];
//    fprintf(stderr,"eval(olattice,olattice): site %d\n",i);
  //op(dest.elem(i), forEach(rhs, EvalLeaf1(i), OpCombine()));
  //}

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif
}


//-----------------------------------------------------------------------------
//! dest = (mask) ? s1 : dest
template<class T1, class T2>
void copymask(OSubLattice<T2> d, const OLattice<T1>& mask, const OLattice<T2>& s1) 
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
void copymask(OLattice<T2>& dest, const OLattice<T1>& mask, const OLattice<T2>& s1) 
{
  int nodeSites = Layout::sitesOnNode();
  for(int i=0; i < nodeSites; ++i) 
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
  Seed skewed_seed;

  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    seed = RNG::ran_seed;
    skewed_seed.elem() = RNG::ran_seed.elem() * RNG::lattice_ran_mult->elem(i);
    fill_random(d.elem(i), seed, skewed_seed, RNG::ran_mult_n);
  }

  RNG::ran_seed = seed;  // The seed from any site is the same as the new global seed
}



//! dest  = random   under a subset
template<class T>
void random(OSubLattice<T> dd)
{
  OLattice<T>& d = dd.field();
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

  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    fill_gaussian(d.elem(i), r1.elem(i), r2.elem(i));
  }
}



//! dest  = gaussian   under a subset
template<class T>
void gaussian(OSubLattice<T> dd)
{
  OLattice<T>& d = dd.field();
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
inline
void zero_rep(OLattice<T>& dest, const Subset& s) 
{
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    zero_rep(dest.elem(i));
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
  const int nodeSites = Layout::sitesOnNode();
  for(int i=0; i < nodeSites; ++i) 
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
 */
#if 1
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnSum>::Type_t
sum(const QDPExpr<RHS,OLattice<T> >& s1, const Subset& s)
{
  typename UnaryReturn<OLattice<T>, FnSum>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // Must initialize to zero since we do not know if the loop will be entered
  zero_rep(d.elem());

  ShiftPhase1 phase1;
  int maps_involved = forEach(s1, phase1 , BitOrCombine());

  if (maps_involved > 0) {

    const multi1d<int>& innerSites = MasterMap::Instance().getInnerSites(maps_involved);
    const multi1d<int>& faceSites = MasterMap::Instance().getFaceSites(maps_involved);

    for (int j=0 ; j<innerSites.size() ; j++ )
      if (s.isElement(innerSites[j])) 
	d.elem() += forEach(s1, EvalLeaf1(innerSites[j]), OpCombine());

    ShiftPhase2 phase2;
    forEach(s1, phase2 , NullCombine());

    for (int j=0 ; j<faceSites.size() ; j++ )
      if (s.isElement(faceSites[j]))
	d.elem() += forEach(s1, EvalLeaf1(faceSites[j]), OpCombine());

  } else {

    const int *tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); ++j) 
      {
	int i = tab[j];
	d.elem() += forEach(s1, EvalLeaf1(i), OpCombine());
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
#else
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnSum>::Type_t
sum(const QDPExpr<RHS,OLattice<T> >& s1, const Subset& s)
{
  typename UnaryReturn<OLattice<T>, FnSum>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // Must initialize to zero since we do not know if the loop will be entered
  zero_rep(d.elem());

  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    d.elem() += forEach(s1, EvalLeaf1(i), OpCombine());
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

//! OScalar = sum(OLattice)
/*!
 * Allow a global sum that sums over the lattice, but returns an object
 * of the same primitive type. E.g., contract only over lattice indices
 */
#if 1  // use overlap shift version ?
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnSum>::Type_t
sum(const QDPExpr<RHS,OLattice<T> >& s1)
{
  typename UnaryReturn<OLattice<T>, FnSum>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // Loop always entered - could unroll
  zero_rep(d.elem());
  const int nodeSites = Layout::sitesOnNode();

  ShiftPhase1 phase1;
  int maps_involved = forEach(s1, phase1 , BitOrCombine());

  if (maps_involved > 0) {

    const multi1d<int>& innerSites = MasterMap::Instance().getInnerSites(maps_involved);
    const multi1d<int>& faceSites = MasterMap::Instance().getFaceSites(maps_involved);

    for (int j=0 ; j<innerSites.size() ; j++ )
      d.elem() += forEach(s1, EvalLeaf1(innerSites[j]), OpCombine());

    ShiftPhase2 phase2;
    forEach(s1, phase2 , NullCombine());

    for (int j=0 ; j<faceSites.size() ; j++ )
      d.elem() += forEach(s1, EvalLeaf1(faceSites[j]), OpCombine());

  } else {

    for(int j=0; j < nodeSites; ++j) {
      d.elem() += forEach(s1, EvalLeaf1(j), OpCombine());
    }

  }

  QDPInternal::globalSum(d);

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}
#else
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

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // Loop always entered - could unroll
  zero_rep(d.elem());
  const int nodeSites = Layout::sitesOnNode();

  for(int i=0; i < nodeSites; ++i) 
    d.elem() += forEach(s1, EvalLeaf1(i), OpCombine());

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
typename UnaryReturn<OScalar<T>, FnSumMulti>::Type_t
sumMulti(const QDPExpr<RHS,OScalar<T> >& s1, const Set& ss)
{
  typename UnaryReturn<OScalar<T>, FnSumMulti>::Type_t  dest(ss.numSubsets());

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest[0], OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // lazy - evaluate repeatedly
  for(int i=0; i < ss.numSubsets(); ++i)
    evaluate(dest[i],OpAssign(),s1,all);


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
#if 1
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t
sumMulti(const QDPExpr<RHS,OLattice<T> >& s1, const Set& ss)
{
  typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t  dest(ss.numSubsets());

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest[0], OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // Initialize result with zero
  for(int k=0; k < ss.numSubsets(); ++k)
    zero_rep(dest[k]);

  // Loop over all sites and accumulate based on the coloring 
  const multi1d<int>& lat_color =  ss.latticeColoring();
  const int nodeSites = Layout::sitesOnNode();

  ShiftPhase1 phase1;
  int maps_involved = forEach(s1, phase1 , BitOrCombine());

  if (maps_involved > 0) {

    const multi1d<int>& innerSites = MasterMap::Instance().getInnerSites(maps_involved);
    const multi1d<int>& faceSites = MasterMap::Instance().getFaceSites(maps_involved);

    for (int j=0 ; j<innerSites.size() ; j++ ) {
      int i = lat_color[innerSites[j]];
      dest[i].elem() += forEach(s1, EvalLeaf1(innerSites[j]), OpCombine());
    }

    ShiftPhase2 phase2;
    forEach(s1, phase2 , NullCombine());

    for (int j=0 ; j<faceSites.size() ; j++ ) {
      int i = lat_color[faceSites[j]];
      dest[i].elem() += forEach(s1, EvalLeaf1(faceSites[j]), OpCombine());
    }

  } else {

    for(int i=0; i < nodeSites; ++i) {
      int j = lat_color[i];
      dest[j].elem() += forEach(s1, EvalLeaf1(i), OpCombine());
    }

  }



#if 0
  ShiftPhase1 phase1;
  forEach(s1, phase1 , NullCombine());

  for(int i=0; i < nodeSites; ++i) {
    if (phase1.isFace[i] == 0) {
      int j = lat_color[i];
      dest[j].elem() += forEach(s1, EvalLeaf1(i), OpCombine());
    }
  }

  ShiftPhase2 phase2;
  forEach(s1, phase2 , NullCombine());

  for(int i=0; i < nodeSites; ++i) {
    if (phase1.isFace[i] == 1) {
      int j = lat_color[i];
      dest[j].elem() += forEach(s1, EvalLeaf1(i), OpCombine());
    }
  }
#endif

  // Do a global sum on the result
  QDPInternal::globalSumArray(dest);

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return dest;
}
#else
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t
sumMulti(const QDPExpr<RHS,OLattice<T> >& s1, const Set& ss)
{
  typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t  dest(ss.numSubsets());

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest[0], OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // Initialize result with zero
  for(int k=0; k < ss.numSubsets(); ++k)
    zero_rep(dest[k]);

  // Loop over all sites and accumulate based on the coloring 
  const multi1d<int>& lat_color =  ss.latticeColoring();
  const int nodeSites = Layout::sitesOnNode();

  for(int i=0; i < nodeSites; ++i) 
  {
    int j = lat_color[i];
    dest[j].elem() += forEach(s1, EvalLeaf1(i), OpCombine());
  }

  // Do a global sum on the result
  QDPInternal::globalSumArray(dest);

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return dest;
}
#endif

//-----------------------------------------------------------------------------
// Multiple global sums on an array
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
  multi2d<typename UnaryReturn<OScalar<T>, FnSumMulti>::Type_t> dest(s1.size(), ss.numSubsets());

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
  multi2d<typename UnaryReturn<OLattice<T>, FnSum>::Type_t> dest(s1.size(), ss.numSubsets());

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest[0], OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // Initialize result with zero
  for(int i=0; i < dest.size1(); ++i)
    for(int j=0; j < dest.size2(); ++j)
      zero_rep(dest(j,i));

  // Loop over all sites and accumulate based on the coloring 
  const multi1d<int>& lat_color =  ss.latticeColoring();

  for(int k=0; k < s1.size(); ++k)
  {
    const OLattice<T>& ss1 = s1[k];
    const int nodeSites = Layout::sitesOnNode();
    for(int i=0; i < nodeSites; ++i) 
    {
      int j = lat_color[i];
      dest(k,j).elem() += ss1.elem(i);
    }
  }

  // Do a global sum on the result
  QDPInternal::globalSumArray(dest);

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

  const int *tab = s.siteTable().slice();
  for(int n=0; n < s1.size(); ++n)
  {
    const OLattice<T>& ss1 = s1[n];
    for(int j=0; j < s.numSiteTable(); ++j) 
    {
      int i = tab[j];
      d.elem() += localNorm2(ss1.elem(i));
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



//-----------------------------------------------------------------------------
//! OScalar = innerProduct(multi1d<source1>,multi1d<source2>))
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T1, class T2>
inline typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProduct>::Type_t
innerProduct(const multi1d< OScalar<T1> >& s1, const multi1d< OScalar<T2> >& s2)
{
  typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProduct>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnInnerProduct(), s1[0]);
  prof.time -= getClockTime();
#endif

  // Possibly loop entered
  zero_rep(d.elem());

  for(int n=0; n < s1.size(); ++n)
  {
    OScalar<T1>& ss1 = s1[n];
    OScalar<T2>& ss2 = s2[n];
    d.elem() += localInnerProduct(ss1.elem(),ss2.elem());
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
template<class T1, class T2>
inline typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProduct>::Type_t
innerProduct(const multi1d< OScalar<T1> >& s1, const multi1d< OScalar<T2> >& s2,
	     const Subset& s)
{
  return innerProduct(s1,s2);
}



//! OScalar = innerProduct(multi1d<OLattice>,multi1d<OLattice>) under an explicit subset
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T1, class T2>
inline typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProduct>::Type_t
innerProduct(const multi1d< OLattice<T1> >& s1, const multi1d< OLattice<T2> >& s2,
	     const Subset& s)
{
  typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProduct>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnInnerProduct(), s1[0]);
  prof.time -= getClockTime();
#endif

  // Possibly loop entered
  zero_rep(d.elem());

  const int *tab = s.siteTable().slice();
  for(int n=0; n < s1.size(); ++n)
  {
    const OLattice<T1>& ss1 = s1[n];
    const OLattice<T2>& ss2 = s2[n];
    for(int j=0; j < s.numSiteTable(); ++j) 
    {
      int i = tab[j];
      d.elem() += localInnerProduct(ss1.elem(i),ss2.elem(i));
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


//! OScalar = innerProduct(multi1d<OLattice>,multi1d<OLattice>)
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T1, class T2>
inline typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProduct>::Type_t
innerProduct(const multi1d< OLattice<T1> >& s1, const multi1d< OLattice<T2> >& s2)
{
  return innerProduct(s1,s2,all);
}



//-----------------------------------------------------------------------------
//! OScalar = innerProductReal(multi1d<source1>,multi1d<source2>))
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T1, class T2>
inline typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProductReal>::Type_t
innerProductReal(const multi1d< OScalar<T1> >& s1, const multi1d< OScalar<T2> >& s2)
{
  typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProductReal>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnInnerProductReal(), s1[0]);
  prof.time -= getClockTime();
#endif

  // Possibly loop entered
  zero_rep(d.elem());

  for(int n=0; n < s1.size(); ++n)
  {
    OScalar<T1>& ss1 = s1[n];
    OScalar<T2>& ss2 = s2[n];
    d.elem() += localInnerProductReal(ss1.elem(),ss2.elem());
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
template<class T1, class T2>
inline typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProductReal>::Type_t
innerProductReal(const multi1d< OScalar<T1> >& s1, const multi1d< OScalar<T2> >& s2,
		 const Subset& s)
{
  return innerProductReal(s1,s2);
}



//! OScalar = innerProductReal(multi1d<OLattice>,multi1d<OLattice>) under an explicit subset
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T1, class T2>
inline typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProductReal>::Type_t
innerProductReal(const multi1d< OLattice<T1> >& s1, const multi1d< OLattice<T2> >& s2,
		 const Subset& s)
{
  typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProductReal>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnInnerProductReal(), s1[0]);
  prof.time -= getClockTime();
#endif

  // Possibly loop entered
  zero_rep(d.elem());

  const int *tab = s.siteTable().slice();
  for(int n=0; n < s1.size(); ++n)
  {
    const OLattice<T1>& ss1 = s1[n];
    const OLattice<T2>& ss2 = s2[n];
    for(int j=0; j < s.numSiteTable(); ++j) 
    {
      int i = tab[j];
      d.elem() += localInnerProductReal(ss1.elem(i),ss2.elem(i));
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


//! OScalar = innerProductReal(multi1d<OLattice>,multi1d<OLattice>)
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T1, class T2>
inline typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProductReal>::Type_t
innerProductReal(const multi1d< OLattice<T1> >& s1, const multi1d< OLattice<T2> >& s2)
{
  return innerProductReal(s1,s2,all);
}




//-----------------------------------------------
// Global max and min
// NOTE: there are no subset version of these operations. It is very problematic
// and QMP does not support them.
//! OScalar = globalMax(OScalar)
/*!
 * Find the maximum an object has across the lattice
 */
template<class RHS, class T>
typename UnaryReturn<OScalar<T>, FnGlobalMax>::Type_t
globalMax(const QDPExpr<RHS,OScalar<T> >& s1)
{
  typename UnaryReturn<OScalar<T>, FnGlobalMax>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnGlobalMax(), s1);
  prof.time -= getClockTime();
#endif

  evaluate(d,OpAssign(),s1,all);   // since OScalar, no global max needed

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}


//! OScalar = globalMax(OLattice)
/*!
 * Find the maximum an object has across the lattice
 */
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t
globalMax(const QDPExpr<RHS,OLattice<T> >& s1)
{
  typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnGlobalMax(), s1);
  prof.time -= getClockTime();
#endif

  ShiftPhase1 phase1;
  int maps_involved = forEach(s1, phase1 , BitOrCombine());
  if (maps_involved > 0) {
    ShiftPhase2 phase2;
    forEach(s1, phase2 , NullCombine());
  }

  // Loop always entered so unroll
  d.elem() = forEach(s1, EvalLeaf1(0), OpCombine());   // SINGLE NODE VERSION FOR NOW

  const int vvol = Layout::sitesOnNode();
  for(int i=1; i < vvol; ++i) 
  {
    typename UnaryReturn<T, FnGlobalMax>::Type_t  dd = 
      forEach(s1, EvalLeaf1(i), OpCombine());   // SINGLE NODE VERSION FOR NOW

    if (toBool(dd > d.elem()))
      d.elem() = dd;
  }

  // Do a global max on the result
  QDPInternal::globalMax(d); 

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}


//! OScalar = globalMin(OScalar)
/*!
 * Find the minimum an object has across the lattice
 */
template<class RHS, class T>
typename UnaryReturn<OScalar<T>, FnGlobalMin>::Type_t
globalMin(const QDPExpr<RHS,OScalar<T> >& s1)
{
  typename UnaryReturn<OScalar<T>, FnGlobalMin>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnGlobalMin(), s1);
  prof.time -= getClockTime();
#endif

  evaluate(d,OpAssign(),s1,all);   // since OScalar, no global min needed

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}


//! OScalar = globalMin(OLattice)
/*!
 * Find the minimum an object has across the lattice
 */
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnGlobalMin>::Type_t
globalMin(const QDPExpr<RHS,OLattice<T> >& s1)
{
  typename UnaryReturn<OLattice<T>, FnGlobalMin>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnGlobalMin(), s1);
  prof.time -= getClockTime();
#endif

  ShiftPhase1 phase1;
  int maps_involved = forEach(s1, phase1 , BitOrCombine());
  if (maps_involved > 0) {
    ShiftPhase2 phase2;
    forEach(s1, phase2 , NullCombine());
  }

  // Loop always entered so unroll
  d.elem() = forEach(s1, EvalLeaf1(0), OpCombine());   // SINGLE NODE VERSION FOR NOW

  const int vvol = Layout::sitesOnNode();
  for(int i=1; i < vvol; ++i) 
  {
    typename UnaryReturn<T, FnGlobalMin>::Type_t  dd = 
      forEach(s1, EvalLeaf1(i), OpCombine());   // SINGLE NODE VERSION FOR NOW

    if (toBool(dd < d.elem()))
      d.elem() = dd;
  }

  // Do a global min on the result
  QDPInternal::globalMin(d); 

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
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
inline OScalar<T1>
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
inline OScalar<T1>
peekSite(const OLattice<T1>& l, const multi1d<int>& coord)
{
  OScalar<T1> dest;
  int nodenum = Layout::nodeNumber(coord);

  // Find the result somewhere within the machine.
  // Then we must get it to node zero so we can broadcast it
  // out to all nodes
  if (Layout::nodeNumber() == nodenum)
    dest.elem() = l.elem(Layout::linearSiteIndex(coord));
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
template<class T1>
inline OLattice<T1>&
pokeSite(OLattice<T1>& l, const OScalar<T1>& r, const multi1d<int>& coord)
{
  if (Layout::nodeNumber() == Layout::nodeNumber(coord))
    l.elem(Layout::linearSiteIndex(coord)) = r.elem();

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
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    dest[i].elem() = src.elem(i);
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
QDP_insert(OLattice<T>& dest, const multi1d<OScalar<T> >& src, const Subset& s)
{
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    dest.elem(i) = src[i].elem();
  }
}



//-----------------------------------------------------------------------------
// Map
//

// Empty map
struct FnMap;


#if defined(QDP_USE_PROFILING)   
template <>
struct TagVisitor<FnMap, PrintTag> : public ParenPrinter<FnMap>
{ 
  static void visit(FnMap op, PrintTag t) 
    { t.os_m << "shift"; }
};
#endif


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


  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPType<T1,C1> & l)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(*this),
	CreateLeaf<QDPType<T1,C1> >::make(l)));
    }


  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPExpr<T1,C1> & l)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(*this),
	CreateLeaf<QDPExpr<T1,C1> >::make(l)));
    }


public:
  //! Accessor to offsets
  const multi1d<int>& goffset() const {return goffsets;}
  const multi1d<int>& soffset() const {return soffsets;}
  const multi1d<int>& roffset() const {return roffsets;}
  const multi1d<int>& get_ind_array() const {return ind_array;}
  int getId() const {return myId;}

private:
  //! Hide copy constructor
  Map(const Map&) {}

  //! Hide operator=
  void operator=(const Map&) {}

private:
  friend class FnMap;
  friend class FnMapRsrc;
  template<class E,class F,class C> friend class ForEach;

  //! Offset table used for communications. 
  /*! 
   * The direction is in the sense of the Map or Shift functions from QDP.
   * goffsets(position) 
   */ 
  multi1d<int> goffsets;
  multi1d<int> soffsets;
  multi1d<int> srcnode;
  multi1d<int> dstnode;

  multi1d<int> roffsets;
  multi1d<int> ind_array;
  int myId;

  multi1d<int> srcenodes;
  multi1d<int> destnodes;

  multi1d<int> srcenodes_num;
  multi1d<int> destnodes_num;

  // Indicate off-node communications is needed;
  bool offnodeP;
};





struct FnMap
{
  //PETE_EMPTY_CONSTRUCTORS(FnMap)

  const Map& map;
  std::shared_ptr<RsrcWrapper> pRsrc;

  FnMap(const Map& m): map(m), pRsrc(new RsrcWrapper( m.destnodes , m.srcenodes )) {}
  FnMap(const FnMap& f) : map(f.map) , pRsrc(f.pRsrc) {}

  FnMap& operator=(const FnMap& f) = delete;

  const FnMapRsrc& getResource(int srcnum_, int dstnum_) {
    return (*pRsrc).getResource( srcnum_ , dstnum_ );
  }

  const FnMapRsrc& getCached() const {
    return (*pRsrc).get();
  }
  
  template<class T>
  inline typename UnaryReturn<T, FnMap>::Type_t
  operator()(const T &a) const
  {
    return (a);
  }

};









template<class A>
struct ForEach<UnaryNode<FnMap, A>, ShiftPhase1 , BitOrCombine>
{
  typedef typename ForEach<A, EvalLeaf1, OpCombine>::Type_t InnerTypeA_t;
  typedef typename Combine1<InnerTypeA_t, FnMap, OpCombine>::Type_t InnerType_t;
  typedef int Type_t;
  typedef QDPExpr<A,OLattice<InnerType_t> > Expr;
  inline static
  Type_t apply(const UnaryNode<FnMap, A> &expr, const ShiftPhase1 &f, const BitOrCombine &c)
  {
    const Map& map = expr.operation().map;
    FnMap& fnmap = const_cast<FnMap&>(expr.operation());

    const int nodeSites = Layout::sitesOnNode();
    int returnVal=0;

    if (map.offnodeP)
      {
#if QDP_DEBUG >= 3
	QDP_info("Map: off-node communications required");
#endif

	int dstnum = map.destnodes_num[0]*sizeof(InnerType_t);
	int srcnum = map.srcenodes_num[0]*sizeof(InnerType_t);

	const FnMapRsrc& rRSrc = fnmap.getResource(srcnum,dstnum);

	const InnerType_t *send_buf_c = rRSrc.getSendBufPtr<InnerType_t>();

	InnerType_t* send_buf = const_cast<InnerType_t*>(send_buf_c);

	if ( send_buf == 0x0 ) { QDP_error_exit("QMP_get_memory_pointer returned NULL pointer from non NULL QMP_mem_t (send_buf)\n"); }

	const int my_node = Layout::nodeNumber();

#if 1
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
	//printf("soffsets.size=%d",map.soffsets.size());
#if 0
	user_arg_face<InnerType_t,InnerType_t,OpAssign,Expr> a0(send_buf, map.soffsets.slice() , subexpr, OpAssign()  );
	dispatch_to_threads< user_arg_face<InnerType_t,InnerType_t,OpAssign,Expr> >(map.soffsets.size(), a0, evaluate_userfunc_face );
#else
	for(int si=0; si < map.soffsets.size(); ++si)
  	  send_buf[si] = forEach( subexpr , EvalLeaf1(map.soffsets[si]) , OpCombine() );
#endif
#endif

	rRSrc.send_receive();
	
	returnVal = maps_involved | map.getId();
      } else {

      returnVal = ForEach<A, ShiftPhase1, BitOrCombine>::apply(expr.child(), f, c);
    }

    return returnVal;
  }
};






template<class A, class CTag>
struct ForEach<UnaryNode<FnMap, A>, ShiftPhase2 , CTag>
{
  typedef typename ForEach<A, EvalLeaf1, OpCombine>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnMap, OpCombine>::Type_t Type_t;
  typedef QDPExpr<A,OLattice<Type_t> > Expr;
  inline static
  Type_t apply(const UnaryNode<FnMap, A> &expr, const ShiftPhase2 &f, const CTag &c)
  {
    const Map& map = expr.operation().map;
    FnMap& fnmap = const_cast<FnMap&>(expr.operation());
#if 1
    if (map.offnodeP) {
      // can be optimized
      const FnMapRsrc& rRSrc = fnmap.getCached();
      rRSrc.qmp_wait();

    }
#endif
  }
};



template<class A, class CTag>
struct ForEach<UnaryNode<FnMap, A>, EvalLeaf1, CTag>
{
  typedef typename ForEach<A, EvalLeaf1, CTag>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnMap, CTag>::Type_t Type_t;
  inline static
  Type_t apply(const UnaryNode<FnMap, A> &expr, const EvalLeaf1 &f, const CTag &c)
  {
    const Map& map = expr.operation().map;
    FnMap& fnmap = const_cast<FnMap&>(expr.operation());
    
    if (map.offnodeP) {
#if QDP_DEBUG >= 3
      QDP_info("ind_array[%d]=%d %d",f.val1(),map.get_ind_array()[f.val1()],map.goffsets[f.val1()]);
#endif
      if (map.get_ind_array()[f.val1()] < 0) {
	EvalLeaf1 ff( -map.get_ind_array()[f.val1()] - 1 );
	return Combine1<TypeA_t, FnMap, CTag>::combine(ForEach<A, EvalLeaf1, CTag>::apply(expr.child(), ff, c),expr.operation(), c);
      } else {
	const FnMapRsrc& rRSrc = fnmap.getCached();
	const Type_t *recv_buf_c = rRSrc.getRecvBufPtr<Type_t>();
	Type_t* recv_buf = const_cast<Type_t*>(recv_buf_c);
	if ( recv_buf == 0x0 ) { QDP_error_exit("QMP_get_memory_pointer returned NULL pointer from non NULL QMP_mem_t (recv_buf). Do you use shifts of shifts?"); }
	return recv_buf[map.get_ind_array()[f.val1()]];
      }
    } else {
      EvalLeaf1 ff( map.goffsets[f.val1()]);
      return Combine1<TypeA_t, FnMap, CTag>::combine(ForEach<A, EvalLeaf1, CTag>::apply(expr.child(), ff, c),expr.operation(), c);
    }
  }
};


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
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(mapsa[dir]),
	CreateLeaf<QDPType<T1,C1> >::make(l)));
    }


  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPExpr<T1,C1> & l, int dir)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(mapsa[dir]),
	CreateLeaf<QDPExpr<T1,C1> >::make(l)));
    }


#if 0
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
#endif

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
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(bimaps[(isign+1)>>1]),
	CreateLeaf<QDPType<T1,C1> >::make(l)));
    }


  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPExpr<T1,C1> & l, int isign)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(bimaps[(isign+1)>>1]),
	CreateLeaf<QDPExpr<T1,C1> >::make(l)));
    }



#if 0
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
#endif


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
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(bimapsa((isign+1)>>1,dir)),
	CreateLeaf<QDPType<T1,C1> >::make(l)));
    }


  template<class T1,class C1>
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPExpr<T1,C1> & l, int isign, int dir)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(bimapsa((isign+1)>>1,dir)),
	CreateLeaf<QDPExpr<T1,C1> >::make(l)));
    }



#if 0
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
#endif

private:
  //! Hide copy constructor
  ArrayBiDirectionalMap(const ArrayBiDirectionalMap&) {}

  //! Hide operator=
  void operator=(const ArrayBiDirectionalMap&) {}

private:
  multi2d<Map> bimapsa;
  
};


//-----------------------------------------------------------------------------

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
template<class T>  
XMLWriter& operator<<(XMLWriter& xml, const OLattice<T>& d)
{
  T recv_buf;

  xml.openTag("OLattice");
  XMLWriterAPI::AttributeList alist;

  // Find the location of each site and send to primary node
  for(int site=0; site < Layout::vol(); ++site)
  {
    multi1d<int> coord = crtesn(site, Layout::lattSize());

    int node   = Layout::nodeNumber(coord);
    int linear = Layout::linearSiteIndex(coord);

    // Copy to buffer: be really careful since max(linear) could vary among nodes
    if (Layout::nodeNumber() == node)
      recv_buf = d.elem(linear);

    // Send result to primary node. Avoid sending prim-node sending to itself
    if (node != 0)
    {
#if 1
      // All nodes participate
      QDPInternal::route((void *)&recv_buf, node, 0, sizeof(T));
#else
      if (Layout::primaryNode())
	QDPInternal::recvFromWait((void *)&recv_buf, node, sizeof(T));

      if (Layout::nodeNumber() == node)
	QDPInternal::sendToWait((void *)&recv_buf, 0, sizeof(T));
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


//! Write a lattice quantity
/*! This code assumes no inner grid */
void writeOLattice(BinaryWriter& bin, 
		   const char* output, size_t size, size_t nmemb);

//! Binary output
/*! Assumes no inner grid */
template<class T>
void write(BinaryWriter& bin, const OLattice<T>& d)
{
  writeOLattice(bin, (const char *)&(d.elem(0)), 
		sizeof(typename WordType<T>::Type_t), 
		sizeof(T) / sizeof(typename WordType<T>::Type_t));
}

//! Write a single site of a lattice quantity
/*! This code assumes no inner grid */
void writeOLattice(BinaryWriter& bin, 
		   const char* output, size_t size, size_t nmemb,
		   const multi1d<int>& coord);

//! Write a single site of a lattice quantity
/*! Assumes no inner grid */
template<class T>
void write(BinaryWriter& bin, const OLattice<T>& d, const multi1d<int>& coord)
{
  writeOLattice(bin, (const char *)&(d.elem(0)), 
		sizeof(typename WordType<T>::Type_t), 
		sizeof(T) / sizeof(typename WordType<T>::Type_t),
		coord);
}

//! Write a single site of a lattice quantity
/*! This code assumes no inner grid */
void writeOLattice(BinaryWriter& bin, 
		   const char* output, size_t size, size_t nmemb,
		   const Subset& sub);

//! Write a single site of a lattice quantity
/*! Assumes no inner grid */
template<class T>
void write(BinaryWriter& bin, OSubLattice<T> dd)
{
  const OLattice<T>& d = dd.field();

  writeOLattice(bin, (const char *)&(d.elem(0)), 
		sizeof(typename WordType<T>::Type_t), 
		sizeof(T) / sizeof(typename WordType<T>::Type_t),
		dd.subset());
}


//! Read a lattice quantity
/*! This code assumes no inner grid */
void readOLattice(BinaryReader& bin, 
		  char* input, size_t size, size_t nmemb);

//! Binary input
/*! Assumes no inner grid */
template<class T>
void read(BinaryReader& bin, OLattice<T>& d)
{
  readOLattice(bin, (char *)&(d.elem(0)), 
	       sizeof(typename WordType<T>::Type_t), 
	       sizeof(T) / sizeof(typename WordType<T>::Type_t));
}

//! Read a single site of a lattice quantity
/*! This code assumes no inner grid */
void readOLattice(BinaryReader& bin, 
		  char* input, size_t size, size_t nmemb, 
		  const multi1d<int>& coord);

//! Read a single site of a lattice quantity
/*! Assumes no inner grid */
template<class T>
void read(BinaryReader& bin, OLattice<T>& d, const multi1d<int>& coord)
{
  readOLattice(bin, (char *)&(d.elem(0)), 
	       sizeof(typename WordType<T>::Type_t), 
	       sizeof(T) / sizeof(typename WordType<T>::Type_t),
	       coord);
}

//! Read a single site of a lattice quantity
/*! This code assumes no inner grid */
void readOLattice(BinaryReader& bin, 
		  char* input, size_t size, size_t nmemb, 
		  const Subset& sub);

//! Read a single site of a lattice quantity
/*! Assumes no inner grid */
template<class T>
void read(BinaryReader& bin, OSubLattice<T> d)
{
  readOLattice(bin, (char *)(d.field().getF()),
	       sizeof(typename WordType<T>::Type_t), 
	       sizeof(T) / sizeof(typename WordType<T>::Type_t),
	       d.subset());
}



// **************************************************************
// Special support for slices of a lattice
namespace LatticeTimeSliceIO 
{
  //! Lattice time slice reader
  void readOLatticeSlice(BinaryReader& bin, char* data, 
			 size_t size, size_t nmemb,
			 int start_lexico, int stop_lexico);

  void writeOLatticeSlice(BinaryWriter& bin, const char* data, 
			  size_t size, size_t nmemb,
			  int start_lexico, int stop_lexico);


  // Read a time slice of a lattice quantity (time must be most slowly varying)
  template<class T>
  void readSlice(BinaryReader& bin, OLattice<T>& data, 
		 int start_lexico, int stop_lexico)
  {
    readOLatticeSlice(bin, (char *)&(data.elem(0)), 
		      sizeof(typename WordType<T>::Type_t), 
		      sizeof(T) / sizeof(typename WordType<T>::Type_t),
		      start_lexico, stop_lexico);
  }


  // Write a time slice of a lattice quantity (time must be most slowly varying)
  template<class T>
  void writeSlice(BinaryWriter& bin, const OLattice<T>& data, 
		  int start_lexico, int stop_lexico)
  {
    writeOLatticeSlice(bin, (const char *)&(data.elem(0)), 
		       sizeof(typename WordType<T>::Type_t), 
		       sizeof(T) / sizeof(typename WordType<T>::Type_t),
		       start_lexico, stop_lexico);
  }

} // namespace LatticeTimeSliceIO

} // namespace QDP
#endif
