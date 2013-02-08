// -*- C++ -*-

/*! @file
 * @brief Outer lattice routines specific to a parallel platform with scalar layout
 */

#ifndef QDP_PARSCALAR_SPECIFIC_JIT_H
#define QDP_PARSCALAR_SPECIFIC_JIT_H

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

  ///////////////////////////////////
#if 0
  //! Global sum on a multi1d
  template<class T>
  inline void globalSumArray(multi1d<T>& dest)
  {
    // The implementation here is relying on the structure being packed
    // tightly in memory - no padding
    typedef typename WordType<T>::Type_t  W;   // find the machine word type

#if 1
    QDPIO::cout << "sizeof(T) = " << sizeof(T) << endl;
    QDPIO::cout << "sizeof(W) = " << sizeof(W) << endl;
    QDPIO::cout << "Calling multi1d global sum array with length " << dest.size()*sizeof(T)/sizeof(W) << endl;
#endif
    globalSumArray((W *)dest.slice(), dest.size()*sizeof(T)/sizeof(W)); // call appropriate hook
  }
#endif
  //////////////////////////////////

  //! Global sum on a multi1d
  template<class T>
  inline void globalSumArray(multi1d<T>& dest)
  {
    // The implementation here is relying on the structure being packed
    // tightly in memory - no padding
    typedef typename WordType<T>::Type_t  W;   // find the machine word type
    typedef typename T::SubType_t         P;   // Primitive type

#if 0
    QDPIO::cout << "sizeof(P) = " << sizeof(P) << endl;
    QDPIO::cout << "sizeof(W) = " << sizeof(W) << endl;
    QDPIO::cout << "Calling multi1d global sum array with length " << sizeof(P)/sizeof(W) << endl;
#endif

    for (int i = 0 ; i < dest.size() ; ++i )
      globalSumArray( (W *)dest[i].getF(), sizeof(P)/sizeof(W)); // call appropriate hook

    //globalSumArray((W *)dest.slice(), dest.size()*sizeof(T)/sizeof(W)); // call appropriate hook
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

  template<class T>
  inline void globalSum(OScalar<T>& dest)
  {
    // The implementation here is relying on the structure being packed
    // tightly in memory - no padding
    typedef typename WordType<T>::Type_t  W;   // find the machine word type

#if 0 
    QDPIO::cout << "sizeof(T) = " << sizeof(T) << endl;
    QDPIO::cout << "sizeof(W) = " << sizeof(W) << endl;
    QDPIO::cout << "Calling global sum array with length " << sizeof(T)/sizeof(W) << endl;
#endif
    if (QMP_get_number_of_nodes() > 1) {
      globalSumArray((W *)dest.getF(), int(sizeof(T)/sizeof(W))); // call appropriate hook
    } else {
      QDP_debug("global sum: no MPI reduction");
    }
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

    if (QMP_get_number_of_nodes() > 1) {
      globalMaxValue((W *)dest.getF());
    } else {
      QDP_debug("global max: no MPI reduction");
    }
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

    globalMinValue((W *)dest.getF());
  }


  //! Broadcast from primary node to all other nodes
  template<class T>
  inline void broadcast(T& dest)
  {
    QMP_broadcast((void *)&dest, sizeof(T));
  }

  //! Broadcast from primary node to all other nodes
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

//! include the header file for dispatch
#include "qdp_dispatch.h"


//
// Unless the overhead of kernel calls reduces significantly
// eval(Sca,Sca) remains on host
//




template<class T, class T1, class Op, class RHS>
void evaluate(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs, const Subset& s)
{
  // Don't pass pretty function to QDP_info, buffer length!!
#ifdef GPU_DEBUG_DEEP
  cout << __PRETTY_FUNCTION__ << endl;
#endif

  while (1) {

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest, op, rhs);
  prof.time -= getClockTime();
#endif

    static QDPJit::SharedLibEntry sharedLibEntry;
    static MapVolumes*              mapVolumes;
    static string                   strId;
    static string                   prg;

    QDPJitArgs cudaArgs;

    QDP_debug("eval(Lat,Sca) dev!");

    int argNum = cudaArgs.addInt( s.numSiteTable() );
    int argOrd = cudaArgs.addBool( s.hasOrderedRep() );
    int argStart = cudaArgs.addInt( s.start() );
    int argSubset = cudaArgs.addPtr( QDPCache::Instance().getDevicePtr( s.getId() ) );

    if (!mapVolumes) {

      string codeRHS,codeDest,codeOp;

      if (!getCodeString( codeRHS , rhs , "idx", cudaArgs )) { QDP_info("eval: could not cache RHS");  break;   }      
      if (!getCodeString( codeDest , dest , "idx", cudaArgs )) { QDP_info("eval: could not cache dest"); break; }
      if (!getCodeStringOp( codeOp , op , "idx", cudaArgs )) { QDP_info("eval: could not cache op");  break;   }      

      ostringstream osId;
      osId << "evaluateOS " << codeDest << codeOp << codeRHS;
      strId = osId.str();
      xmlready(strId);

#ifdef GPU_DEBUG_DEEP
      cout << "strId = " << strId << endl;
#endif
      
      std::ostringstream sprg;
      sprg << "  if (" << cudaArgs.getCode(argOrd) << ") {" << endl;
      sprg << "    int idx = " << cudaArgs.getCode(argStart) << ";" << endl;
      sprg << "    idx += blockDim.x * blockIdx.x + blockDim.x * gridDim.x * blockIdx.y + threadIdx.x;" << endl;
      sprg << "    if (idx < " << cudaArgs.getCode(argNum) << "+" <<  cudaArgs.getCode(argStart) << ") {\n";
      sprg << "      " << codeOp << "(" << codeDest << "," << codeRHS << ");\n";
      sprg << "    }" << endl;
      sprg << "  } else {" << endl;
      sprg << "    int idx0 = blockDim.x * blockIdx.x + blockDim.x * gridDim.x * blockIdx.y + threadIdx.x; \n";
      sprg << "    if (idx0 < " << cudaArgs.getCode(argNum) << ") {\n";
      sprg << "      int idx  = ((int*)" << cudaArgs.getCode(argSubset) << ")[idx0];" << endl;
      sprg << "      " << codeOp << "(" << codeDest << "," << codeRHS << ");\n";
      sprg << "    }" << endl;
      sprg << "  }" << endl;
      prg = sprg.str();

#ifdef GPU_DEBUG_DEEP
      cout << "Cuda kernel code = " << endl << prg << endl << endl;
#endif
    } else {
      if (!cacheLock(  rhs , cudaArgs )) { QDP_info("eval: could not cache RHS");  break;   }
      if (!cacheLock(  dest ,  cudaArgs )) { QDP_info("eval: could not cache dest"); break; }
      if (!cacheLockOp(  op ,  cudaArgs )) { QDP_info("eval: could not cache op");  break;   }
    }

    if (!QDPJit::Instance()( strId , prg , cudaArgs.getDevPtr() , s.numSiteTable() , sharedLibEntry , mapVolumes )) {
      QDP_info("eval(Lat,Lat) call to cuda jitter failed");
      break;
    }

#if defined(QDP_USE_PROFILING)   
    prof.time += getClockTime();
    prof.count++;
    prof.print();
#endif    

    return;
  }

  QDP_debug("eval(Lat,Sca) host!");

  int numSiteTable = s.numSiteTable();
  u_arg<T,T1,Op,RHS> a(dest, rhs, op, s.siteTable().slice());
  dispatch_to_threads< u_arg<T,T1,Op,RHS> >(numSiteTable, a, ev_userfunc);

}
 


struct ShiftPhase1
{
};

struct ShiftPhase2
{
};


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




template<class T, class T1, class Op, class RHS>
void evaluate(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs, const Subset& s)
{
  // Don't pass pretty function to QDP_info, buffer length!!
#ifdef GPU_DEBUG_DEEP
  cout << __PRETTY_FUNCTION__ << endl;
#endif
  
#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest, op, rhs);
  prof.time -= getClockTime();
#endif

  ShiftPhase1 phase1;
  int maps_involved = forEach(rhs, phase1 , BitOrCombine());

#ifdef GPU_DEBUG_DEEP
  QDP_info("maps_involved=%d",maps_involved);
#endif

  cout << "0\n";

  if (maps_involved > 0) {
    cout << "0a\n";

    int innerId = MasterMap::Instance().getIdInner(maps_involved);
    int innerCount = MasterMap::Instance().getCountInner(maps_involved);
    int faceId = MasterMap::Instance().getIdFace(maps_involved);
    int faceCount = MasterMap::Instance().getCountFace(maps_involved);

#ifdef GPU_DEBUG_DEEP
    QDP_info("innerId=%d count=%d  faceId=%d count=%d",innerId,innerCount,faceId,faceCount);
#endif

    StopWatch watchov;
    watchov.start();


    static QDPJit::SharedLibEntry sharedLibEntry;
    static MapVolumes*              mapVolumes;
    static string                   strId;
    static string                   prg;

    {
      QDPJitArgs cudaArgs;

      int argNum = cudaArgs.addInt( innerCount );
      int argInner = cudaArgs.addPtr( QDPCache::Instance().getDevicePtr( innerId ) );
      int argMember = cudaArgs.addPtr( QDPCache::Instance().getDevicePtr( s.getIdMemberTable() ) );

      if (!mapVolumes) {
	string codeRHS,codeDest,codeOp;
	if (!getCodeString( codeRHS , rhs , "idx_inner", cudaArgs )) { QDP_error_exit("eval: could not cache RHS");     }      
	if (!getCodeString( codeDest , dest , "idx_inner", cudaArgs )) { QDP_error_exit("eval: could not cache dest");  }
	if (!getCodeStringOp( codeOp , op , "idx_inner", cudaArgs )) { QDP_error_exit("eval: could not cache op");     }      

	ostringstream osId;
	osId << "eval(lat,lat) inner-member " << codeDest << codeOp << codeRHS;
	strId = osId.str();
	xmlready(strId);
#ifdef GPU_DEBUG_DEEP
	cout << "strId = " << strId << endl;
#endif
      
	std::ostringstream sprg;
	sprg << "    int idx = blockDim.x * blockIdx.x + blockDim.x * gridDim.x * blockIdx.y + threadIdx.x;" << endl;
	sprg << "    if (idx < " << cudaArgs.getCode(argNum) << ") {\n";
	sprg << "      int idx_inner = (((int*)(" << cudaArgs.getCode(argInner) << "))[idx]);\n";
	sprg << "      if (((bool*)(" << cudaArgs.getCode(argMember) << "))[idx_inner]) {\n";
	sprg << "        " << codeOp << "(" << codeDest << "," << codeRHS << ");\n";
	sprg << "      }" << endl;
	sprg << "    }" << endl;
	prg = sprg.str();

#ifdef GPU_DEBUG_DEEP
	cout << "Cuda kernel code = " << endl << prg << endl << endl;
#endif
      } else {
	if (!cacheLock(  rhs ,  cudaArgs )) { QDP_error_exit("eval: could not cache RHS");     }      
	if (!cacheLock(  dest , cudaArgs )) { QDP_error_exit("eval: could not cache dest");  }
	if (!cacheLockOp(  op , cudaArgs )) { QDP_error_exit("eval: could not cache op");     }      
      }

#ifdef GPU_DEBUG
      QDP_debug("eval(Lat,Lat) inner dev!");
#endif

      if (!QDPJit::Instance()( strId , prg , cudaArgs.getDevPtr() , innerCount , sharedLibEntry  , mapVolumes )) 
	QDP_error_exit("eval(Lat,Lat) inner call to cuda jitter failed");
    }

    ShiftPhase2 phase2;
    forEach(rhs, phase2 , NullCombine());

    {
      QDPJitArgs cudaArgs;

      int argNum = cudaArgs.addInt( faceCount );
      int argFace = cudaArgs.addPtr( QDPCache::Instance().getDevicePtr( faceId ) );
      int argMember = cudaArgs.addPtr( QDPCache::Instance().getDevicePtr( s.getIdMemberTable() ) );

      if (!mapVolumes) 
	QDP_error_exit("internal error: map volumes gone!");

      if (!cacheLock(  rhs ,  cudaArgs )) { QDP_error_exit("eval: could not cache RHS");     }      
      if (!cacheLock(  dest , cudaArgs )) { QDP_error_exit("eval: could not cache dest");  }
      if (!cacheLockOp(  op , cudaArgs )) { QDP_error_exit("eval: could not cache op");     }

      QDP_debug("eval(Lat,Lat) face dev!");

      string prg;
      if (!QDPJit::Instance()( strId , prg , cudaArgs.getDevPtr() , faceCount , sharedLibEntry  , mapVolumes )) 
	QDP_error_exit("eval(Lat,Lat) face call to cuda jitter failed");
    }

  } else {

    cout << "0b\n";

    static QDPJit::SharedLibEntry sharedLibEntry;
    static MapVolumes*              mapVolumes;
    static string                   strId;
    static string                   prg;
    static std::vector<QDPJitArgs>  vecCudaArgs;

    vecCudaArgs.reserve( DeviceParams::Instance().getMaxJitValueSet() );
    cout << "0c\n";
    assert( vecCudaArgs.capacity() > vecCudaArgs.size() );
    cout << "0d " << vecCudaArgs.size()+1 << "\n";
    vecCudaArgs.resize( vecCudaArgs.size()+1 );
    cout << "0e " << "\n";
    
    QDPJitArgs& cudaArgs = vecCudaArgs.back();

    int argNum = cudaArgs.addInt( s.numSiteTable() );
    int argOrd = cudaArgs.addBool( s.hasOrderedRep() );
    int argStart = cudaArgs.addInt( s.start() );
    int argSubset = cudaArgs.addPtr( QDPCache::Instance().getDevicePtr( s.getId() ) );

    if (!mapVolumes) {
      string codeRHS,codeDest,codeOp;
      if (!getCodeString( codeRHS , rhs , "idx", cudaArgs )) { QDP_error_exit("eval: could not cache RHS");     }      
      if (!getCodeString( codeDest , dest , "idx", cudaArgs )) { QDP_error_exit("eval: could not cache dest");  }
      if (!getCodeStringOp( codeOp , op , "idx", cudaArgs )) { QDP_error_exit("eval: could not cache op");     }      

      ostringstream osId;
      osId << "eval(lat,lat) " << codeDest << codeOp << codeRHS;
      strId = osId.str();
      xmlready(strId);
#ifdef GPU_DEBUG_DEEP
      cout << "strId = " << strId << endl;
#endif
      
      std::ostringstream sprg;
      sprg << "  if (" << cudaArgs.getCode(argOrd) << ") {" << endl;
      sprg << "    int idx = " << cudaArgs.getCode(argStart) << ";" << endl;
      sprg << "    idx += blockDim.x * blockIdx.x + blockDim.x * gridDim.x * blockIdx.y + threadIdx.x;" << endl;
      sprg << "    if (idx < " << cudaArgs.getCode(argNum) << "+" <<  cudaArgs.getCode(argStart) << ") {\n";
      sprg << "      " << codeOp << "(" << codeDest << "," << codeRHS << ");\n";
      sprg << "    }" << endl;
      sprg << "  } else {" << endl;
      sprg << "    int idx0 = blockDim.x * blockIdx.x + blockDim.x * gridDim.x * blockIdx.y + threadIdx.x; \n";
      sprg << "    if (idx0 < " << cudaArgs.getCode(argNum) << ") {\n";
      sprg << "      int idx  = ((int*)" << cudaArgs.getCode(argSubset) << ")[idx0];" << endl;
      sprg << "      " << codeOp << "(" << codeDest << "," << codeRHS << ");\n";
      sprg << "    }" << endl;
      sprg << "  }" << endl;
      prg = sprg.str();

#ifdef GPU_DEBUG_DEEP
      cout << "Cuda kernel code = " << endl << prg << endl << endl;
#endif
    } else {
      if (!cacheLock(  rhs ,  cudaArgs )) { QDP_error_exit("eval: could not cache RHS");     }      
      if (!cacheLock(  dest , cudaArgs )) { QDP_error_exit("eval: could not cache dest");  }
      if (!cacheLockOp(  op , cudaArgs )) { QDP_error_exit("eval: could not cache op");     }      
    }

    QDP_debug("eval(Lat,Lat) !");

    StopWatch watch0;
    watch0.start();

    cout << "1\n";

    // Probablity greater zero that we already have this value set ?
    std::vector<QDPJitArgs>::iterator iterCudaArg = vecCudaArgs.begin();
    cout << "2\n";
    if (vecCudaArgs.size() > 1) {
      cout << "3\n";
      if ((iterCudaArg = std::find( vecCudaArgs.begin() , --vecCudaArgs.end() , vecCudaArgs.back() )) != --vecCudaArgs.end()) {
	// We already have these runtime values of cudaArgs
	cout << "4\n";
	vecCudaArgs.pop_back();
      } else {
	cout << "5\n";
	iterCudaArg = --vecCudaArgs.end();
      }
    }
    cout << "6\n";
    

    if (!QDPJit::Instance()( strId , prg , (*iterCudaArg).getDevPtr() , s.numSiteTable() , sharedLibEntry  , mapVolumes )) 
      QDP_error_exit("eval(Lat,Lat) call to cuda jitter failed");
  }


#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  
  // QDP_debug("eval(Lat,Lat) host!");
  
  // int numSiteTable = s.numSiteTable();
  // user_arg<T,T1,Op,RHS> a(dest, rhs, op, s.siteTable().slice());
  // dispatch_to_threads<user_arg<T,T1,Op,RHS> >(numSiteTable, a, evaluate_userfunc);

}

    ///////////////////
    // Original code
    //////////////////
    // const int *tab = s.siteTable().slice();
    // for(int j=0; j < s.numSiteTable(); ++j) 
    //   {
    // 	int i = tab[j];
    // 	op(dest.elem(i), forEach(rhs, EvalLeaf1(i), OpCombine()));
    //   }





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



template<class T1, class T2> 
void 
copymask(OLattice<T2>& dest, const OLattice<T1>& mask, const OLattice<T2>& s1) 
{
  // Don't pass pretty function to QDP_info, buffer length!!
#ifdef GPU_DEBUG_DEEP
  cout << __PRETTY_FUNCTION__ << endl;
#endif

  while (1) {
    QDPJitArgs cudaArgs;

    static QDPJit::SharedLibEntry sharedLibEntry;
    static MapVolumes*              mapVolumes;
    static string                   strId;
    static string                   prg;

    int argNum = cudaArgs.addInt( Layout::sitesOnNode() );

    if (!mapVolumes) {

      string codeDest,codeMask,codeS1;

      if (!getCodeString( codeDest , dest , "idx", cudaArgs )) { QDP_info("copymask: could not cache dest"); break;  }      
      if (!getCodeString( codeMask , mask , "idx", cudaArgs )) { QDP_info("copymask: could not cache mask"); break;  }      
      if (!getCodeString( codeS1 , s1 , "idx", cudaArgs )) { QDP_info("copymask: could not cache s1"); break; }      

      ostringstream osId;
      osId << "copymask " << codeDest << codeMask << codeS1;
      strId = osId.str();
      xmlready(strId);

#ifdef GPU_DEBUG_DEEP
      cout << "strId = " << strId << endl;
#endif

      std::ostringstream sprg;
      sprg << "  int idx = blockDim.x * gridDim.x * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;\n";
      sprg << "  if (idx < " << cudaArgs.getCode(argNum) << ")" << endl;
      sprg << "    copymask( " << codeDest << " , " << codeMask << " , " << codeS1 << " );\n";
      prg = sprg.str();
    } else {
      if (!cacheLock(  dest ,  cudaArgs )) { QDP_info("copymask: could not cache dest"); break;  }      
      if (!cacheLock(  mask ,  cudaArgs )) { QDP_info("copymask: could not cache mask"); break;  }      
      if (!cacheLock(  s1 , cudaArgs )) { QDP_info("copymask: could not cache s1"); break; }      
    }

    QDP_debug("copymask(Lat,Lat,Lat) dev!");


#ifdef GPU_DEBUG_DEEP
    cout << "Cuda kernel code = " << endl << prg << endl << endl;
#endif


    if (!QDPJit::Instance()( strId , prg , cudaArgs.getDevPtr() , Layout::sitesOnNode() , sharedLibEntry , mapVolumes )) {
      QDP_info("copymask(Lat,Lat,Lat) call to cuda jitter failed");
      break;
    }

    return;
  }

  QDP_debug("copymask(Lat,Lat,Lat) host!");

  const int vvol = Layout::sitesOnNode();
  for(int i=0; i < vvol; ++i) 
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





template<class T>
void 
random(OLattice<T>& d, const Subset& s)
{
  while (1) {
    QDPJitArgs cudaArgs;

    static QDPJit::SharedLibEntry sharedLibEntry;
    static MapVolumes*              mapVolumes;
    static string                   strId;
    static string                   prg;

    int argNum = cudaArgs.addInt( s.numSiteTable() );
    int argOrd = cudaArgs.addBool( s.hasOrderedRep() );
    int argStart = cudaArgs.addInt( s.start() );
    int argSubset = cudaArgs.addIntPtr( (int*)QDPCache::Instance().getDevicePtr(s.getId()) );

    if (!mapVolumes) {

      string codeDest,codeRan_seed,codeRan_mult_n,codeLattice_ran_mult,codeLastSeed;
      string typeSeed;

      getTypeString( typeSeed , RNG::ran_seed , cudaArgs );
      if (!getCodeString( codeDest , d , "idx", cudaArgs )) { QDP_info(": could not cache dest"); break;  }      
      if (!getCodeString( codeRan_seed , RNG::ran_seed , "idx", cudaArgs )) { QDP_info(": could not cache RNG::ran_seed");break;}
      if (!getCodeString( codeRan_mult_n , 
			  RNG::ran_mult_n , 
			  "idx", cudaArgs )) { QDP_info(": could not cache RNG::ran_mult_n");break;}
      if (!getCodeString( codeLattice_ran_mult , 
			  *RNG::lattice_ran_mult , 
			  "idx", cudaArgs )) { QDP_info(": could not cache *RNG::lattice_ran_mult"); break;  }      


      ostringstream osId;
      osId << "random " << codeDest;
      strId = osId.str();
      xmlready(strId);

#ifdef GPU_DEBUG_DEEP
      cout << "strId = " << strId << endl;
#endif

      std::ostringstream sprg;
      sprg << "    " << typeSeed << " skewed_seed;" << endl;
      sprg << "    " << typeSeed << " seed;" << endl;
      sprg << "    int idx0 = blockDim.x * blockIdx.x + blockDim.x * gridDim.x * blockIdx.y + threadIdx.x; \n";
      sprg << "    if (idx0 < " << cudaArgs.getCode(argNum) << ") {\n";
      sprg << "      int idx  = " << cudaArgs.getCode(argSubset) << "[idx0];" << endl;
      sprg << "      seed = "        << codeRan_seed << ";" << endl;
      sprg << "      skewed_seed = " << codeRan_seed << " * " << codeLattice_ran_mult << ";" << endl;
      sprg << "      fill_random(" << codeDest << ", seed , skewed_seed, " << codeRan_mult_n << ");" << endl;
      sprg << "      if (idx0 == " << cudaArgs.getCode(argNum) << "-1) {" << endl;
      sprg << "        " << codeRan_seed << " = seed;" << endl;
      sprg << "      }" << endl;
      sprg << "    }" << endl;
      prg = sprg.str();

#ifdef GPU_DEBUG_DEEP
      cout << "Cuda kernel code = " << endl << prg << endl << endl;
#endif

    } else {
      if (!cacheLock( d , cudaArgs )) { QDP_info(": could not cache dest"); break;  }      
      if (!cacheLock( RNG::ran_seed , cudaArgs )) { QDP_info(": could not cache RNG::ran_seed");break;}
      if (!cacheLock( RNG::ran_mult_n , cudaArgs )) { QDP_info(": could not cache RNG::ran_mult_n");break;}
      if (!cacheLock( *RNG::lattice_ran_mult , cudaArgs )) { QDP_info(": could not cache *RNG::lattice_ran_mult"); break;  }      
    }

    QDP_debug("random(Lat,Subset) dev");

    if (!QDPJit::Instance()( strId , prg , cudaArgs.getDevPtr() , s.numSiteTable() , sharedLibEntry  , mapVolumes)) {
      QDP_info("eval(Lat,Lat) call to cuda jitter failed");
      break;
    }

    return;
  }

  QDP_debug("random(Lat,Subset) host");

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

  while (1) {

    QDPJitArgs cudaArgs;

    static QDPJit::SharedLibEntry sharedLibEntry;
    static MapVolumes*              mapVolumes;
    static string                   strId;
    static string                   prg;

    int argNum = cudaArgs.addInt( s.numSiteTable() );
    int argOrd = cudaArgs.addBool( s.hasOrderedRep() );
    int argStart = cudaArgs.addInt( s.start() );
    int argSubset = cudaArgs.addPtr( QDPCache::Instance().getDevicePtr( s.getId() ) );

    if (!mapVolumes) {

      string codeDest,codeR1,codeR2;

      if (!getCodeString( codeDest , d , "idx", cudaArgs )) { QDP_info("gaussian: could not cache dest"); break; }
      if (!getCodeString( codeR1 , r1 , "idx", cudaArgs )) { QDP_info("gaussian: could not cache r1"); break; }
      if (!getCodeString( codeR2 , r2 , "idx", cudaArgs )) { QDP_info("gaussian: could not cache r2"); break; }

      ostringstream osId;
      osId << "gaussian " << codeDest;
      strId = osId.str();
      xmlready(strId);

#ifdef GPU_DEBUG_DEEP
      cout << "strId = " << strId << endl;
#endif

      ostringstream sprg;
      sprg << "  if (" << cudaArgs.getCode(argOrd) << ") {" << endl;
      sprg << "    int idx = " << cudaArgs.getCode(argStart) << " + blockDim.x * blockIdx.x + blockDim.x * gridDim.x * blockIdx.y + threadIdx.x;" << endl;
      sprg << "    if (idx < " << cudaArgs.getCode(argNum) << " + " << cudaArgs.getCode(argStart) << ") {" << endl;
      sprg << "      fill_gaussian( " << codeDest << " , " << codeR1 << " , " << codeR2 << " );" << endl;
      sprg << "    }" << endl;
      sprg << "  } else {" << endl;
      sprg << "    int idx0 = blockDim.x * blockIdx.x + blockDim.x * gridDim.x * blockIdx.y + threadIdx.x;" << endl;
      sprg << "    if (idx0 < " << cudaArgs.getCode(argNum) << ") {" << endl;
      sprg << "      int idx  = ((int*)" << cudaArgs.getCode(argSubset) << ")[idx0];" << endl;
      sprg << "      fill_gaussian( " << codeDest << " , " << codeR1 << " , " << codeR2 << " );" << endl;
      sprg << "    }" << endl;
      sprg << "  }" << endl;
      prg = sprg.str();

#ifdef GPU_DEBUG_DEEP
      cout << "Cuda kernel code = " << endl << prg << endl << endl;
#endif

    } else {
      if (!cacheLock(  d , cudaArgs )) { QDP_info("gaussian: could not cache dest"); break; }
      if (!cacheLock(  r1 , cudaArgs )) { QDP_info("gaussian: could not cache r1"); break; }
      if (!cacheLock(  r2 , cudaArgs )) { QDP_info("gaussian: could not cache r2"); break; }
    }

    QDP_debug("gaussian(Lat,Subset) dev");
      
    if (!QDPJit::Instance()( strId , prg , cudaArgs.getDevPtr() , s.numSiteTable() , sharedLibEntry  , mapVolumes)) {
      QDP_info("gaussian(Lat,subset) call to cuda jitter failed");
      break;
    }

    return;
  }

  QDP_debug("gaussian(Lat,Subset) host");

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




template<class T> 
inline 
void zero_rep(OLattice<T>& dest, const Subset& s) 
{
  // Don't pass pretty function to QDP_info, buffer length!!
#ifdef GPU_DEBUG_DEEP
  cout << __PRETTY_FUNCTION__ << endl;
#endif

  while (1) {

    QDPJitArgs cudaArgs;

    static QDPJit::SharedLibEntry sharedLibEntry;
    static MapVolumes*              mapVolumes;
    static string                   strId;
    static string                   prg;

    int argNum = cudaArgs.addInt( s.numSiteTable() );
    int argOrd = cudaArgs.addBool( s.hasOrderedRep() );
    int argStart = cudaArgs.addInt( s.start() );
    int argSubset = cudaArgs.addPtr( QDPCache::Instance().getDevicePtr( s.getId() ) );

    if (!mapVolumes) {

      string codeDest;

      if (!getCodeString( codeDest , dest , "idx", cudaArgs )) { QDP_info("zero_rep: could not cache dest"); break; }

      ostringstream osId;
      osId << "zero_rep_lat_subset " << codeDest;
      strId = osId.str();
      xmlready(strId);

#ifdef GPU_DEBUG_DEEP
      cout << "strId = " << strId << endl;
#endif

      ostringstream sprg;
      sprg << "  if (" << cudaArgs.getCode(argOrd) << ") {" << endl;
      sprg << "    int idx = " << cudaArgs.getCode(argStart) << " + blockDim.x * blockIdx.x + blockDim.x * gridDim.x * blockIdx.y + threadIdx.x; \n";
      sprg << "    if (idx < " << cudaArgs.getCode(argNum) << " + " << cudaArgs.getCode(argStart) << ") {\n";
      sprg << "      zero_rep( " << codeDest << " );\n";
      sprg << "    }" << endl;
      sprg << "  } else {" << endl;
      sprg << "    int idx0 = blockDim.x * blockIdx.x + blockDim.x * gridDim.x * blockIdx.y + threadIdx.x; \n";
      sprg << "    if (idx0 < " << cudaArgs.getCode(argNum) << ") {\n";
      sprg << "      int idx  = ((int*)" << cudaArgs.getCode(argSubset) << ")[idx0];" << endl;
      sprg << "      zero_rep( " << codeDest << " );\n";
      sprg << "    }" << endl;
      sprg << "  }" << endl;
      prg = sprg.str();

#ifdef GPU_DEBUG_DEEP
      cout << "Cuda kernel code = " << endl << prg << endl << endl;
#endif

    } else {
      if (!cacheLock( dest , cudaArgs )) { QDP_info("zero_rep: could not cache dest"); break; }
    }

    QDP_debug("zero_rep(Lat,Subset) dev");

    if (!QDPJit::Instance()( strId , prg , cudaArgs.getDevPtr() , s.numSiteTable() , sharedLibEntry  , mapVolumes)) {
      QDP_info("zero_rep(Lat,subset) call to cuda jitter failed");
      break;
    }

    return;
  }

  QDP_debug("zero_rep(Lat,Subset) host");

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



template<class T> 
void zero_rep(OLattice<T>& dest) 
{
  // Don't pass pretty function to QDP_info, buffer length!!
#ifdef GPU_DEBUG_DEEP
  cout << __PRETTY_FUNCTION__ << endl;
#endif

  while (1) {
    QDPJitArgs cudaArgs;

    static QDPJit::SharedLibEntry sharedLibEntry;
    static MapVolumes*              mapVolumes;
    static string                   strId;
    static string                   prg;

    int argNum = cudaArgs.addInt( Layout::sitesOnNode() );

    if (!mapVolumes) {

      string codeDest;

      if (!getCodeString( codeDest , dest , "idx", cudaArgs )) { QDP_info("zero_rep: could not cache dest"); break;  }

      ostringstream osId;
      osId << "zero_rep_lat " << codeDest;
      strId = osId.str();
      xmlready(strId);

#ifdef GPU_DEBUG_DEEP
      cout << "strId = " << strId << endl;
#endif

      ostringstream sprg;
      sprg << "  int idx = blockDim.x * gridDim.x * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;\n";
      sprg << "  if (idx < " << cudaArgs.getCode(argNum) << ")" << endl;
      sprg << "    zero_rep( " << codeDest << " );\n";
      prg = sprg.str();

#ifdef GPU_DEBUG_DEEP
      cout << "Cuda kernel code = " << endl << prg << endl << endl;
#endif
    } else {
      if (!cacheLock( dest , cudaArgs )) { QDP_info("zero_rep: could not cache dest"); break;  }      
    }

    QDP_debug("zero_rep(Lat) dev");

    if (!QDPJit::Instance()( strId , prg , cudaArgs.getDevPtr() , Layout::sitesOnNode() , sharedLibEntry  , mapVolumes)) {
      QDP_info("zero_rep(Lat) call to cuda jitter failed");
      break;
    }

    return;
  }

  QDP_debug("zero_rep(Lat) host");
  
  const int vvol = Layout::sitesOnNode();
  for(int i=0; i < vvol; ++i) 
    zero_rep(dest.elem(i));

}




#if 0
template<class T> 
void zero_rep(OScalar<T>& dest) 
{
  zero_rep(dest.elem());
}
#endif


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





//
//  DOUBLE PRECISION version
//
template<class T1>
typename UnaryReturn<OLattice<T1>, FnSum>::Type_t
sum(const OLattice<T1>& s1, const Subset& s)
{
  // Don't pass pretty function to QDP_info, buffer length!!
#ifdef GPU_DEBUG_DEEP
  cout << __PRETTY_FUNCTION__ << endl;
#endif

  while (1) {

    // DP version of T1
    typedef typename UnaryReturn<OLattice<T1>, FnSum>::Type_t::SubType_t T2;

    QDP_debug("sum(lat,subset) dev");

    T2 * out_dev;
    T2 * in_dev;

    typename UnaryReturn<OLattice<T1>, FnSum>::Type_t  d;

    int actsize=s.numSiteTable();
    bool first=true;
    while (1) {

      int numThreads = DeviceParams::Instance().getMaxBlockX();
      while ((numThreads*sizeof(T2) > DeviceParams::Instance().getMaxSMem()) || (numThreads > actsize)) {
	numThreads >>= 1;
      }
      int numBlocks=(int)ceil(float(actsize)/numThreads);

      if (numBlocks > DeviceParams::Instance().getMaxGridX()) {
	QDP_info( "sum(Lat,subset) numBlocks > %d, continue on host",(int)DeviceParams::Instance().getMaxGridX());
	break;
      }

#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep("sum(Lat,subset): using %d threads per block, %d blocks, shared mem=%d" , 
		     numThreads , numBlocks , numThreads*sizeof(T2) );
#endif

      if (first) {
	if (!QDPCache::Instance().allocate_device_static( (void**)&out_dev , numBlocks*sizeof(T2) ))
	  QDP_error_exit( "sum(lat,subset) reduction buffer: 1st buffer no memory, exit");
	if (!QDPCache::Instance().allocate_device_static( (void**)&in_dev , numBlocks*sizeof(T2) ))
	  QDP_error_exit( "sum(lat,subset) reduction buffer: 2nd buffer no memory, exit");
      }

      if (numBlocks == 1) {
	if (first)
	  reduce_convert_indirection<T1,T2>(actsize, numThreads, numBlocks,  
					    (T1*)s1.getFdev() , (T2*)d.getFdev(), true , (int*)QDPCache::Instance().getDevicePtr(s.getId()) );
	else
	  reduce_convert_indirection<T2,T2>(actsize, numThreads, numBlocks,  
					    in_dev, (T2*)d.getFdev(), false , NULL );
      } else {
	if (first)
	  reduce_convert_indirection<T1,T2>(actsize, numThreads, numBlocks,  
					    (T1*)s1.getFdev(), out_dev , true , (int*)QDPCache::Instance().getDevicePtr(s.getId()) );
	else
	  reduce_convert_indirection<T2,T2>(actsize, numThreads, numBlocks, 
					    in_dev , out_dev , false , NULL );

      }

      first =false;

      if (numBlocks==1) 
	break;

      actsize=numBlocks;

      T2 * tmp = in_dev;
      in_dev = out_dev;
      out_dev = tmp;
    }

    QDPCache::Instance().free_device_static( in_dev );
    QDPCache::Instance().free_device_static( out_dev );

    QDPInternal::globalSum(d);

    return d;
  }

  QDP_debug("sum(lat,subset) host");
  

  typename UnaryReturn<OLattice<T1>, FnSum>::Type_t d;

  // Must initialize to zero since we do not know if the loop will be entered
  zero_rep(d.elem());

  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
    {
      int i = tab[j];
      d.elem() += s1.elem(i);
    }
  
  // Do a global sum on the result
  QDPInternal::globalSum(d);
  
  return d;
}




template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnSum>::Type_t
sum(const QDPExpr<RHS,OLattice<T> >& s1, const Subset& s)
{
  QDP_debug("sum(Expr,Subset)");
  OLattice<T> l;
  l[s]=s1;
  return sum(l,s);
}






//
// DOUBLE PRECISION version
//
template<class T1>
typename UnaryReturn<OLattice<T1>, FnSum>::Type_t
sum(const OLattice<T1>& s1)
{
  // Don't pass pretty function to QDP_info, buffer length!!
#ifdef GPU_DEBUG_DEEP
  cout << __PRETTY_FUNCTION__ << endl;
#endif

  const int nodeSites = Layout::sitesOnNode();

  while(1) {

    // T2 is the upcasted version of T1
    typedef typename UnaryReturn<OLattice<T1>, FnSum>::Type_t::SubType_t T2;

    QDP_debug("sum(Lat) dev");

    T2 * out_dev;
    T2 * in_dev;

    // This is DP
    typename UnaryReturn<OLattice<T1>, FnSum>::Type_t  d;

    int actsize=nodeSites;
    bool first=true;
    while (1) {

      int numThreads = DeviceParams::Instance().getMaxBlockX();
      while ((numThreads*sizeof(T2) > DeviceParams::Instance().getMaxSMem()) || (numThreads > actsize)) {
	numThreads >>= 1;
      }
      int numBlocks=(int)ceil(float(actsize)/numThreads);

      if (numBlocks > DeviceParams::Instance().getMaxGridX()) {
	QDP_info( "sum(Lat) numBlocks > %d, continue on host",(int)DeviceParams::Instance().getMaxGridX());
	break;
      }

#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep("using %d threads per block, %d blocks, shared mem=%d" , numThreads , numBlocks , numThreads*sizeof(T2) );
#endif

      if (first) {
	if (!QDPCache::Instance().allocate_device_static( (void**)&out_dev , numBlocks*sizeof(T2) ))
	  QDP_error_exit( "sum(lat) reduction buffer: 1st buffer no memory, exit");
	if (!QDPCache::Instance().allocate_device_static( (void**)&in_dev , numBlocks*sizeof(T2) ))
	  QDP_error_exit( "sum(lat) reduction buffer: 2nd buffer no memory, exit");
      }

      if ( numBlocks == 1 ) {
	if (first)
	  reduce_convert_indirection<T1,T2>(actsize, numThreads, numBlocks,  
					    (T1*)s1.getFdev() , (T2*)d.getFdev(), false , NULL );
	else
	  reduce_convert_indirection<T2,T2>(actsize, numThreads, numBlocks, 
					    in_dev, (T2*)d.getFdev(), false , NULL );
      } else {
	if (first)
	  reduce_convert_indirection<T1,T2>(actsize, numThreads, numBlocks, 
					    (T1*)s1.getFdev() , out_dev , false , NULL );
	else
	  reduce_convert_indirection<T2,T2>(actsize, numThreads, numBlocks, 
					    in_dev, out_dev , false , NULL );
      }

      first=false;

      if (numBlocks==1)
	break;

      actsize=numBlocks;

      T2 * tmp = in_dev;
      in_dev = out_dev;
      out_dev = tmp;

    }

    QDPCache::Instance().free_device_static( in_dev );
    QDPCache::Instance().free_device_static( out_dev );

    QDPInternal::globalSum(d);

    return d;
  }

  QDP_debug("sum(Lat) host");

  typename UnaryReturn<OLattice<T1>, FnSum>::Type_t  d;
  zero_rep(d.elem());

  for(int i=0; i < nodeSites; ++i)
    d.elem() += s1.elem(i);

  QDPInternal::globalSum(d);

  return d;
}








template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnSum>::Type_t
sum(const QDPExpr<RHS,OLattice<T> >& s1)
{
  QDP_debug("sum(Expr)");
  OLattice<T> l(s1);
  return sum(l);
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





//
// sumMulti DOUBLE PRECISION
//
template<class T1>
typename UnaryReturn<OLattice<T1>, FnSumMulti>::Type_t
sumMulti( const OLattice<T1>& s1 , const Set& ss )
{


  // Don't pass pretty function to QDP_info, buffer length!!
#ifdef GPU_DEBUG_DEEP
  cout << __PRETTY_FUNCTION__ << endl;
#endif

  typename UnaryReturn<OLattice<T1>, FnSumMulti>::Type_t  dest(ss.numSubsets());

  const int nodeSites = Layout::sitesOnNode();

#if 1
  while(1) {

    //
    // T2 is the upcasted version of T1
    // FnSum is okay to use here!
    //
    typedef typename UnaryReturn<OLattice<T1>, FnSum>::Type_t::SubType_t T2;

    if (!ss.enableGPU) {
      QDP_info("sumMulti called with a set, that is not supported for execution on the device");
      break;
    }

    QDP_debug("sumMulti(Lat) dev");

    QDP_debug("ss.largest_subset = " ,ss.largest_subset );
    int numThreads = ss.largest_subset > DeviceParams::Instance().getMaxBlockX() ? DeviceParams::Instance().getMaxBlockX() : ss.largest_subset;
    bool no_way=false;
    while ( (numThreads*sizeof(T2) > DeviceParams::Instance().getMaxSMem()) || 
	    (ss.largest_subset % numThreads) ) {
      numThreads >>= 1;
      if (numThreads == 0) {
	no_way=true;
	break;
      }
      string tmp = ss.largest_subset % numThreads ? "true":"false";
      QDP_debug("numThreads=%d subset size mod=%s" , numThreads , tmp.c_str());
    }

    if (no_way) {
      QDP_info( "sumMulti: No number of threads per blocks found that suits the requirements. Largest subset size = %d " , ss.largest_subset );
      break;
    }

    int numBlocks=(int)ceil(float(nodeSites)/numThreads);

#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("using %d threads per block smem=%d numBlocks=%d" , numThreads , numThreads*sizeof(T2) , numBlocks);
#endif

    if (numBlocks > 65535) {
      QDP_info( "sum(Lat) numBlocks > 65535, continue on host" );
      break;
    }

    T2 * out_dev;
    T2 * in_dev;

    if (!QDPCache::Instance().allocate_device_static( (void**)&out_dev , numBlocks*sizeof(T2) ))
      QDP_error_exit( "sumMulti(lat) reduction buffer: 2nd buffer no memory" );

    if (!QDPCache::Instance().allocate_device_static( (void**)&in_dev , numBlocks*sizeof(T2) ))
      QDP_error_exit("sumMulti(lat) reduction buffer: 3rd buffer no memory" );

    int virt_size = ss.largest_subset;

    int actsize=nodeSites;
    bool first=true;
    bool success=false;
    while (1) {

      QDP_debug("numBlocks=%d actsize=%d virt_size=%d",numBlocks,actsize,virt_size);

      if (first) {
	reduce_convert_indirection<T1,T2>(actsize, numThreads, numBlocks,  
					  (T1*)s1.getFdev(), out_dev , first, (int*)QDPCache::Instance().getDevicePtr( ss.getIdStrided() ) );
      } else {
	reduce_convert_indirection<T2,T2>(actsize, numThreads, numBlocks, 
					  in_dev, out_dev , first, (int*)QDPCache::Instance().getDevicePtr( ss.getIdStrided() ) );
      }

      if (first) {
	first =false;
      }

      T2 * tmp = in_dev;
      in_dev = out_dev;
      out_dev = tmp;
      
#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep( "checking for break numBlocks = %d %d" , numBlocks , ss.nonEmptySubsetsOnNode );
#endif
      if ( numBlocks == ss.nonEmptySubsetsOnNode ) {
	success=true;
	break;
      }

      virt_size /= numThreads;

      numThreads = virt_size > DeviceParams::Instance().getMaxBlockX() ? DeviceParams::Instance().getMaxBlockX() : virt_size;
      actsize = numBlocks;
      numBlocks=(int)ceil(float(actsize)/numThreads);

      QDP_debug("numThreads=%d numBlocks=%d",numThreads,numBlocks);

      no_way=false;
      while ( (numThreads*sizeof(T2) > DeviceParams::Instance().getMaxSMem()) || 
	      (virt_size % numThreads) ) {
#ifdef GPU_DEBUG_DEEP
	QDP_debug_deep( "loop entered %d %d %d" , numThreads*sizeof(T2) , virt_size , numThreads );
#endif
	numThreads >>= 1;
	if (numThreads == 0) {
	  no_way=true;
	  break;
	}
	numBlocks=(int)ceil(float(actsize)/numThreads);
	string tmp = virt_size % numThreads ? "true":"false";
#ifdef GPU_DEBUG_DEEP
	QDP_debug_deep("numThreads=%d subset size mod=%s" , numThreads , tmp.c_str() );
#endif
      }
      
      if (no_way) {
	QDP_info( "sumMulti: No number of threads per blocks found that suits the requirements. Largest subset size = %d",
		  ss.largest_subset);
	break;
      }

#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep("using %d threads per block smem=%d numBlocks=%d" , numThreads , numThreads*sizeof(T2) , numBlocks);
#endif
      
    }

    T2 * tmp = in_dev;
    in_dev = out_dev;
    out_dev = tmp;

    T2* slice = new T2[ss.numSubsets()];

    CudaMemcpy( (void*)slice , (void*)out_dev , ss.numSubsets()*sizeof(T2) );

    QDPCache::Instance().free_device_static( in_dev );
    QDPCache::Instance().free_device_static( out_dev );
    
    if (!success) {
      QDP_info("sumMulti: there was a problem, continue on host");
      delete[] slice;
      break;
    }

    for (int i = 0 ; i < ss.numSubsets() ; ++i ) {
      zero_rep(dest[i].elem());
    }

    QDP_debug("ss.stride_offset = %d ss.nonEmptySubsetsOnNode = %d" ,ss.stride_offset, ss.nonEmptySubsetsOnNode);

    //
    // "dest" and "slice" are both on DP
    //
    for (int i = 0 ; i < ss.nonEmptySubsetsOnNode ; ++i ) {
      dest[ ss.stride_offset + i ].elem() = slice[i];
    }
    
    delete[] slice;

    //
    // We need to unlock things, so that the global sum can be carried out
    //
    

    QDPInternal::globalSumArray(dest);

    return dest;
  }
#endif

  QDP_debug("sumMulti(Lat) host");


  // Initialize result with zero
  for(int k=0; k < ss.numSubsets(); ++k)
    zero_rep(dest[k]);

  // Loop over all sites and accumulate based on the coloring 
  const multi1d<int>& lat_color =  ss.latticeColoring();

  for(int i=0; i < nodeSites; ++i) 
  {
    int j = lat_color[i];
    dest[j].elem() += s1.elem(i);
  }

  // Do a global sum on the result
  QDPInternal::globalSumArray(dest);

  return dest;
}




template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t
sumMulti(const QDPExpr<RHS,OLattice<T> >& s1, const Set& ss)
{
  QDP_debug("sumMulti(Expr)");
  OLattice<T> l(s1);
  return sumMulti(l,ss);
}



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




template<class T>
typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t
globalMax(const OLattice<T>& s1)
{
  // Don't pass pretty function to QDP_info, buffer length!!
#ifdef GPU_DEBUG_DEEP
  cout << __PRETTY_FUNCTION__ << endl;
#endif

  const int nodeSites = Layout::sitesOnNode();

  while(1) {

    QDP_debug("globalMax(Lat) dev");

    typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t  d;

    T * out_dev;
    T * in_dev;

    int actsize=nodeSites;
    bool first=true;
    while (1) {

      int numThreads = DeviceParams::Instance().getMaxBlockX();
      while ((numThreads*sizeof(T) > DeviceParams::Instance().getMaxSMem()) || (numThreads > actsize)) {
	numThreads >>= 1;
      }
      int numBlocks=(int)ceil(float(actsize)/numThreads);

      if (numBlocks > DeviceParams::Instance().getMaxGridX()) {
	QDP_info( "globalMax(Lat) numBlocks > %d, continue on host",(int)DeviceParams::Instance().getMaxGridX());
	break;
      }

#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep("using %d threads per block, %d blocks, shared mem=%d" , numThreads , numBlocks , numThreads*sizeof(T) );
#endif

      if (first) {
	if (!QDPCache::Instance().allocate_device_static( (void**)&out_dev , numBlocks*sizeof(T) ))
	  QDP_error_exit( "globalMax(lat) reduction buffer: 1st buffer no memory, exit");
	if (!QDPCache::Instance().allocate_device_static( (void**)&in_dev , numBlocks*sizeof(T) ))
	  QDP_error_exit( "globalMax(lat) reduction buffer: 2nd buffer no memory, exit");
      }

      if ( numBlocks == 1 ) {
	if (first)
	  globalMax_kernel<T>(actsize, numThreads, numBlocks, s1.getFdev() , d.getFdev() );
	else
	  globalMax_kernel<T>(actsize, numThreads, numBlocks, in_dev, d.getFdev() );
      } else {
	if (first)
	  globalMax_kernel<T>(actsize, numThreads, numBlocks, s1.getFdev() , out_dev );
	else
	  globalMax_kernel<T>(actsize, numThreads, numBlocks, in_dev, out_dev );
      }

      first=false;

      if (numBlocks==1)
	break;

      actsize=numBlocks;

      T * tmp = in_dev;
      in_dev = out_dev;
      out_dev = tmp;

    }

    QDPCache::Instance().free_device_static( in_dev );
    QDPCache::Instance().free_device_static( out_dev );

    QDPInternal::globalMax(d);

    return d;
  }

  QDP_debug("globalMax(Lat) host");

  typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t  d;

  // Loop always entered so unroll
  d.elem() = s1.elem(0);

  const int vvol = Layout::sitesOnNode();
  for(int i=1; i < vvol; ++i) 
  {
    typename UnaryReturn<T, FnGlobalMax>::Type_t  dd = s1.elem(i);

    if (toBool(dd > d.elem()))
      d.elem() = dd;
  }

  // Do a global max on the result
  QDPInternal::globalMax(d); 

  return d;

}



#if 0
template<class T>
typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t
globalMax(const OLattice<T>& s1)
{
  // Don't pass pretty function to QDP_info, buffer length!!
#ifdef GPU_DEBUG_DEEP
  cout << __PRETTY_FUNCTION__ << endl;
#endif

  const int nodeSites = Layout::sitesOnNode();

  while(1) {

    // lock s1
    s1.getFdev();

    QDP_debug("globalMax(Lat) dev");

    T * orig_dev = (T*)s1.getFdev();

    int numThreads = DeviceParams::Instance().getMaxBlockX();
    while ( numThreads*sizeof(T) > DeviceParams::Instance().getMaxSMem() ) {
      numThreads >>= 1;
    }
    int numBlocks=(int)ceil(float(nodeSites)/numThreads);
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("using %d threads per block smem=%d " , numThreads , numThreads*sizeof(T) );
#endif

    if (numBlocks > 65535) {
      QDP_info( "globalMax(Lat) numBlocks > 65535, continue on host" );
      break;
    }

    T * out_dev;
    T * in_dev;
    T * in_dev_save;

    //
    // These buffers can be of significant size.
    // We use the spilling version of allocate.
    //
    if (!QDPCache::Instance().allocate_device_static( (void**)&out_dev , numBlocks*sizeof(T) ))
      QDP_error_exit( "globalMax(lat) reduction buffer: 1st buffer no memory" );

    if (!QDPCache::Instance().allocate_device_static( (void**)&in_dev_save , numBlocks*sizeof(T) ))
      QDP_error_exit( "globalMax(lat) reduction buffer: 2nd buffer no memory" );


    //
    // If s1 was spilled we give up.
    //
    if (!s1.onDevice()) {
      QDP_error_exit( "globalMax(lat) allocation spilled source object, continue on host" );
    }

    in_dev = orig_dev;

    int actsize=nodeSites;
    bool first=true;
    while (1) {

      QDP_debug("numBlocks=%d actsize=%d",numBlocks,actsize);

      globalMax_kernel<T>(actsize, numThreads, numBlocks, in_dev, out_dev);

      if (first) {
	first =false;
	in_dev=in_dev_save;
      }

      if (numBlocks==1) break;
      actsize=numBlocks;

      numThreads = DeviceParams::Instance().getMaxBlockX();
      while ((numThreads*sizeof(T) > DeviceParams::Instance().getMaxSMem()) || (numThreads > actsize)) {
	numThreads >>= 1;
      }
      numBlocks=(int)ceil(float(actsize)/numThreads);

#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep("using %d threads per block" , numThreads );
#endif

      T * tmp = in_dev;
      in_dev = out_dev;
      out_dev = tmp;

    }

    //
    // We were using default precision, now upcast to return precision
    //
    OScalar<T> d_sp;
  
    CudaMemcpy( (void*)d_sp.getF() , (void*)out_dev , 1*sizeof(T) );

    QDPCache::Instance().free_device_static( in_dev );
    QDPCache::Instance().free_device_static( out_dev );

    typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t  d(d_sp);

    //
    // We must unlock here, so that objects can get spilled
    //
    
    QDPInternal::globalMax(d);

    return d;
    
  }

  QDP_debug("globalMax(Lat) host");

  

  typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t  d;

#if 0
#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnGlobalMax(), s1);
  prof.time -= getClockTime();
#endif
#endif

  // Loop always entered so unroll
  d.elem() = s1.elem(0);

  const int vvol = Layout::sitesOnNode();
  for(int i=1; i < vvol; ++i) 
  {
    typename UnaryReturn<T, FnGlobalMax>::Type_t  dd = s1.elem(i);

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
#endif


template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t
globalMax(const QDPExpr<RHS,OLattice<T> >& s1)
{
  OLattice<T> l(s1);
  return globalMax(l);
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
  QDP_info("globalMin(Lat) not implemented for the GPU. Please report to devel.");

  typename UnaryReturn<OLattice<T>, FnGlobalMin>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnGlobalMin(), s1);
  prof.time -= getClockTime();
#endif

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


#if 0
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnGlobalMin>::Type_t
globalMin(const QDPExpr<RHS,OLattice<T> >& s1)
{
  OLattice<T> l(s1);
  return globalMin(l);
}
#endif

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
//struct FnMap;


#if defined(QDP_USE_PROFILING)   
template <>
struct TagVisitor<FnMap, PrintTag> : public ParenPrinter<FnMap>
{ 
  static void visit(FnMap op, PrintTag t) 
    { t.os_m << "shift"; }
};
#endif



// template<typename T>
// void printme() {
//   QDP_info_primary("%s",__PRETTY_FUNCTION__);
// }



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

    Expr subexpr(expr.child());

    if (map.offnodeP)
      {
#if QDP_DEBUG >= 3
	QDP_info("Map: off-node communications required");
#endif

	int dstnum = map.destnodes_num[0]*sizeof(InnerType_t);
	int srcnum = map.srcenodes_num[0]*sizeof(InnerType_t);

	const FnMapRsrc& rRSrc = fnmap.getResource(srcnum,dstnum);

	// const InnerType_t *send_buf_c = 
	// InnerType_t* send_buf = const_cast<InnerType_t*>(send_buf_c);
	// if ( send_buf == 0x0 ) { QDP_error_exit("QMP_get_memory_pointer returned NULL pointer from non NULL QMP_mem_t (send_buf)\n")
	//; }

	const int my_node = Layout::nodeNumber();

	// Make sure the inner expression's map function
	// send and receive before recursing down
	int maps_involved = forEach(subexpr, f , BitOrCombine());
	if (maps_involved > 0) {
	  ShiftPhase2 phase2;
	  forEach(subexpr, phase2 , NullCombine());
	}


	static QDPJit::SharedLibEntry sharedLibEntry;
	static MapVolumes*              mapVolumes;
	static string                   strId;
	static string                   prg;

	QDPJitArgs cudaArgs;

	int argNum = cudaArgs.addInt( map.soffsets.size() );
	int argSoffsets = cudaArgs.addIntPtr( (int*)QDPCache::Instance().getDevicePtr( map.getSoffsetsId() ) );
	int argSendBuf = cudaArgs.addPtr( rRSrc.getSendBufDevPtr() );

	if (!mapVolumes) {

	  string codeSubexpr,codeInnerT;

	  getTypeString( codeInnerT , InnerType_t() , cudaArgs );
	  if (!getCodeString( codeSubexpr , subexpr , "idx2", cudaArgs )) 
	    QDP_error_exit("eval: could not cache RHS");
	  
	  ostringstream osId;
	  osId << "shift_phase1 " << codeInnerT << codeSubexpr;
	  strId = osId.str();
	  xmlready(strId);
#ifdef GPU_DEBUG_DEEP
	  cout << "strId = " << strId << endl;
#endif
      
	  std::ostringstream sprg;
	  sprg << "  int idx = blockDim.x * blockIdx.x + blockDim.x * gridDim.x * blockIdx.y + threadIdx.x;" << endl;
	  sprg << "  if (idx < " << cudaArgs.getCode(argNum) << ") {\n";
	  sprg << "    int idx2 = ((int*)(" << cudaArgs.getCode(argSoffsets) << "))[idx];\n";
	  sprg << "    ((" << codeInnerT << "*)(" << cudaArgs.getCode(argSendBuf) << "))[idx] = " << codeSubexpr << ";\n";
	  sprg << "  }" << endl;
	  prg = sprg.str();

#ifdef GPU_DEBUG_DEEP
	  cout << "Cuda kernel code = " << endl << prg << endl << endl;
#endif
	} else {
	  if (!cacheLock( subexpr ,  cudaArgs ))
	    QDP_error_exit("shift_phase1: could not cache RHS");
	}

	QDP_debug("shift_phase1() dev!");

	if (!QDPJit::Instance()( strId , prg , cudaArgs.getDevPtr() , map.soffsets.size() , sharedLibEntry  , mapVolumes )) {
	  QDP_error_exit("shift_phase1 call to cuda jitter failed");
	}
      
#if 0
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
  //typedef typename ForEach<A, EvalLeaf1, OpCombine>::Type_t TypeA_t;
  //typedef typename Combine1<TypeA_t, FnMap, OpCombine>::Type_t Type_t;
  //typedef QDPExpr<A,OLattice<Type_t> > Expr;
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
    ForEach<A, ShiftPhase2, CTag>::apply(expr.child(), f, c);
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
