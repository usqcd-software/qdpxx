// -*- C++ -*-
// $Id: qdp_scalarsite_sse.h,v 1.15 2003-11-02 01:21:45 edwards Exp $

/*! @file
 * @brief Intel SSE optimizations
 *
 * SSE optimizations of basic operations
 */

#ifndef QDP_SCALARSITE_SSE_H
#define QDP_SCALARSITE_SSE_H

// These SSE asm instructions are only supported under GCC/G++
#if defined(__GNUC__)

QDP_BEGIN_NAMESPACE(QDP);

/*! @defgroup optimizations  Optimizations
 *
 * Optimizations for basic QDP operations
 *
 * @{
 */

// Use this def just to safe some typing later on in the file
typedef RComplex<PScalar<float> >  RComplexFloat;
typedef PDWVector<float,4>         PDWVectorFloat4;
typedef RComplex<PDWVectorFloat4>  RComplexFloat4;



typedef struct
{
   unsigned int c1,c2,c3,c4;
} sse_mask __attribute__ ((aligned (16)));

static sse_mask _sse_sgn13 __attribute__ ((unused)) ={0x80000000, 0x00000000, 0x80000000, 0x00000000};
static sse_mask _sse_sgn24 __attribute__ ((unused)) ={0x00000000, 0x80000000, 0x00000000, 0x80000000};
static sse_mask _sse_sgn3  __attribute__ ((unused)) ={0x00000000, 0x00000000, 0x80000000, 0x00000000};
static sse_mask _sse_sgn4  __attribute__ ((unused)) ={0x00000000, 0x00000000, 0x00000000, 0x80000000};


#include "scalarsite_sse/sse_mult_nn.h"
#include "scalarsite_sse/sse_mult_na.h"
#include "scalarsite_sse/sse_mult_an.h"
#include "scalarsite_sse/sse_mat_vec.h"
#include "scalarsite_sse/sse_adj_mat_vec.h"
#include "scalarsite_sse/sse_addvec.h"
#include "scalarsite_sse/sse_mat_hwvec.h"
#include "scalarsite_sse/sse_adj_mat_hwvec.h"

// #define QDP_SCALARSITE_DEBUG

// Optimized version of  
//    PColorMatrix<RComplexFloat,3> <- PColorMatrix<RComplexFloat,3> * PColorMatrix<RComplexFloat,3>
template<>
inline BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
  PMatrix<RComplexFloat,3,PColorMatrix>, OpMultiply>::Type_t
operator*(const PMatrix<RComplexFloat,3,PColorMatrix>& l, 
	  const PMatrix<RComplexFloat,3,PColorMatrix>& r)
{
  BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
    PMatrix<RComplexFloat,3,PColorMatrix>, OpMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "M*M" << endl;
#endif

  _inline_sse_mult_su3_nn(l,r,d);

  return d;
}


// Optimized version of  
//    PScalar<PColorMatrix<RComplexFloat,3>> <- PScalar<PColorMatrix<RComplexFloat,3>> * 
//                         PScalar<PColorMatrix<RComplexFloat,3>>
template<>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >,
  PScalar<PColorMatrix<RComplexFloat,3> >, OpMultiply>::Type_t
operator*(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	  const PScalar<PColorMatrix<RComplexFloat,3> >& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PScalar<PColorMatrix<RComplexFloat,3> >, OpMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "PSc<M>*PSc<M>" << endl;
#endif

  _inline_sse_mult_su3_nn(l.elem(),r.elem(),d.elem());

  return d;
}


// Optimized version of  
//    PScalar<PScalar<PColorMatrix<RComplexFloat,3>>> <- PScalar<PScalar<PColorMatrix<RComplexFloat,3>>> * 
//                         PScalar<PScalar<PColorMatrix<RComplexFloat,3>>>
template<>
inline BinaryReturn<PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >,
  PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, OpMultiply>::Type_t
operator*(const PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >& l, 
	  const PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >& r)
{
  BinaryReturn<PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, 
    PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, OpMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "PSc<PSc<M>>*PSc<PSc<M>>" << endl;
#endif

  _inline_sse_mult_su3_nn(l.elem().elem(),r.elem().elem(),d.elem().elem());

  return d;
}


// Optimized version of  
//   PColorMatrix<RComplexFloat,3> <- adj(PColorMatrix<RComplexFloat,3>) * PColorMatrix<RComplexFloat,3>
template<>
inline BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
  PMatrix<RComplexFloat,3,PColorMatrix>, OpAdjMultiply>::Type_t
adjMultiply(const PMatrix<RComplexFloat,3,PColorMatrix>& l, 
	    const PMatrix<RComplexFloat,3,PColorMatrix>& r)
{
  BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
    PMatrix<RComplexFloat,3,PColorMatrix>, OpAdjMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "adj(M)*M" << endl;
#endif

  _inline_sse_mult_su3_an(l,r,d);

  return d;
}


// Optimized version of  
//   PScalar<PColorMatrix<RComplexFloat,3>> <- adj(PScalar<PColorMatrix<RComplexFloat,3>>) * PScalar<PColorMatrix<RComplexFloat,3>>
template<>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PScalar<PColorMatrix<RComplexFloat,3> >, OpAdjMultiply>::Type_t
adjMultiply(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	    const PScalar<PColorMatrix<RComplexFloat,3> >& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PScalar<PColorMatrix<RComplexFloat,3> >, OpAdjMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "adj(PSc<M>)*PSc<M>" << endl;
#endif

  _inline_sse_mult_su3_an(l.elem(),r.elem(),d.elem());

  return d;
}


// Optimized version of  
//   PScalar<PScalar<PColorMatrix<RComplexFloat,3>>> <- adj(PScalar<PScalar<PColorMatrix<RComplexFloat,3>>>) * 
//        PScalar<PScalar<PColorMatrix<RComplexFloat,3>>>
template<>
inline BinaryReturn<PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, 
  PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, OpAdjMultiply>::Type_t
adjMultiply(const PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >& l, 
	    const PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >& r)
{
  BinaryReturn<PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, 
    PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, OpAdjMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "adj(PSc<PSc<M>>)*PSc<PSc<M>>" << endl;
#endif

  _inline_sse_mult_su3_an(l.elem().elem(),r.elem().elem(),d.elem().elem());

  return d;
}


// Optimized version of  
//   PColorMatrix<RComplexFloat,3> <- PColorMatrix<RComplexFloat,3> * adj(PColorMatrix<RComplexFloat,3>)
template<>
inline BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
  PMatrix<RComplexFloat,3,PColorMatrix>, OpMultiplyAdj>::Type_t
multiplyAdj(const PMatrix<RComplexFloat,3,PColorMatrix>& l, 
	    const PMatrix<RComplexFloat,3,PColorMatrix>& r)
{
  BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
    PMatrix<RComplexFloat,3,PColorMatrix>, OpMultiplyAdj>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "M*adj(M)" << endl;
#endif

  _inline_sse_mult_su3_na(l,r,d);

  return d;
}


// Optimized version of  
//   PScalar<PColorMatrix<RComplexFloat,3>> <- PScalar<PColorMatrix<RComplexFloat,3>> * 
//          adj(PScalar<PColorMatrix<RComplexFloat,3>>)
template<>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PScalar<PColorMatrix<RComplexFloat,3> >, OpMultiplyAdj>::Type_t
multiplyAdj(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	    const PScalar<PColorMatrix<RComplexFloat,3> >& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PScalar<PColorMatrix<RComplexFloat,3> >, OpMultiplyAdj>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "PSc<M>*adj(PSc<M>)" << endl;
#endif

  _inline_sse_mult_su3_na(l.elem(),r.elem(),d.elem());

  return d;
}


// Optimized version of  
//   PScalar<Pscalar<PColorMatrix<RComplexFloat,3>>> <- PScalar<PScalar<PColorMatrix<RComplexFloat,3>>> * 
//           adj(PScalar<PScalar<PColorMatrix<RComplexFloat,3>>>)
template<>
inline BinaryReturn<PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, 
  PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, OpMultiplyAdj>::Type_t
multiplyAdj(const PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >& l, 
	    const PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >& r)
{
  BinaryReturn<PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, 
    PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, OpMultiplyAdj>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "PSc<PSc<M>>*adj(PSc<PSc<M>>)" << endl;
#endif

  _inline_sse_mult_su3_na(l.elem().elem(),r.elem().elem(),d.elem().elem());

  return d;
}


#if 0
// Ooops, this macro does not exist!!

// Optimized version of  
//   PColorMatrix<RComplexFloat,3> <- adj(PColorMatrix<RComplexFloat,3>) * adj(PColorMatrix<RComplexFloat,3>)
template<>
inline BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
  PMatrix<RComplexFloat,3,PColorMatrix>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(const PMatrix<RComplexFloat,3,PColorMatrix>& l, 
	       const PMatrix<RComplexFloat,3,PColorMatrix>& r)
{
  BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
    PMatrix<RComplexFloat,3,PColorMatrix>, OpAdjMultiplyAdj>::Type_t  d;

  _inline_sse_mult_su3_aa(l,r,d);

  return d;
}
#endif


// Optimized version of  
//    PColorVector<RComplexFloat,3> <- PColorMatrix<RComplexFloat,3> * PColorVector<RComplexFloat,3>
template<>
inline BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
  PVector<RComplexFloat,3,PColorVector>, OpMultiply>::Type_t
operator*(const PMatrix<RComplexFloat,3,PColorMatrix>& l, 
	  const PVector<RComplexFloat,3,PColorVector>& r)
{
  BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
    PVector<RComplexFloat,3,PColorVector>, OpMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "M*V" << endl;
#endif

  _inline_sse_mult_su3_mat_vec(l,r,d);

  return d;
}


// Optimized version of  
//    PScalar<PColorVector<RComplexFloat,3>> <- PScalar<PColorMatrix<RComplexFloat,3>> * PScalar<PColorVector<RComplexFloat,3>>
template<>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PScalar<PColorVector<RComplexFloat,3> >, OpMultiply>::Type_t
operator*(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	  const PScalar<PColorVector<RComplexFloat,3> >& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PScalar<PColorVector<RComplexFloat,3> >, OpMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "PSc<M>*PSc<V>" << endl;
#endif

  _inline_sse_mult_su3_mat_vec(l.elem(),r.elem(),d.elem());

  return d;
}


// Optimized version of  
//    PScalar<PScalar<PColorVector<RComplexFloat,3>>> <- PScalar<PScalar<PColorMatrix<RComplexFloat,3>>> * 
//          PScalar<PScalar<PColorVector<RComplexFloat,3>>>
template<>
inline BinaryReturn<PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, 
  PScalar<PScalar<PColorVector<RComplexFloat,3> > >, OpMultiply>::Type_t
operator*(const PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >& l, 
	  const PScalar<PScalar<PColorVector<RComplexFloat,3> > >& r)
{
  BinaryReturn<PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, 
    PScalar<PScalar<PColorVector<RComplexFloat,3> > >, OpMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "PSc<PSc<M>>*PSc<PSc<V>>" << endl;
#endif

  _inline_sse_mult_su3_mat_vec(l.elem().elem(),r.elem().elem(),d.elem().elem());

  return d;
}


// Optimized version of  
//    PColorVector<RComplexFloat,3> <- adj(PColorMatrix<RComplexFloat,3>) * PColorVector<RComplexFloat,3>
template<>
inline BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
  PVector<RComplexFloat,3,PColorVector>, OpAdjMultiply>::Type_t
adjMultiply(const PMatrix<RComplexFloat,3,PColorMatrix>& l, 
	    const PVector<RComplexFloat,3,PColorVector>& r)
{
  BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
    PVector<RComplexFloat,3,PColorVector>, OpAdjMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "adj(M)*V" << endl;
#endif

  _inline_sse_mult_adj_su3_mat_vec(l,r,d);

  return d;
}


// Optimized version of   StaggeredFermion <- ColorMatrix*StaggeredFermion
//    PSpinVector<PColorVector<RComplexFloat,3>,1> <- PScalar<PColorMatrix<RComplexFloat,3>> * PSpinVector<PColorVector<RComplexFloat,3>,1>
template<>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PVector<PColorVector<RComplexFloat,3>,1,PSpinVector>, OpMultiply>::Type_t
operator*(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	  const PVector<PColorVector<RComplexFloat,3>,1,PSpinVector>& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PVector<PColorVector<RComplexFloat,3>,1,PSpinVector>, OpMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "M*S" << endl;
#endif

  _inline_sse_mult_su3_mat_vec(l.elem(),r.elem(0),d.elem(0));

  return d;
}


// Optimized version of   StaggeredFermion <- ColorMatrix*StaggeredFermion
//    PScalar<PSpinVector<PColorVector<RComplexFloat,3>,1>> <- PScalar<PScalar<PColorMatrix<RComplexFloat,3>>> * 
//         PScalar<PSpinVector<PColorVector<RComplexFloat,3>,1>>
template<>
inline BinaryReturn<PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, 
  PScalar<PSpinVector<PColorVector<RComplexFloat,3>,1> >, OpMultiply>::Type_t
operator*(const PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >& l, 
	  const PScalar<PSpinVector<PColorVector<RComplexFloat,3>,1> >& r)
{
  BinaryReturn<PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, 
    PScalar<PSpinVector<PColorVector<RComplexFloat,3>,1> >, OpMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "PSc<M>*PSc<S>" << endl;
#endif

  _inline_sse_mult_su3_mat_vec(l.elem().elem(),r.elem().elem(0),d.elem().elem(0));

  return d;
}


// Optimized version of  
//    PScalar<PColorVector<RComplexFloat,3>> <- adj(PScalar<PColorMatrix<RComplexFloat,3>>) * PScalar<PColorVector<RComplexFloat,3>>
template<>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PScalar<PColorVector<RComplexFloat,3> >, OpAdjMultiply>::Type_t
adjMultiply(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	    const PScalar<PColorVector<RComplexFloat,3> >& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PScalar<PColorVector<RComplexFloat,3> >, OpAdjMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "adj(PSc<M>)*PSc<V>" << endl;
#endif

  _inline_sse_mult_adj_su3_mat_vec(l.elem(),r.elem(),d.elem());

  return d;
}


// Optimized version of  
//    PScalar<PScalar<PColorVector<RComplexFloat,3>>> <- adj(PScalar<PScalar<PColorMatrix<RComplexFloat,3>>>) * 
//       PScalar<PScalar<PColorVector<RComplexFloat,3>>>
template<>
inline BinaryReturn<PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, 
  PScalar<PScalar<PColorVector<RComplexFloat,3> > >, OpAdjMultiply>::Type_t
adjMultiply(const PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >& l, 
	    const PScalar<PScalar<PColorVector<RComplexFloat,3> > >& r)
{
  BinaryReturn<PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, 
    PScalar<PScalar<PColorVector<RComplexFloat,3> > >, OpAdjMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "adj(PSc<PSc<M>>)*PSc<PSc<V>>" << endl;
#endif

  _inline_sse_mult_adj_su3_mat_vec(l.elem().elem(),r.elem().elem(),d.elem().elem());

  return d;
}


// Optimized version of   StaggeredFermion <- adj(ColorMatrix)*StaggeredFermion
//    PSpinVector<PColorVector<RComplexFloat,3>,1> <- adj(PScalar<PColorMatrix<RComplexFloat,3>>) * PSpinVector<PColorVector<RComplexFloat,3>,1>
template<>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PVector<PColorVector<RComplexFloat,3>,1,PSpinVector>, OpAdjMultiply>::Type_t
adjMultiply(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	    const PVector<PColorVector<RComplexFloat,3>,1,PSpinVector>& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PVector<PColorVector<RComplexFloat,3>,1,PSpinVector>, OpAdjMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "adj(PSc<M>)*S" << endl;
#endif

  _inline_sse_mult_adj_su3_mat_vec(l.elem(),r.elem(0),d.elem(0));

  return d;
}


// Optimized version of   StaggeredFermion <- adj(ColorMatrix)*StaggeredFermion
//    PScalar<PSpinVector<PColorVector<RComplexFloat,3>,1>> <- adj(PScalar<PScalar<PColorMatrix<RComplexFloat,3>>>) * 
//         PScalar<PSpinVector<PColorVector<RComplexFloat,3>,1>>
template<>
inline BinaryReturn<PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, 
  PScalar<PSpinVector<PColorVector<RComplexFloat,3>,1> >, OpAdjMultiply>::Type_t
adjMultiply(const PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >& l, 
	    const PScalar<PSpinVector<PColorVector<RComplexFloat,3>,1> >& r)
{
  BinaryReturn<PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, 
    PScalar<PSpinVector<PColorVector<RComplexFloat,3>,1> >, OpAdjMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "adj(PSc<PSc<M>>)*S" << endl;
#endif

  _inline_sse_mult_adj_su3_mat_vec(l.elem().elem(),r.elem().elem(0),d.elem().elem(0));

  return d;
}


// Optimized version of    HalfFermion <- ColorMatrix*HalfFermion
//    PSpinVector<PColorVector<RComplexFloat,3>,2> <- PScalar<PColorMatrix<RComplexFloat,3>> * 
//                     PSpinVector<ColorVector<RComplexFloat,3>,2>
template<>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PVector<PColorVector<RComplexFloat,3>,2,PSpinVector>, OpMultiply>::Type_t
operator*(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
          const PVector<PColorVector<RComplexFloat,3>,2,PSpinVector>& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PVector<PColorVector<RComplexFloat,3>,2,PSpinVector>, OpMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "PSc<M>*H" << endl;
#endif

#if 0
  _inline_sse_mult_su3_mat_hwvec(l,r,d);
#else
  _inline_sse_mult_su3_mat_vec(l.elem(),r.elem(0),d.elem(0));
  _inline_sse_mult_su3_mat_vec(l.elem(),r.elem(1),d.elem(1));
#endif

  return d;
}


// Optimized version of    HalfFermion <- ColorMatrix*HalfFermion
//    PScalar<PSpinVector<PColorVector<RComplexFloat,3>,2>> <- PScalar<PScalar<PColorMatrix<RComplexFloat,3>>> * 
//                     PScalar<PSpinVector<ColorVector<RComplexFloat,3>,2>>
template<>
inline BinaryReturn<PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, 
  PScalar<PSpinVector<PColorVector<RComplexFloat,3>,2> >, OpMultiply>::Type_t
operator*(const PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >& l, 
          const PScalar<PSpinVector<PColorVector<RComplexFloat,3>,2> >& r)
{
  BinaryReturn<PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, 
    PScalar<PSpinVector<PColorVector<RComplexFloat,3>,2> >, OpMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "PSc<PSc<M>>*H" << endl;
#endif

#if 0
  _inline_sse_mult_su3_mat_hwvec(l.elem(),r.elem(),d.elem());
#else
  _inline_sse_mult_su3_mat_vec(l.elem().elem(),r.elem().elem(0),d.elem().elem(0));
  _inline_sse_mult_su3_mat_vec(l.elem().elem(),r.elem().elem(1),d.elem().elem(1));
#endif

  return d;
}


// Optimized version of    HalfFermion <- ColorMatrix*HalfFermion
//    PSpinVector<PColorVector<RComplexFloat,3>,2> <- adj(PScalar<PColorMatrix<RComplexFloat,3>>) * 
//                     PSpinVector<ColorVector<RComplexFloat,3>,2>
template<>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PVector<PColorVector<RComplexFloat,3>,2,PSpinVector>, OpAdjMultiply>::Type_t
adjMultiply(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
            const PVector<PColorVector<RComplexFloat,3>,2,PSpinVector>& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PVector<PColorVector<RComplexFloat,3>,2,PSpinVector>, OpAdjMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "adj(PSc<M>)*H" << endl;
#endif

  _inline_sse_mult_adj_su3_mat_hwvec(l,r,d);

  return d;
}


// Optimized version of    HalfFermion <- ColorMatrix*HalfFermion
//    PScalar<PSpinVector<PColorVector<RComplexFloat,3>,2>> <- adj(PScalar<PScalar<PColorMatrix<RComplexFloat,3>>>) * 
//                     PScalar<PSpinVector<ColorVector<RComplexFloat,3>,2>>
template<>
inline BinaryReturn<PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, 
  PScalar<PSpinVector<PColorVector<RComplexFloat,3>,2> >, OpAdjMultiply>::Type_t
adjMultiply(const PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >& l, 
            const PScalar<PSpinVector<PColorVector<RComplexFloat,3>,2> >& r)
{
  BinaryReturn<PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, 
    PScalar<PSpinVector<PColorVector<RComplexFloat,3>,2> >, OpAdjMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "adj(PSc<PSc<M>>)*H" << endl;
#endif

  _inline_sse_mult_adj_su3_mat_hwvec(l.elem(),r.elem(),d.elem());

  return d;
}


// Optimized version of  
//    PColorVector<RComplexFloat,3> <- PColorVector<RComplexFloat,3> + PColorVector<RComplexFloat,3>
template<>
inline BinaryReturn<PVector<RComplexFloat,3,PColorVector>, 
  PVector<RComplexFloat,3,PColorVector>, OpAdd>::Type_t
operator+(const PVector<RComplexFloat,3,PColorVector>& l, 
	  const PVector<RComplexFloat,3,PColorVector>& r)
{
  BinaryReturn<PVector<RComplexFloat,3,PColorVector>, 
    PVector<RComplexFloat,3,PColorVector>, OpAdd>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "V+V" << endl;
#endif

  _inline_sse_add_su3_vector(l,r,d);

  return d;
}


// Optimized version of   DiracFermion <- ColorMatrix*DiracFermion
//    PSpinVector<PColorVector<RComplexFloat,3>,4> <- PScalar<PColorMatrix<RComplexFloat,3>> 
//                           * PSpinVector<PColorVector<RComplexFloat,3>,4>
template<>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PVector<PColorVector<RComplexFloat,3>,4,PSpinVector>, OpMultiply>::Type_t
operator*(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	  const PVector<PColorVector<RComplexFloat,3>,4,PSpinVector>& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PVector<PColorVector<RComplexFloat,3>,4,PSpinVector>, OpMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "PSc<M>*D" << endl;
#endif

  _inline_sse_mult_su3_mat_vec(l.elem(),r.elem(0),d.elem(0));
  _inline_sse_mult_su3_mat_vec(l.elem(),r.elem(1),d.elem(1));
  _inline_sse_mult_su3_mat_vec(l.elem(),r.elem(2),d.elem(2));
  _inline_sse_mult_su3_mat_vec(l.elem(),r.elem(3),d.elem(3));

  return d;
}


// Optimized version of   DiracFermion <- ColorMatrix*DiracFermion
//    PScalar<PSpinVector<PColorVector<RComplexFloat,3>,4>> <- PScalar<PScalar<PColorMatrix<RComplexFloat,3>>>
//                           * PScalar<PSpinVector<PColorVector<RComplexFloat,3>,4>>
template<>
inline BinaryReturn<PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, 
  PScalar<PSpinVector<PColorVector<RComplexFloat,3>,4> >, OpMultiply>::Type_t
operator*(const PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >& l, 
	  const PScalar<PSpinVector<PColorVector<RComplexFloat,3>,4> >& r)
{
  BinaryReturn<PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, 
    PScalar<PSpinVector<PColorVector<RComplexFloat,3>,4> >, OpMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "PSc<PSc<M>>*D" << endl;
#endif

  _inline_sse_mult_su3_mat_vec(l.elem().elem(),r.elem().elem(0),d.elem().elem(0));
  _inline_sse_mult_su3_mat_vec(l.elem().elem(),r.elem().elem(1),d.elem().elem(1));
  _inline_sse_mult_su3_mat_vec(l.elem().elem(),r.elem().elem(2),d.elem().elem(2));
  _inline_sse_mult_su3_mat_vec(l.elem().elem(),r.elem().elem(3),d.elem().elem(3));

  return d;
}


// Optimized version of   DiracFermion <- adj(ColorMatrix)*DiracFermion
//    PSpinVector<PColorVector<RComplexFloat,3>,4> <- adj(PScalar<PColorMatrix<RComplexFloat,3>>)
//                           * PSpinVector<PColorVector<RComplexFloat,3>,4>
template<>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PVector<PColorVector<RComplexFloat,3>,4,PSpinVector>, OpAdjMultiply>::Type_t
adjMultiply(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	    const PVector<PColorVector<RComplexFloat,3>,4,PSpinVector>& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PVector<PColorVector<RComplexFloat,3>,4,PSpinVector>, OpAdjMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "adj(PSc<M>)*D" << endl;
#endif

  _inline_sse_mult_adj_su3_mat_vec(l.elem(),r.elem(0),d.elem(0));
  _inline_sse_mult_adj_su3_mat_vec(l.elem(),r.elem(1),d.elem(1));
  _inline_sse_mult_adj_su3_mat_vec(l.elem(),r.elem(2),d.elem(2));
  _inline_sse_mult_adj_su3_mat_vec(l.elem(),r.elem(3),d.elem(3));

  return d;
}


// Optimized version of   DiracFermion <- adj(ColorMatrix)*DiracFermion
//    PScalar<PSpinVector<PColorVector<RComplexFloat,3>,4>> <- adj(PScalar<PScalar<PColorMatrix<RComplexFloat,3>>>)
//                           * PScalar<PSpinVector<PColorVector<RComplexFloat,3>,4>>
template<>
inline BinaryReturn<PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, 
  PScalar<PSpinVector<PColorVector<RComplexFloat,3>,4> >, OpAdjMultiply>::Type_t
adjMultiply(const PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >& l, 
	    const PScalar<PSpinVector<PColorVector<RComplexFloat,3>,4> >& r)
{
  BinaryReturn<PScalar<PScalar<PColorMatrix<RComplexFloat,3> > >, 
    PScalar<PSpinVector<PColorVector<RComplexFloat,3>,4> >, OpAdjMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "adj(PSc<PSc<M>>)*D" << endl;
#endif

  _inline_sse_mult_adj_su3_mat_vec(l.elem().elem(),r.elem().elem(0),d.elem().elem(0));
  _inline_sse_mult_adj_su3_mat_vec(l.elem().elem(),r.elem().elem(1),d.elem().elem(1));
  _inline_sse_mult_adj_su3_mat_vec(l.elem().elem(),r.elem().elem(2),d.elem().elem(2));
  _inline_sse_mult_adj_su3_mat_vec(l.elem().elem(),r.elem().elem(3),d.elem().elem(3));

  return d;
}



// Optimized version of  
//    PScalar<PColorVector<RComplexFloat,3>> <- PScalar<PColorVector<RComplexFloat,3>> 
//                                            + PScalar<PColorVector<RComplexFloat,3>>
template<>
inline BinaryReturn<PScalar<PColorVector<RComplexFloat,3> >, 
  PScalar<PColorVector<RComplexFloat,3> >, OpAdd>::Type_t
operator+(const PScalar<PColorVector<RComplexFloat,3> >& l, 
	  const PScalar<PColorVector<RComplexFloat,3> >& r)
{
  BinaryReturn<PScalar<PColorVector<RComplexFloat,3> >, 
    PScalar<PColorVector<RComplexFloat,3> >, OpAdd>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "PSc<V>+PSc<V>" << endl;
#endif

  _inline_sse_add_su3_vector(l.elem(),r.elem(),d.elem());

  return d;
}


#if 1
// Specialization to optimize the case   
//    LatticeColorMatrix = LatticeColorMatrix * LatticeColorMatrix
// NOTE: let this be a subroutine to save space
template<>
void evaluate(OLattice<PScalar<PScalar<PColorMatrix<RComplexFloat, 3> > > >& d, 
	      const OpAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	      Reference<QDPType<PScalar<PScalar<PColorMatrix<RComplexFloat, 3> > >, 
	      OLattice<PScalar<PScalar<PColorMatrix<RComplexFloat, 3> > > > > >, 
	      Reference<QDPType<PScalar<PScalar<PColorMatrix<RComplexFloat, 3> > >, 
	      OLattice<PScalar<PScalar<PColorMatrix<RComplexFloat, 3> > > > > > >,
	      OLattice<PScalar<PScalar<PColorMatrix<RComplexFloat, 3> > > > >& rhs,
	      const OrderedSubset& s);

#endif

#if 1
// Specialization to optimize the case   
//    LatticeColorMatrix = LatticeColorMatrix * LatticeColorMatrix
// NOTE: let this be a subroutine to save space
template<>
void evaluate(OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >& d, 
	      const OpAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	      Reference<QDPType<PScalar<PColorMatrix<RComplexFloat, 3> >, 
	      OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > > >, 
	      Reference<QDPType<PScalar<PColorMatrix<RComplexFloat, 3> >, 
	      OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > > > >,
	      OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > >& rhs,
	      const OrderedSubset& s);

#endif

#if 1
// Specialization to optimize the case   
//    LatticeColorMatrix = LatticeColorMatrix * LatticeColorMatrix
// NOTE: let this be a subroutine to save space
template<>
inline 
void evaluate(OLattice<PSpinVector<PColorVector<RComplexFloat, 3>, 2> >& d, 
	      const OpAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	      Reference<QDPType<PScalar<PColorMatrix<RComplexFloat, 3> >, 
	      OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > > >, 
	      Reference<QDPType<PSpinVector<PColorVector<RComplexFloat, 3>, 2>, 
	      OLattice<PSpinVector<PColorVector<RComplexFloat, 3>, 2> > > > >,
	      OLattice<PSpinVector<PColorVector<RComplexFloat, 3>, 2> > >& rhs,
	      const OrderedSubset& s)
{
#if defined(QDP_SCALARSITE_DEBUG)
  cout << "specialized QDP_H_M_times_H" << endl;
#endif

  typedef OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >       C;
  typedef OLattice<PSpinVector<PColorVector<RComplexFloat, 3>, 2> > H;

  const C& l = static_cast<const C&>(rhs.expression().left());
  const H& r = static_cast<const H&>(rhs.expression().right());

  for(int i=s.start(); i <= s.end(); ++i) 
  {
#if 0
    // This form appears significantly slower than below
    _inline_sse_mult_su3_mat_hwvec(l.elem(i),
				   r.elem(i),
				   d.elem(i));
#else
    _inline_sse_mult_su3_mat_vec(l.elem(i).elem(),
				 r.elem(i).elem(0),
				 d.elem(i).elem(0));
    _inline_sse_mult_su3_mat_vec(l.elem(i).elem(),
				 r.elem(i).elem(1),
				 d.elem(i).elem(1));
#endif
  }
}

#endif


//-------------------------------------------------------------------------
// Start of PDWVector optimizations
#if 1


// Use this def just to safe some typing later on in the file
//#define PVectorFloat  PDWVector<float,4>
//#define RComplexFloat  RComplex<ILattice<float,4> >


typedef float v4sf __attribute__ ((aligned (16),mode(V4SF)));


#if 0
// NOTE: the   operator+(v4sf,v4sf) first exists in gcc 3.3.X, not 3.2.X

// v4sf + v4sf
inline v4sf
operator+(v4sf l, v4sf r)
{
  v4sf tmp = __builtin_ia32_addps(l, r);
  return tmp;
}


// v4sf - v4sf
inline v4sf
operator-(v4sf l, v4sf r)
{
  return __builtin_ia32_subps(l, r);
}


// v4sf * v4sf
inline v4sf
operator*(v4sf l, v4sf r)
{
  return __builtin_ia32_mulps(l, r);
}


// v4sf / v4sf
inline v4sf
operator/(v4sf l, v4sf r)
{
  return __builtin_ia32_divps(l, r);
}
#endif





#if 1
//! Primitive domain-wall vector class
/*! 
 * Supports domain-wall manipulations
 */
template<> class PDWVector<float,4> : public PVector<float, 4, PDWVector>
{
public:
  typedef float  T;
  static const int N = 4;

  PDWVector() {}
  ~PDWVector() {}

  //---------------------------------------------------------
  //! construct dest = const
  PDWVector(const WordType<float>::Type_t& rhs)
    {
      for(int i=0; i < N; ++i)
	elem(i) = rhs;
    }

  //! construct dest = rhs
  template<class T1>
  PDWVector(const PDWVector<T1,N>& rhs)
    {
      for(int i=0; i < N; ++i)
	elem(i) = rhs.elem(i);
    }

  //! construct dest = rhs
  template<class T1>
  PDWVector(const T1& rhs)
    {
      for(int i=0; i < N; ++i)
	elem(i) = rhs;
    }


  //! construct dest = rhs
  PDWVector(const v4sf& rhs)
    {
      F.v = rhs;
    }


  //---------------------------------------------------------
  //! PDWVector = PScalar
  /*! Set equal to an PScalar */
  template<class T1>
  inline
  PDWVector& operator=(const PScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) = rhs.elem();

      return *this;
    }

  //! PDWVector += PScalar
  template<class T1>
  inline
  PDWVector& operator+=(const PScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) += rhs.elem();

      return *this;
    }

  //! PDWVector -= PScalar
  template<class T1>
  inline
  PDWVector& operator-=(const PScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) -= rhs.elem();

      return *this;
    }

  //! PDWVector *= PScalar
  template<class T1>
  inline
  PDWVector& operator*=(const PScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) *= rhs.elem();

      return *this;
    }

  //! PDWVector /= PScalar
  template<class T1>
  inline
  PDWVector& operator/=(const PScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) /= rhs.elem();

      return *this;
    }


  //---------------------------------------------------------
  //! PDWVector = PDWVector
  /*! Set equal to another PDWVector */
  inline
  PDWVector& operator=(const PDWVector& rhs) 
    {
      F.v = rhs.F.v;
      return *this;
    }

  //! PDWVector += PDWVector
  inline
  PDWVector& operator+=(const PDWVector& rhs) 
    {
      F.v = __builtin_ia32_addps(F.v, rhs.F.v);
      return *this;
    }

  //! PDWVector -= PDWVector
  inline
  PDWVector& operator-=(const PDWVector& rhs) 
    {
      F.v = __builtin_ia32_subps(F.v, rhs.F.v);
      return *this;
    }

  //! PDWVector *= PDWVector
  inline
  PDWVector& operator*=(const PDWVector& rhs) 
    {
      F.v = __builtin_ia32_mulps(F.v, rhs.F.v);
      return *this;
    }

  //! PDWVector /= PDWVector
  inline
  PDWVector& operator/=(const PDWVector& rhs) 
    {
      F.v = __builtin_ia32_divps(F.v, rhs.F.v);
      return *this;
    }


  //! Deep copy constructor
  PDWVector(const PDWVector& a)
    {
      // fprintf(stderr,"copy PDWVector\n");
      F.v = a.F.v;
    }


public:
  //! The backdoor
  /*! 
   * Used by optimization routines (e.g., SSE) that need the memory address of data.
   * BTW: to make this a friend would be a real pain since functions are templatized.
   */
  inline T* data() {return F.a;}


public:
  T& elem(int i) {return F.a[i];}
  const T& elem(int i) const {return F.a[i];}

  v4sf& elem_v() {return F.v;}
  const v4sf elem_v() const {return F.v;}

private:
  // SSE attributes
  union {
    v4sf v;
    T    a[4];
  } F  QDP_ALIGN16;

};
#endif




//--------------------------------------------------------------------------------------
// Optimized version of  
//    PDWVectorFloat4 <- PDWVectorFloat4 + PDWVectorFloat4
//template<>
inline PDWVectorFloat4
operator+(const PDWVectorFloat4& l, const PDWVectorFloat4& r)
{
#if defined(QDP_SCALARSITE_DEBUG)
  cout << "DWV+DWV" << endl;
#endif

  return __builtin_ia32_addps(l.elem_v(), r.elem_v());
}


// Optimized version of  
//    PDWVectorFloat4 <- PDWVectorFloat4 - PDWVectorFloat4
//template<>
inline PDWVectorFloat4
operator-(const PDWVectorFloat4& l, const PDWVectorFloat4& r)
{
#if defined(QDP_SCALARSITE_DEBUG)
  cout << "DWV-DWV" << endl;
#endif

  return __builtin_ia32_subps(l.elem_v(), r.elem_v());
}


// Optimized version of  
//    PDWVectorFloat4 <- PDWVectorFloat4 * PDWVectorFloat4
//template<>
inline PDWVectorFloat4
operator*(const PDWVectorFloat4& l, const PDWVectorFloat4& r)
{
#if defined(QDP_SCALARSITE_DEBUG)
  cout << "DWV * DWV" << endl;
#endif

  return __builtin_ia32_mulps(l.elem_v(), r.elem_v());
}

// Optimized version of  
//    PDWVectorFloat4 <- PScalar * PDWVectorFloat4
inline PDWVectorFloat4
operator*(const PScalar<float>& l, const PDWVectorFloat4& r)
{
#if defined(QDP_SCALARSITE_DEBUG)
  cout << "P * DWV" << endl;
#endif

  register v4sf x = __builtin_ia32_loadss((float*)&(l.elem()));

  asm("shufps\t$0,%0,%0"
      : "+x" (x));

  return __builtin_ia32_mulps(x, r.elem_v());
}

// Optimized version of  
//    PDWVectorFloat4 <- PDWVectorFloat4 * PScalar
inline PDWVectorFloat4
operator*(const PDWVectorFloat4& l, const PScalar<float>& r)
{
#if defined(QDP_SCALARSITE_DEBUG)
  cout << "DWV * P" << endl;
#endif

  register v4sf x = __builtin_ia32_loadss((float*)&(r.elem()));

  asm("shufps\t$0,%0,%0"
      : "+x" (x));

  return __builtin_ia32_mulps(l.elem_v(), x);
}



// Optimized version of  
//    PDWVectorFloat4 <- PDWVectorFloat4 / PDWVectorFloat4
//template<>
inline PDWVectorFloat4
operator/(const PDWVectorFloat4& l, const PDWVectorFloat4& r)
{
#if defined(QDP_SCALARSITE_DEBUG)
  cout << "DWV / DWV" << endl;
#endif

  return __builtin_ia32_mulps(l.elem_v(), r.elem_v());
}


//--------------------------------------------------------------------------------------
// Optimized version of  
//    RComplexFloat4 <- RComplexFloat4 + RComplexFloat4
template<>
inline RComplexFloat4
operator+(const RComplexFloat4& l, const RComplexFloat4& r)
{
#if defined(QDP_SCALARSITE_DEBUG)
  cout << "C<DWV> + C<DWV>" << endl;
#endif

  return RComplexFloat4(__builtin_ia32_addps(l.real().elem_v(), r.real().elem_v()),
			__builtin_ia32_addps(l.imag().elem_v(), r.imag().elem_v()));
}


// Optimized version of  
//    RComplexFloat4 <- RComplexFloat4 - RComplexFloat4
template<>
inline RComplexFloat4
operator-(const RComplexFloat4& l, const RComplexFloat4& r)
{
#if defined(QDP_SCALARSITE_DEBUG)
  cout << "C<DWV> - C<DWV" << endl;
#endif

  return RComplexFloat4(__builtin_ia32_subps(l.real().elem_v(), r.real().elem_v()),
			__builtin_ia32_subps(l.imag().elem_v(), r.imag().elem_v()));
}


// Optimized version of  
//    RComplexFloat4 <- RComplexFloat4 * RComplexFloat4
template<>
inline RComplexFloat4
operator*(const RComplexFloat4& l, const RComplexFloat4& r)
{
  RComplexFloat4 d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "C<DWV> * C<DWV>" << endl;
#endif

  v4sf tmp1 = __builtin_ia32_mulps(l.real().elem_v(), r.real().elem_v());
  v4sf tmp2 = __builtin_ia32_mulps(l.imag().elem_v(), r.imag().elem_v());
  d.real().elem_v() = __builtin_ia32_subps(tmp1, tmp2);

  v4sf tmp3 = __builtin_ia32_mulps(l.real().elem_v(), r.imag().elem_v());
  v4sf tmp4 = __builtin_ia32_mulps(l.imag().elem_v(), r.real().elem_v());
  d.imag().elem_v() = __builtin_ia32_addps(tmp3, tmp4);

  return d;
}

// Optimized version of  
//    RComplexFloat4 <- adj(RComplexFloat4) * RComplexFloat4
template<>
inline BinaryReturn<RComplexFloat4, RComplexFloat4, OpAdjMultiply>::Type_t
adjMultiply(const RComplexFloat4& l, const RComplexFloat4& r)
{
#if defined(QDP_SCALARSITE_DEBUG)
  cout << "adj(C<DWV>) * C<DWV>" << endl;
#endif

  BinaryReturn<RComplexFloat4, RComplexFloat4, OpAdjMultiply>::Type_t  d;

  v4sf tmp1 = __builtin_ia32_mulps(l.real().elem_v(), r.real().elem_v());
  v4sf tmp2 = __builtin_ia32_mulps(l.imag().elem_v(), r.imag().elem_v());
  d.real().elem_v() = __builtin_ia32_addps(tmp1, tmp2);

  v4sf tmp3 = __builtin_ia32_mulps(l.real().elem_v(), r.imag().elem_v());
  v4sf tmp4 = __builtin_ia32_mulps(l.imag().elem_v(), r.real().elem_v());
  d.imag().elem_v() = __builtin_ia32_subps(tmp3, tmp4);

  return d;
}

// Optimized  RComplex*adj(RComplex)
template<>
inline BinaryReturn<RComplexFloat4, RComplexFloat4, OpMultiplyAdj>::Type_t
multiplyAdj(const RComplexFloat4& l, const RComplexFloat4& r)
{
  BinaryReturn<RComplexFloat4, RComplexFloat4, OpMultiplyAdj>::Type_t  d;

  v4sf tmp1 = __builtin_ia32_mulps(l.real().elem_v(), r.real().elem_v());
  v4sf tmp2 = __builtin_ia32_mulps(l.imag().elem_v(), r.imag().elem_v());
  d.real().elem_v() = __builtin_ia32_addps(tmp1, tmp2);

  v4sf tmp3 = __builtin_ia32_mulps(l.imag().elem_v(), r.real().elem_v());
  v4sf tmp4 = __builtin_ia32_mulps(l.real().elem_v(), r.imag().elem_v());
  d.imag().elem_v() = __builtin_ia32_subps(tmp3, tmp4);

  return d;
}

// Optimized  adj(RComplex)*adj(RComplex)
template<>
inline BinaryReturn<RComplexFloat4, RComplexFloat4, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(const RComplexFloat4& l, const RComplexFloat4& r)
{
  BinaryReturn<RComplexFloat4, RComplexFloat4, OpAdjMultiplyAdj>::Type_t  d;

  typedef struct
  {
    unsigned int c[4];
  } sse_mask __attribute__ ((aligned (16)));
  
  static sse_mask _sse_sgn __attribute__ ((unused)) ={0x80000000, 0x80000000, 0x80000000, 0x80000000};

  v4sf tmp1 = __builtin_ia32_mulps(l.real().elem_v(), r.real().elem_v());
  v4sf tmp2 = __builtin_ia32_mulps(l.imag().elem_v(), r.imag().elem_v());
  d.real().elem_v() = __builtin_ia32_subps(tmp1, tmp2);

  v4sf tmp3 = __builtin_ia32_mulps(l.real().elem_v(), r.imag().elem_v());
  v4sf tmp4 = __builtin_ia32_mulps(l.imag().elem_v(), r.real().elem_v());
  v4sf tmp5 = __builtin_ia32_addps(tmp3, tmp4);
//  d.imag().elem_v() = __builtin_ia32_xorps(tmp5, _sse_sgn.v);
  v4sf tmp6 = __builtin_ia32_loadaps((float*)&_sse_sgn);
  d.imag().elem_v() = __builtin_ia32_xorps(tmp5, tmp6);

  return d;
}

#endif


/*! @} */   // end of group optimizations

#if defined(QDP_SCALARSITE_DEBUG)
#undef QDP_SCALARSITE_DEBUG
#endif

QDP_END_NAMESPACE();

#endif  // defined(__GNUC__)

#endif
