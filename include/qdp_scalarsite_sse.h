// -*- C++ -*-
// $Id: qdp_scalarsite_sse.h,v 1.7 2003-08-12 04:38:45 edwards Exp $

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
#define RComplexFloat  RComplex<float>


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

// Optimized version of  
//    PColorMatrix<RComplexFloat,3> <- PColorMatrix<RComplexFloat,3> * PColorMatrix<RComplexFloat,3>
template<>
inline BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
  PMatrix<RComplexFloat,3,PColorMatrix>, OpMultiply>::Type_t
operator*<>(const PMatrix<RComplexFloat,3,PColorMatrix>& l, 
	    const PMatrix<RComplexFloat,3,PColorMatrix>& r)
{
  BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
    PMatrix<RComplexFloat,3,PColorMatrix>, OpMultiply>::Type_t  d;

  _inline_sse_mult_su3_nn(l,r,d);

  return d;
}


// Optimized version of  
//   PColorMatrix<RComplexFloat,3> <- adj(PColorMatrix<RComplexFloat,3>) * PColorMatrix<RComplexFloat,3>
template<>
inline BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
  PMatrix<RComplexFloat,3,PColorMatrix>, OpAdjMultiply>::Type_t
adjMultiply<>(const PMatrix<RComplexFloat,3,PColorMatrix>& l, 
	      const PMatrix<RComplexFloat,3,PColorMatrix>& r)
{
  BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
    PMatrix<RComplexFloat,3,PColorMatrix>, OpAdjMultiply>::Type_t  d;

  _inline_sse_mult_su3_an(l,r,d);

  return d;
}


// Optimized version of  
//   PColorMatrix<RComplexFloat,3> <- PColorMatrix<RComplexFloat,3> * adj(PColorMatrix<RComplexFloat,3>)
template<>
inline BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
  PMatrix<RComplexFloat,3,PColorMatrix>, OpMultiplyAdj>::Type_t
multiplyAdj<>(const PMatrix<RComplexFloat,3,PColorMatrix>& l, 
	      const PMatrix<RComplexFloat,3,PColorMatrix>& r)
{
  BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
    PMatrix<RComplexFloat,3,PColorMatrix>, OpMultiplyAdj>::Type_t  d;

  _inline_sse_mult_su3_na(l,r,d);

  return d;
}


#if 0
// Ooops, this macro does not exist!!

// Optimized version of  
//   PColorMatrix<RComplexFloat,3> <- adj(PColorMatrix<RComplexFloat,3>) * adj(PColorMatrix<RComplexFloat,3>)
template<>
inline BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
  PMatrix<RComplexFloat,3,PColorMatrix>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj<>(const PMatrix<RComplexFloat,3,PColorMatrix>& l, 
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
operator*<>(const PMatrix<RComplexFloat,3,PColorMatrix>& l, 
	    const PVector<RComplexFloat,3,PColorVector>& r)
{
  BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
    PVector<RComplexFloat,3,PColorVector>, OpMultiply>::Type_t  d;

  _inline_sse_mult_su3_mat_vec(l,r,d);

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

  _inline_sse_mult_adj_su3_mat_vec(l,r,d);

  return d;
}


// Optimized version of    HalfFermion <- ColorMatrix*HalfFermion
//    PSpinVector<PColorVector<RComplexFloat,3>,2> <- PScalar<PColorMatrix<RComplexFloat,3>> * 
//                     PSpinVector<ColorVector<RComplexFloat,3>,2>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PSpinVector<PColorVector<RComplexFloat,3>,2>, OpMultiply>::Type_t
operator*(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
          const PSpinVector<PColorVector<RComplexFloat,3>,2>& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PSpinVector<PColorVector<RComplexFloat,3>,2>, OpMultiply>::Type_t  d;

  _inline_sse_mult_su3_mat_hwvec(l,r,d);

  return d;
}


// Optimized version of    HalfFermion <- ColorMatrix*HalfFermion
//    PSpinVector<PColorVector<RComplexFloat,3>,2> <- adj(PScalar<PColorMatrix<RComplexFloat,3>>) * 
//                     PSpinVector<ColorVector<RComplexFloat,3>,2>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PSpinVector<PColorVector<RComplexFloat,3>,2>, OpAdjMultiply>::Type_t
adjMultiply(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
            const PSpinVector<PColorVector<RComplexFloat,3>,2>& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PSpinVector<PColorVector<RComplexFloat,3>,2>, OpAdjMultiply>::Type_t  d;

  _inline_sse_mult_adj_su3_mat_hwvec(l,r,d);

  return d;
}


// Optimized version of  
//    PColorVector<RComplexFloat,3> <- PColorVector<RComplexFloat,3> + PColorVector<RComplexFloat,3>
template<>
inline BinaryReturn<PVector<RComplexFloat,3,PColorVector>, 
  PVector<RComplexFloat,3,PColorVector>, OpAdd>::Type_t
operator+<>(const PVector<RComplexFloat,3,PColorVector>& l, 
	    const PVector<RComplexFloat,3,PColorVector>& r)
{
  BinaryReturn<PVector<RComplexFloat,3,PColorVector>, 
    PVector<RComplexFloat,3,PColorVector>, OpAdd>::Type_t  d;

  _inline_sse_add_su3_vector(l,r,d);

  return d;
}


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


/*! @} */   // end of group optimizations

QDP_END_NAMESPACE();

#endif  // defined(__GNUC__)

#endif
