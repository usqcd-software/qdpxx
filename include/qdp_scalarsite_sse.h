// -*- C++ -*-
// $Id: qdp_scalarsite_sse.h,v 1.8 2003-08-23 21:10:07 edwards Exp $

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
inline BinaryReturn<PColorMatrix<RComplexFloat,3>, 
  PColorMatrix<RComplexFloat,3>, OpMultiply>::Type_t
operator*(const PColorMatrix<RComplexFloat,3>& l, 
	  const PColorMatrix<RComplexFloat,3>& r)
{
  BinaryReturn<PColorMatrix<RComplexFloat,3>, 
    PColorMatrix<RComplexFloat,3>, OpMultiply>::Type_t  d;

//  cout << "M*M" << endl;

  _inline_sse_mult_su3_nn(l,r,d);

  return d;
}


// Optimized version of  
//    PScalar<PColorMatrix<RComplexFloat,3>> <- PScalar<PColorMatrix<RComplexFloat,3>> * 
//                         PScalar<PColorMatrix<RComplexFloat,3>>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >,
  PScalar<PColorMatrix<RComplexFloat,3> >, OpMultiply>::Type_t
operator*(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	  const PScalar<PColorMatrix<RComplexFloat,3> >& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PScalar<PColorMatrix<RComplexFloat,3> >, OpMultiply>::Type_t  d;

//  cout << "PSc<M>*PSc<M>" << endl;

  _inline_sse_mult_su3_nn(l.elem(),r.elem(),d.elem());

  return d;
}


// Optimized version of  
//   PColorMatrix<RComplexFloat,3> <- adj(PColorMatrix<RComplexFloat,3>) * PColorMatrix<RComplexFloat,3>
inline BinaryReturn<PColorMatrix<RComplexFloat,3>, 
  PColorMatrix<RComplexFloat,3>, OpAdjMultiply>::Type_t
adjMultiply(const PColorMatrix<RComplexFloat,3>& l, 
	    const PColorMatrix<RComplexFloat,3>& r)
{
  BinaryReturn<PColorMatrix<RComplexFloat,3>, 
    PColorMatrix<RComplexFloat,3>, OpAdjMultiply>::Type_t  d;

//  cout << "adj(M)*M" << endl;

  _inline_sse_mult_su3_an(l,r,d);

  return d;
}


// Optimized version of  
//   PScalar<PColorMatrix<RComplexFloat,3>> <- adj(PScalar<PColorMatrix<RComplexFloat,3>>) * PScalar<PColorMatrix<RComplexFloat,3>>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PScalar<PColorMatrix<RComplexFloat,3> >, OpAdjMultiply>::Type_t
adjMultiply(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	    const PScalar<PColorMatrix<RComplexFloat,3> >& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PScalar<PColorMatrix<RComplexFloat,3> >, OpAdjMultiply>::Type_t  d;

//  cout << "adj(PSc<M>)*PSc<M>" << endl;

  _inline_sse_mult_su3_an(l.elem(),r.elem(),d.elem());

  return d;
}


// Optimized version of  
//   PColorMatrix<RComplexFloat,3> <- PColorMatrix<RComplexFloat,3> * adj(PColorMatrix<RComplexFloat,3>)
inline BinaryReturn<PColorMatrix<RComplexFloat,3>, 
  PColorMatrix<RComplexFloat,3>, OpMultiplyAdj>::Type_t
multiplyAdj(const PColorMatrix<RComplexFloat,3>& l, 
	    const PColorMatrix<RComplexFloat,3>& r)
{
  BinaryReturn<PColorMatrix<RComplexFloat,3>, 
    PColorMatrix<RComplexFloat,3>, OpMultiplyAdj>::Type_t  d;

//  cout << "M*adj(M)" << endl;

  _inline_sse_mult_su3_na(l,r,d);

  return d;
}


// Optimized version of  
//   PColorMatrix<RComplexFloat,3> <- PColorMatrix<RComplexFloat,3> * adj(PColorMatrix<RComplexFloat,3>)
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PScalar<PColorMatrix<RComplexFloat,3> >, OpMultiplyAdj>::Type_t
multiplyAdj(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	    const PScalar<PColorMatrix<RComplexFloat,3> >& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PScalar<PColorMatrix<RComplexFloat,3> >, OpMultiplyAdj>::Type_t  d;

//  cout << "PS<M>*adj(PSc<M>)" << endl;

  _inline_sse_mult_su3_na(l.elem(),r.elem(),d.elem());

  return d;
}


#if 0
// Ooops, this macro does not exist!!

// Optimized version of  
//   PColorMatrix<RComplexFloat,3> <- adj(PColorMatrix<RComplexFloat,3>) * adj(PColorMatrix<RComplexFloat,3>)
inline BinaryReturn<PColorMatrix<RComplexFloat,3>, 
  PColorMatrix<RComplexFloat,3>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(const PColorMatrix<RComplexFloat,3>& l, 
	       const PColorMatrix<RComplexFloat,3>& r)
{
  BinaryReturn<PColorMatrix<RComplexFloat,3>, 
    PColorMatrix<RComplexFloat,3>, OpAdjMultiplyAdj>::Type_t  d;

  _inline_sse_mult_su3_aa(l,r,d);

  return d;
}
#endif


// Optimized version of  
//    PColorVector<RComplexFloat,3> <- PColorMatrix<RComplexFloat,3> * PColorVector<RComplexFloat,3>
inline BinaryReturn<PColorMatrix<RComplexFloat,3>, 
  PColorVector<RComplexFloat,3>, OpMultiply>::Type_t
operator*(const PColorMatrix<RComplexFloat,3>& l, 
	  const PColorVector<RComplexFloat,3>& r)
{
  BinaryReturn<PColorMatrix<RComplexFloat,3>, 
    PColorVector<RComplexFloat,3>, OpMultiply>::Type_t  d;

//  cout << "M*V" << endl;

  _inline_sse_mult_su3_mat_vec(l,r,d);

  return d;
}


// Optimized version of  
//    PScalar<PColorVector<RComplexFloat,3>> <- PScalar<PColorMatrix<RComplexFloat,3>> * PScalar<PColorVector<RComplexFloat,3>>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PScalar<PColorVector<RComplexFloat,3> >, OpMultiply>::Type_t
operator*(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	  const PScalar<PColorVector<RComplexFloat,3> >& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PScalar<PColorVector<RComplexFloat,3> >, OpMultiply>::Type_t  d;

//  cout << "PSc<M>*PSc<V>" << endl;

  _inline_sse_mult_su3_mat_vec(l.elem(),r.elem(),d.elem());

  return d;
}


// Optimized version of  
//    PColorVector<RComplexFloat,3> <- adj(PColorMatrix<RComplexFloat,3>) * PColorVector<RComplexFloat,3>
inline BinaryReturn<PColorMatrix<RComplexFloat,3>, 
  PColorVector<RComplexFloat,3>, OpAdjMultiply>::Type_t
adjMultiply(const PColorMatrix<RComplexFloat,3>& l, 
	    const PColorVector<RComplexFloat,3>& r)
{
  BinaryReturn<PColorMatrix<RComplexFloat,3>, 
    PColorVector<RComplexFloat,3>, OpAdjMultiply>::Type_t  d;

//  cout << "adj(M)*V" << endl;

  _inline_sse_mult_adj_su3_mat_vec(l,r,d);

  return d;
}


// Optimized version of   StaggeredFermion <- ColorMatrix*StaggeredFermion
//    PSpinVector<PColorVector<RComplexFloat,3>,1> <- PScalar<PColorMatrix<RComplexFloat,3>> * PSpinVector<PColorVector<RComplexFloat,3>,1>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PSpinVector<PColorVector<RComplexFloat,3>,1>, OpMultiply>::Type_t
operator*(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	  const PSpinVector<PColorVector<RComplexFloat,3>,1>& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PSpinVector<PColorVector<RComplexFloat,3>,1>, OpMultiply>::Type_t  d;

//  cout << "M*S" << endl;

  _inline_sse_mult_su3_mat_vec(l.elem(),r.elem(0),d.elem(0));

  return d;
}


// Optimized version of  
//    PScalar<PColorVector<RComplexFloat,3>> <- adj(PScalar<PColorMatrix<RComplexFloat,3>>) * PScalar<PColorVector<RComplexFloat,3>>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PScalar<PColorVector<RComplexFloat,3> >, OpAdjMultiply>::Type_t
adjMultiply(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	    const PScalar<PColorVector<RComplexFloat,3> >& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PScalar<PColorVector<RComplexFloat,3> >, OpAdjMultiply>::Type_t  d;

//  cout << "adj(PSc<M>)*PSc<V>" << endl;

  _inline_sse_mult_adj_su3_mat_vec(l.elem(),r.elem(),d.elem());

  return d;
}


// Optimized version of   StaggeredFermion <- adj(ColorMatrix)*StaggeredFermion
//    PSpinVector<PColorVector<RComplexFloat,3>,1> <- adj(PScalar<PColorMatrix<RComplexFloat,3>>) * PSpinVector<PColorVector<RComplexFloat,3>,1>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PSpinVector<PColorVector<RComplexFloat,3>,1>, OpAdjMultiply>::Type_t
adjMultiply(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	    const PSpinVector<PColorVector<RComplexFloat,3>,1>& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PSpinVector<PColorVector<RComplexFloat,3>,1>, OpAdjMultiply>::Type_t  d;

//  cout << "adj(S)*S" << endl;

  _inline_sse_mult_adj_su3_mat_vec(l.elem(),r.elem(0),d.elem(0));

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

//  cout << "M*H" << endl;

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

//  cout << "adj(M)*H" << endl;

  _inline_sse_mult_adj_su3_mat_hwvec(l,r,d);

  return d;
}


// Optimized version of  
//    PColorVector<RComplexFloat,3> <- PColorVector<RComplexFloat,3> + PColorVector<RComplexFloat,3>
inline BinaryReturn<PColorVector<RComplexFloat,3>, 
  PColorVector<RComplexFloat,3>, OpAdd>::Type_t
operator+(const PColorVector<RComplexFloat,3>& l, 
	  const PColorVector<RComplexFloat,3>& r)
{
  BinaryReturn<PColorVector<RComplexFloat,3>, 
    PColorVector<RComplexFloat,3>, OpAdd>::Type_t  d;

//  cout << "V+V" << endl;

  _inline_sse_add_su3_vector(l,r,d);

  return d;
}


// Optimized version of   DiracFermion <- ColorMatrix*DiracFermion
//    PSpinVector<PColorVector<RComplexFloat,3>,4> <- PScalar<PColorMatrix<RComplexFloat,3>> 
//                           * PSpinVector<PColorVector<RComplexFloat,3>,4>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PSpinVector<PColorVector<RComplexFloat,3>,4>, OpMultiply>::Type_t
operator*(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	  const PSpinVector<PColorVector<RComplexFloat,3>,4>& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PSpinVector<PColorVector<RComplexFloat,3>,4>, OpMultiply>::Type_t  d;

//  cout << "M*D" << endl;

  _inline_sse_mult_su3_mat_vec(l.elem(),r.elem(0),d.elem(0));
  _inline_sse_mult_su3_mat_vec(l.elem(),r.elem(1),d.elem(1));
  _inline_sse_mult_su3_mat_vec(l.elem(),r.elem(2),d.elem(2));
  _inline_sse_mult_su3_mat_vec(l.elem(),r.elem(3),d.elem(3));

  return d;
}


// Optimized version of   DiracFermion <- adj(ColorMatrix)*DiracFermion
//    PSpinVector<PColorVector<RComplexFloat,3>,4> <- adj(PScalar<PColorMatrix<RComplexFloat,3>>)
//                           * PSpinVector<PColorVector<RComplexFloat,3>,4>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PSpinVector<PColorVector<RComplexFloat,3>,4>, OpAdjMultiply>::Type_t
adjMultiply(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	    const PSpinVector<PColorVector<RComplexFloat,3>,4>& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PSpinVector<PColorVector<RComplexFloat,3>,4>, OpAdjMultiply>::Type_t  d;

//  cout << "adj(M)*D" << endl;

  _inline_sse_mult_adj_su3_mat_vec(l.elem(),r.elem(0),d.elem(0));
  _inline_sse_mult_adj_su3_mat_vec(l.elem(),r.elem(1),d.elem(1));
  _inline_sse_mult_adj_su3_mat_vec(l.elem(),r.elem(2),d.elem(2));
  _inline_sse_mult_adj_su3_mat_vec(l.elem(),r.elem(3),d.elem(3));

  return d;
}



// Optimized version of  
//    PScalar<PColorVector<RComplexFloat,3>> <- PScalar<PColorVector<RComplexFloat,3>> 
//                                            + PScalar<PColorVector<RComplexFloat,3>>
inline BinaryReturn<PScalar<PColorVector<RComplexFloat,3> >, 
  PScalar<PColorVector<RComplexFloat,3> >, OpAdd>::Type_t
operator+(const PScalar<PColorVector<RComplexFloat,3> >& l, 
	  const PScalar<PColorVector<RComplexFloat,3> >& r)
{
  BinaryReturn<PScalar<PColorVector<RComplexFloat,3> >, 
    PScalar<PColorVector<RComplexFloat,3> >, OpAdd>::Type_t  d;

//  cout << "PSc<V>+PSc<V>" << endl;

  _inline_sse_add_su3_vector(l.elem(),r.elem(),d.elem());

  return d;
}


#if 0
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

/*! @} */   // end of group optimizations

QDP_END_NAMESPACE();

#endif  // defined(__GNUC__)

#endif
