// $Id: qdp_scalarsite_sse_blas.h,v 1.11 2006-09-26 15:16:31 bjoo Exp $
/*! @file
 * @brief Blas optimizations
 * 
 * Generic and maybe SSE optimizations of basic operations
 */

#ifndef QDP_SCALARSITE_SSE_BLAS_H
#define QDP_SCALARSITE_SSE_BLAS_H

#include "scalarsite_generic/generic_blas_local_vcdot.h"
#include "scalarsite_generic/generic_blas_local_vcdot_real.h"

QDP_BEGIN_NAMESPACE(QDP);

// Forward declarations of BLAS routines
void vaxpy3(REAL32 *Out, REAL32 *scalep,REAL32 *InScale, REAL32 *Add,int n_3vec);
void vaxmy3(REAL32 *Out, REAL32 *scalep,REAL32 *InScale, REAL32 *Sub,int n_3vec);
void vadd(REAL32 *Out, REAL32 *In1, REAL32 *In2, int n_3vec);
void vsub(REAL32 *Out, REAL32 *In1, REAL32 *In2, int n_3vec);
void vscal(REAL32 *Out, REAL32 *scalep, REAL32 *In, int n_3vec);
void local_sumsq(REAL64 *Out, REAL32 *In, int n_3vec);
void axpbyz(REAL32 *Out,REAL32 *scalep,REAL32 *InScale, REAL32 *scalep2, REAL32 *Add,int n_4vec);
void axmbyz(REAL32 *Out,REAL32 *scalep,REAL32 *InScale, REAL32 *scalep2, REAL32 *Add,int n_4vec);

typedef PSpinVector<PColorVector<RComplex<REAL32>, 3>, 4> TVec;
typedef PScalar<PScalar<RScalar<REAL32> > >  TScal;

/* #define DEBUG_BLAS_VAXMBY */
/* #define DEBUG_BLAS_VAXPBY */

#define QDP_SCALARSITE_USE_EVALUATE


// TVec is the LatticeFermion from qdp_dwdefs.h with the OLattice<> stripped
// from around it

// TScalar is the usual Real, with the OScalar<> stripped from it
//
// THis is simply to make the code more readable, and reduces < < s and > >s
// in the template arguments


#if defined(QDP_SCALARSITE_USE_EVALUATE)

// d += Scalar*Vec
template<>
inline
void evaluate(OLattice< TVec >& d, 
	      const OpAddAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	      Reference< QDPType< TScal, OScalar < TScal > > >,
	      Reference< QDPType< TVec, OLattice< TVec > > > >,
	      OLattice< TVec > > &rhs,
	      const OrderedSubset& s)
{

#ifdef DEBUG_BLAS
  QDPIO::cout << "SSE: y += a*x" << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec > &>(rhs.expression().right());
  const OScalar< TScal >& a = static_cast<const OScalar< TScal > &> (rhs.expression().left());
  
  REAL32 ar = a.elem().elem().elem().elem();
  REAL32* aptr = &ar;
  REAL32* xptr = (REAL32 *)&(x.elem(s.start()).elem(0).elem(0).real());
  REAL32* yptr = &(d.elem(s.start()).elem(0).elem(0).real());
  // cout << "Specialised axpy a ="<< ar << endl;
  
  int n_3vec = (s.end()-s.start()+1)*24;
  vaxpy3(yptr, aptr, xptr, yptr, n_3vec);
}

// d -= Scalar*Vec
template<>
inline
void evaluate(OLattice< TVec >& d, 
	      const OpSubtractAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	      Reference< QDPType< TScal, OScalar < TScal > > >,
	      Reference< QDPType< TVec, OLattice< TVec > > > >,
	      OLattice< TVec > > &rhs,
	      const OrderedSubset& s)
{

#ifdef DEBUG_BLAS
  QDPIO::cout << "SSE: y -= a*x" << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec > &>(rhs.expression().right());
  const OScalar< TScal >& a = static_cast<const OScalar< TScal > &> (rhs.expression().left());

  // - sign as y -= ax <=> y = y-ax = -ax + y = axpy with -a 
  REAL32 ar = -( a.elem().elem().elem().elem());
  REAL32* aptr = &ar;
  REAL32* xptr = (REAL32 *)&(x.elem(s.start()).elem(0).elem(0).real());
  REAL32* yptr = &(d.elem(s.start()).elem(0).elem(0).real());
  
  int n_3vec = (s.end()-s.start()+1)*24;
  vaxpy3(yptr, aptr, xptr, yptr, n_3vec);
	
}

// z = ax + y
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpAdd,
	       BinaryNode<OpMultiply, 
	       Reference< QDPType< TScal, OScalar< TScal > > >,
	       Reference< QDPType< TVec, OLattice< TVec > > > >,
	       Reference< QDPType< TVec, OLattice< TVec > > > >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{

#ifdef DEBUG_BLAS
  QDPIO::cout << "SSE: z = a*x + y" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&> (rhs.expression().right());

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN;

  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().left());

  // get a and x out of the bynary node
  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode.left());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode.right());
  // Set pointers 
  REAL32 ar =  a.elem().elem().elem().elem();
  REAL32 *aptr = (REAL32 *)&ar;
  REAL32 *xptr = (REAL32 *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL32 *yptr = (REAL32 *) &(y.elem(s.start()).elem(0).elem(0).real());
  REAL32* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*24;
  vaxpy3(zptr, aptr, xptr, yptr, n_3vec);
}

// Vec = Vec + Scal*Vec
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpAdd,
	       Reference< QDPType< TVec, OLattice< TVec > > >,
	       BinaryNode<OpMultiply, 
	       Reference< QDPType< TScal, OScalar< TScal > > >,
	       Reference< QDPType< TVec, OLattice< TVec > > > > >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "SSE: z = y + a*x" << endl;
#endif


  // Peel the stuff out of the expression

  // y is the left side of rhs
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&> (rhs.expression().left());

  // ax is the right side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN;

  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().right());

  // get a and x out of the bynary node
  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode.left());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode.right());
  // Set pointers 
  REAL32 ar =  a.elem().elem().elem().elem();
  REAL32 *aptr = (REAL32 *)&ar;
  REAL32 *xptr = (REAL32 *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL32 *yptr = (REAL32 *) &(y.elem(s.start()).elem(0).elem(0).real());
  REAL32* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());

  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*24;
  vaxpy3(zptr, aptr, xptr, yptr, n_3vec);

}

// Vec = Scalar*Vec - Vec
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpSubtract,
	       BinaryNode<OpMultiply, 
	       Reference< QDPType< TScal, OScalar< TScal > > >,
	       Reference< QDPType< TVec, OLattice< TVec > > > >,
	       Reference< QDPType< TVec, OLattice< TVec > > > >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "SSE: z = a*x - y" << endl;
#endif


  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&> (rhs.expression().right());

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN;

  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().left());

  // get a and x out of the bynary node
  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode.left());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode.right());
  // Set pointers 
  REAL32 ar =  a.elem().elem().elem().elem();
  REAL32 *aptr = (REAL32 *)&ar;
  REAL32 *xptr = (REAL32 *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL32 *yptr = (REAL32 *) &(y.elem(s.start()).elem(0).elem(0).real());
  REAL32* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*24;
  vaxmy3(zptr, aptr, xptr, yptr, n_3vec);
}

template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpSubtract,
	       Reference< QDPType< TVec, OLattice< TVec > > >,
	       BinaryNode<OpMultiply, 
	       Reference< QDPType< TScal, OScalar< TScal > > >,
	       Reference< QDPType< TVec, OLattice< TVec > > > > >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "SSE: z = y - a*x" << endl;
#endif

  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&> (rhs.expression().left());

  // ax is the right side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN;

  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().right());

  // get a and x out of the bynary node
  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode.left());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode.right());
  // Set pointers etc.

  // -ve sign as y - ax = -ax + y  = axpy with -a.
  REAL32 ar =  -a.elem().elem().elem().elem();
  REAL32 *aptr = (REAL32 *)&ar;
  REAL32 *xptr = (REAL32 *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL32 *yptr = (REAL32 *) &(y.elem(s.start()).elem(0).elem(0).real());
  REAL32* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*24;
  vaxpy3(zptr, aptr, xptr, yptr, n_3vec);
}

// Vec += Vec * Scalar (AXPY)
template<>
inline
void evaluate(OLattice< TVec >& d, 
	      const OpAddAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	      Reference< QDPType< TVec, OLattice< TVec > > >,
	      Reference< QDPType< TScal, OScalar < TScal > > > >,
	      OLattice< TVec > > &rhs,
	      const OrderedSubset& s)
{

#ifdef DEBUG_BLAS
  QDPIO::cout << "SSE: y += x*a" << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec > &>(rhs.expression().left());
  const OScalar< TScal >& a = static_cast<const OScalar< TScal > &> (rhs.expression().right());
  
  REAL32 ar = a.elem().elem().elem().elem();
  REAL32* aptr = &ar;
  REAL32* xptr = (REAL32 *)&(x.elem(s.start()).elem(0).elem(0).real());
  REAL32* yptr = &(d.elem(s.start()).elem(0).elem(0).real());
  // cout << "Specialised axpy a ="<< ar << endl;
  
  int n_3vec = (s.end()-s.start()+1)*24;
  vaxpy3(yptr, aptr, xptr, yptr, n_3vec);
}


// Vec -= Vec *Scalar 
template<>
inline
void evaluate(OLattice< TVec >& d, 
	      const OpSubtractAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	      Reference< QDPType< TVec, OLattice< TVec > > >,
	      Reference< QDPType< TScal, OScalar < TScal > > > >,
	      OLattice< TVec > > &rhs,
	      const OrderedSubset& s)
{

#ifdef DEBUG_BLAS
  QDPIO::cout << "SSE: y -= x*a" << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec > &>(rhs.expression().left());
  const OScalar< TScal >& a = static_cast<const OScalar< TScal > &> (rhs.expression().right());

  // - sign as y -= ax <=> y = y-ax = -ax + y = axpy with -a 
  REAL32 ar = -( a.elem().elem().elem().elem());
  REAL32* aptr = &ar;
  REAL32* xptr = (REAL32 *)&(x.elem(s.start()).elem(0).elem(0).real());
  REAL32* yptr = &(d.elem(s.start()).elem(0).elem(0).real());
  
  int n_3vec = (s.end()-s.start()+1)*24;
  vaxpy3(yptr, aptr, xptr, yptr, n_3vec);
	
}


// Vec = Vec *Scalar  + Vec (AXPY)
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpAdd,
	       BinaryNode<OpMultiply, 
	       Reference< QDPType< TVec, OLattice< TVec > > >,
	       Reference< QDPType< TScal, OScalar< TScal > > > >,
	       Reference< QDPType< TVec, OLattice< TVec > > > >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{

#ifdef DEBUG_BLAS
  QDPIO::cout << "SSE: z = x*a + y" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&> (rhs.expression().right());

  // ax is the right side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TVec, OLattice< TVec > > >,    
    Reference< QDPType< TScal, OScalar< TScal > > > > BN;

  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().left());

  // get a and x out of the bynary node
  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode.right());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode.left());
  // Set pointers 
  REAL32 ar =  a.elem().elem().elem().elem();
  REAL32 *aptr = (REAL32 *)&ar;
  REAL32 *xptr = (REAL32 *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL32 *yptr = (REAL32 *) &(y.elem(s.start()).elem(0).elem(0).real());
  REAL32* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*24;
  vaxpy3(zptr, aptr, xptr, yptr, n_3vec);
}


// Vec = Vec + Vec * Scalar (AXPY)
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpAdd,
	       Reference< QDPType< TVec, OLattice< TVec > > >,
	       BinaryNode<OpMultiply, 
	       Reference< QDPType< TVec, OLattice< TVec > > >,
	       Reference< QDPType< TScal, OScalar< TScal > > > > >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "SSE: z = y + x*a" << endl;
#endif


  // Peel the stuff out of the expression

  // y is the left side of rhs
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&> (rhs.expression().left());

  // ax is the right side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TVec, OLattice< TVec > > >,    
    Reference< QDPType< TScal, OScalar< TScal > > > > BN;

  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().right());

  // get a and x out of the bynary node
  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode.right());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode.left());
  // Set pointers 
  REAL32 ar =  a.elem().elem().elem().elem();
  REAL32 *aptr = (REAL32 *)&ar;
  REAL32 *xptr = (REAL32 *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL32 *yptr = (REAL32 *) &(y.elem(s.start()).elem(0).elem(0).real());
  REAL32* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());

  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*24;
  vaxpy3(zptr, aptr, xptr, yptr, n_3vec);

}


// Vec = Vec*Scalar - Vec (AXMY)
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpSubtract,
	       BinaryNode<OpMultiply, 
	       Reference< QDPType< TVec, OLattice< TVec > > >,
	       Reference< QDPType< TScal, OScalar< TScal > > > >,
	       Reference< QDPType< TVec, OLattice< TVec > > > >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "SSE: z = x*a - y" << endl;
#endif

  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&> (rhs.expression().right());


  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TVec, OLattice< TVec > > >,    
    Reference< QDPType< TScal, OScalar< TScal > > > > BN;

  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().left());

  // get a and x out of the bynary node
  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode.right());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode.left());
  // Set pointers 
  REAL32 ar =  a.elem().elem().elem().elem();
  REAL32 *aptr = (REAL32 *)&ar;
  REAL32 *xptr = (REAL32 *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL32 *yptr = (REAL32 *) &(y.elem(s.start()).elem(0).elem(0).real());
  REAL32* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*24;
  vaxmy3(zptr, aptr, xptr, yptr, n_3vec);
}


// Vec = Vec - Vec*Scalar (AXPY with -Scalar)
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpSubtract,
	       Reference< QDPType< TVec, OLattice< TVec > > >,
	       BinaryNode<OpMultiply, 
	       Reference< QDPType< TVec, OLattice< TVec > > >,
	       Reference< QDPType< TScal, OScalar< TScal > > > > >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "SSE: z = y - x*a" << endl;
#endif

  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&> (rhs.expression().left());

  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TVec, OLattice< TVec > > >,    
    Reference< QDPType< TScal, OScalar< TScal > > > > BN;

  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().right());

  // get a and x out of the bynary node
  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode.right());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode.left());
  // Set pointers etc.

  // -ve sign as y - ax = -ax + y  = axpy with -a.
  REAL32 ar =  -a.elem().elem().elem().elem();
  REAL32 *aptr = (REAL32 *)&ar;
  REAL32 *xptr = (REAL32 *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL32 *yptr = (REAL32 *) &(y.elem(s.start()).elem(0).elem(0).real());
  REAL32* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*24;
  vaxpy3(zptr, aptr, xptr, yptr, n_3vec);
}


template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpAdd,
	       Reference< QDPType< TVec, OLattice< TVec > > >,
	       Reference< QDPType< TVec, OLattice< TVec > > > >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{
#ifdef DEBUG_BLAS
  cout << "SSE: v+v " << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(rhs.expression().left());
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(rhs.expression().right());

  REAL32 *xptr = (REAL32 *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL32 *yptr = (REAL32 *) &(y.elem(s.start()).elem(0).elem(0).real());
  REAL32* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*24;
  vadd(zptr, xptr, yptr, n_3vec);
}

template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpSubtract,
	       Reference< QDPType< TVec, OLattice< TVec > > >,
	       Reference< QDPType< TVec, OLattice< TVec > > > >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{
#ifdef DEBUG_BLAS
  cout << "SSE: v-v " << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(rhs.expression().left());
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(rhs.expression().right());

  REAL32 *xptr = (REAL32 *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL32 *yptr = (REAL32 *) &(y.elem(s.start()).elem(0).elem(0).real());
  REAL32* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());

  
  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*24;
  vsub(zptr, xptr, yptr, n_3vec);
}

// Vec = Scal * Vec
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpMultiply,
	       Reference< QDPType< TScal, OScalar< TScal > > >,
	       Reference< QDPType< TVec, OLattice< TVec > > > >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{
#ifdef DEBUG_BLAS
  cout << "SSE: v = a*v " << endl;
#endif
  const OLattice< TVec > &x = static_cast<const OLattice< TVec >&>(rhs.expression().right());
  const OScalar< TScal > &a = static_cast<const OScalar< TScal >&>(rhs.expression().left());

  REAL32 ar =  a.elem().elem().elem().elem();
  REAL32 *aptr = &ar;  
  REAL32 *xptr = (REAL32 *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL32 *zptr =  &(d.elem(s.start()).elem(0).elem(0).real());
  int n_3vec = (s.end()-s.start()+1)*24;

  vscal(zptr, aptr, xptr, n_3vec);
}

template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpMultiply,
	       Reference< QDPType< TVec, OLattice< TVec > > >,
	       Reference< QDPType< TScal, OScalar< TScal > > > >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{
#ifdef DEBUG_BLAS
  cout << "SSE: v = v*a " << endl;
#endif

  const OLattice< TVec > &x = static_cast<const OLattice< TVec >&>(rhs.expression().left());
  const OScalar< TScal > &a = static_cast<const OScalar< TScal >&>(rhs.expression().right());

  REAL32 ar =  a.elem().elem().elem().elem();
  REAL32 *aptr = &ar;  
  REAL32 *xptr = (REAL32 *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL32 *zptr =  &(d.elem(s.start()).elem(0).elem(0).real());
  int n_3vec = (s.end()-s.start()+1)*24;

  vscal(zptr, aptr, xptr, n_3vec);
}
#endif     // if defined(QDP_SCALARSITE_USE_EVALUATE)



#if 1
// Global norm squared of a vector...
template<>
inline UnaryReturn<OLattice< TVec >, FnNorm2>::Type_t
norm2(const QDPType<TVec ,OLattice< TVec > >& s1, const Subset& s)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "Using SSE sumsq" << endl;
#endif

  if ( s.hasOrderedRep() ) {

#ifdef DEBUG_BLAS
    QDPIO::cout << "BJ sumsq " << endl;
#endif
    int n_3vec = (s.end() - s.start() + 1)*24;
    const REAL32 *s1ptr =  &(s1.elem(s.start()).elem(0).elem(0).real());
    
    // I am relying on this being a Double here 
    REAL64 ltmp=0;
    local_sumsq(&ltmp, (REAL32 *)s1ptr, n_3vec); 

    UnaryReturn< OLattice< TVec >, FnNorm2>::Type_t  lsum(ltmp);
    Internal::globalSum(lsum);
    return lsum;
  }
  else {
   return sum(localNorm2(s1),s);
  }
}

template<>
inline UnaryReturn<OLattice< TVec >, FnNorm2>::Type_t
norm2(const QDPType<TVec ,OLattice< TVec > >& s1)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "Using SSE sumsq all" << endl;
#endif

  int n_3vec = (all.end() - all.start() + 1)*24;
  const REAL32 *s1ptr =  &(s1.elem(all.start()).elem(0).elem(0).real());
    
  // I am relying on this being a Double here 
  REAL64 ltmp=0;
  local_sumsq(&ltmp, (REAL32 *)s1ptr, n_3vec); 
  UnaryReturn< OLattice< TVec >, FnNorm2>::Type_t  lsum(ltmp);
  Internal::globalSum(lsum);
  return lsum;
}
#endif


template<>
inline  BinaryReturn< OLattice<TVec>, OLattice<TVec>, FnInnerProduct>::Type_t
innerProduct(const QDPType< TVec, OLattice<TVec> > &v1,
	     const QDPType< TVec, OLattice<TVec> > &v2)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "BJ: innerProduct all" << endl;
#endif

  // This BinaryReturn has Type_t
  // OScalar<OScalar<OScalar<RComplex<PScalar<REAL> > > > >
  BinaryReturn< OLattice<TVec>, OLattice<TVec>, FnInnerProduct>::Type_t lprod;
  // Inner product is accumulated internally in DOUBLE
  DOUBLE ip[2];
  ip[0]=0;
  ip[1]=0;

  // Length of subset 
  unsigned long n_3vec = (all.end() - all.start() + 1)*Ns;
    
  // Call My CDOT
  local_vcdot(&(ip[0]), &(ip[1]),
	      (REAL *)&(v1.elem(all.start()).elem(0).elem(0).real()),
	      (REAL *)&(v2.elem(all.start()).elem(0).elem(0).real()),
	      n_3vec);


  // Global sum -- still on a vector of doubles
  Internal::globalSumArray(ip,2);

  // Downcast (and possibly lose precision) here 
  lprod.elem().elem().elem().real() = ip[0];
  lprod.elem().elem().elem().imag() = ip[1];

  // Return
  return lprod;
}

template<>
inline  BinaryReturn< OLattice<TVec>, OLattice<TVec>, FnInnerProduct>::Type_t
innerProduct(const QDPType< TVec, OLattice<TVec> > &v1,
	     const QDPType< TVec, OLattice<TVec> > &v2, 
	     const Subset& s)
{
  if( s.hasOrderedRep() ) {
#ifdef DEBUG_BLAS
    QDPIO::cout << "BJ: innerProduct s" << endl;
#endif

    // This BinaryReturn has Type_t
    // OScalar<OScalar<OScalar<RComplex<PScalar<REAL> > > > >
    BinaryReturn< OLattice<TVec>, OLattice<TVec>, FnInnerProduct>::Type_t lprod;
    DOUBLE ip[2];
    ip[0] = 0;
    ip[1] = 0;

    unsigned long n_3vec = (s.end() - s.start() + 1)*Ns;
    local_vcdot(&(ip[0]), &(ip[1]),
		(REAL *)&(v1.elem(s.start()).elem(0).elem(0).real()),
		(REAL *)&(v2.elem(s.start()).elem(0).elem(0).real()),
		n_3vec);


    Internal::globalSumArray(ip,2);

    lprod.elem().elem().elem().real() = ip[0];
    lprod.elem().elem().elem().imag() = ip[1];
    

    return lprod;
  }
  else {
    return sum(localInnerProduct(v1, v2),s);
  }
}



// Inner Product Real
template<>
inline  
BinaryReturn< OLattice<TVec>, OLattice<TVec>, FnInnerProductReal>::Type_t
innerProductReal(const QDPType< TVec, OLattice<TVec> > &v1,
		 const QDPType< TVec, OLattice<TVec> > &v2)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "BJ: innerProductReal all" << endl;
#endif

  // This BinaryReturn has Type_t
  // OScalar<OScalar<OScalar<RScalar<PScalar<REAL> > > > >
  BinaryReturn< OLattice<TVec>, OLattice<TVec>, FnInnerProductReal>::Type_t lprod;
  // Inner product is accumulated internally in DOUBLE
  DOUBLE ip_re=0;

  // Length of subset 
  unsigned long n_3vec = (all.end() - all.start() + 1)*Ns;

  // Call My CDOT
  local_vcdot_real(&ip_re,
		   (REAL *)&(v1.elem(all.start()).elem(0).elem(0).real()),
		   (REAL *)&(v2.elem(all.start()).elem(0).elem(0).real()),
		   n_3vec);

  // Global sum
  Internal::globalSum(ip_re);

  // Whether CDOT did anything or not ip_re and ip_im should 
  // now be right. Assign them to the ReturnType
  lprod.elem().elem().elem().elem() = ip_re;


  // Return
  return lprod;
}


template<>
inline  
BinaryReturn< OLattice<TVec>, OLattice<TVec>, FnInnerProductReal>::Type_t
innerProductReal(const QDPType< TVec, OLattice<TVec> > &v1,
		 const QDPType< TVec, OLattice<TVec> > &v2, 
		 const Subset& s)
{
  if( s.hasOrderedRep() ) {
#ifdef DEBUG_BLAS
    QDPIO::cout << "BJ: innerProductReal s" << endl;
#endif

    // This BinaryReturn has Type_t
    // OScalar<OScalar<OScalar<RScalar<PScalar<REAL> > > > >
    BinaryReturn< OLattice<TVec>, OLattice<TVec>, FnInnerProductReal>::Type_t lprod;
    DOUBLE ip_re=0;

    unsigned long n_3vec = (s.end() - s.start() + 1)*Ns;
    local_vcdot_real(&ip_re,
		     (REAL *)&(v1.elem(s.start()).elem(0).elem(0).real()),
		     (REAL *)&(v2.elem(s.start()).elem(0).elem(0).real()),
		     n_3vec);

    Internal::globalSum(ip_re);
    lprod.elem().elem().elem().elem() = ip_re;


    return lprod;
  }
  else {
    // Do localInnerProduct
    // then sum
    return sum(localInnerProductReal(v1, v2),s);
  }
}


// Norms of arrays
// Global norm squared of an array
template<>
inline UnaryReturn<OLattice< TVec >, FnNorm2>::Type_t
norm2(const multi1d< OLattice< TVec > >& s1, const OrderedSubset& s)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "Using SSE multi1d sumsq" << endl;
#endif

  int n_3vec = (s.end() - s.start() + 1)*24;
  REAL64 ltmp = 0;
  for(int n=0; n < s1.size(); ++n)
  {
    const REAL32 *s1ptr =  &(s1[n].elem(s.start()).elem(0).elem(0).real());
    
    // I am relying on this being a Double here 
    REAL64 lltmp;
    local_sumsq(&lltmp, (REAL32 *)s1ptr, n_3vec); 

    ltmp += lltmp;
  }

  UnaryReturn< OLattice< TVec >, FnNorm2>::Type_t  lsum(ltmp);
  Internal::globalSum(lsum);
  return lsum;
}

template<>
inline UnaryReturn<OLattice< TVec >, FnNorm2>::Type_t
norm2(const multi1d< OLattice< TVec > >& s1)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "Using SSE multi1d sumsq all" << endl;
#endif

  int n_3vec = (all.end() - all.start() + 1)*24;
  REAL64 ltmp = 0;
  for(int n=0; n < s1.size(); ++n)
  {
    const REAL32 *s1ptr =  &(s1[n].elem(all.start()).elem(0).elem(0).real());
    
    // I am relying on this being a Double here 
    REAL64 lltmp;
    local_sumsq(&lltmp, (REAL32 *)s1ptr, n_3vec); 

    ltmp += lltmp;
  }

  UnaryReturn< OLattice< TVec >, FnNorm2>::Type_t  lsum(ltmp);
  Internal::globalSum(lsum);
  return lsum;
}


template<>
inline  BinaryReturn< OLattice<TVec>, OLattice<TVec>, FnInnerProduct>::Type_t
innerProduct(const multi1d< OLattice<TVec> > &v1,
	     const multi1d< OLattice<TVec> > &v2)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "BJ: multi1d innerProduct all" << endl;
#endif

  // This BinaryReturn has Type_t
  // OScalar<OScalar<OScalar<RComplex<PScalar<REAL> > > > >
  BinaryReturn< OLattice<TVec>, OLattice<TVec>, FnInnerProduct>::Type_t lprod;
  // Inner product is accumulated internally in DOUBLE
  DOUBLE ip[2];
  ip[0]=0;
  ip[1]=0;

  // Length of subset 
  unsigned long n_3vec = (all.end() - all.start() + 1)*Ns;
    
  for(int n=0; n < v1.size(); ++n)
  {
    DOUBLE iip[2];
    iip[0]=0;
    iip[1]=0;

    // Call My CDOT
    local_vcdot(&(iip[0]), &(iip[1]),
		(REAL *)&(v1[n].elem(all.start()).elem(0).elem(0).real()),
		(REAL *)&(v2[n].elem(all.start()).elem(0).elem(0).real()),
		n_3vec);
    
    ip[0] += iip[0];
    ip[1] += iip[1];
  }

  // Global sum -- still on a vector of doubles
  Internal::globalSumArray(ip,2);

  // Downcast (and possibly lose precision) here 
  lprod.elem().elem().elem().real() = ip[0];
  lprod.elem().elem().elem().imag() = ip[1];

  // Return
  return lprod;
}

template<>
inline  BinaryReturn< OLattice<TVec>, OLattice<TVec>, FnInnerProduct>::Type_t
innerProduct(const multi1d< OLattice<TVec> > &v1,
	     const multi1d< OLattice<TVec> > &v2, 
	     const OrderedSubset& s)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "BJ: multi1d innerProduct s" << endl;
#endif

  // This BinaryReturn has Type_t
  // OScalar<OScalar<OScalar<RComplex<PScalar<REAL> > > > >
  BinaryReturn< OLattice<TVec>, OLattice<TVec>, FnInnerProduct>::Type_t lprod;
  DOUBLE ip[2];
  ip[0] = 0;
  ip[1] = 0;

  unsigned long n_3vec = (s.end() - s.start() + 1)*Ns;
  DOUBLE iip[2];
  for(int n=0; n < v1.size(); ++n)
  {
    // Call My CDOT
    local_vcdot(&(iip[0]), &(iip[1]),
		(REAL *)&(v1[n].elem(s.start()).elem(0).elem(0).real()),
		(REAL *)&(v2[n].elem(s.start()).elem(0).elem(0).real()),
		n_3vec);

    ip[0] += iip[0];
    ip[1] += iip[1];
  }

  // Global sum -- still on a vector of doubles
  Internal::globalSumArray(ip,2);

  // Downcast (and possibly lose precision) here 
  lprod.elem().elem().elem().real() = ip[0];
  lprod.elem().elem().elem().imag() = ip[1];

  return lprod;
}


// Inner Product Real
template<>
inline  
BinaryReturn< OLattice<TVec>, OLattice<TVec>, FnInnerProductReal>::Type_t
innerProductReal(const multi1d< OLattice<TVec> > &v1,
		 const multi1d< OLattice<TVec> > &v2)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "BJ: innerProductReal(multi1d) all" << endl;
#endif

  // This BinaryReturn has Type_t
  // OScalar<OScalar<OScalar<RScalar<PScalar<REAL> > > > >
  BinaryReturn< OLattice<TVec>, OLattice<TVec>, FnInnerProductReal>::Type_t lprod;
  // Inner product is accumulated internally in DOUBLE
  DOUBLE ip_re=0;

  // Length of subset 
  unsigned long n_3vec = (all.end() - all.start() + 1)*Ns;

  for(int n=0; n < v1.size(); ++n)
  {
    DOUBLE iip_re=0;

    // Call My CDOT
    local_vcdot_real(&iip_re,
		     (REAL *)&(v1[n].elem(all.start()).elem(0).elem(0).real()),
		     (REAL *)&(v2[n].elem(all.start()).elem(0).elem(0).real()),
		     n_3vec);

    ip_re += iip_re;
  }

  // Global sum
  Internal::globalSum(ip_re);

  // Whether CDOT did anything or not ip_re and ip_im should 
  // now be right. Assign them to the ReturnType
  lprod.elem().elem().elem().elem() = ip_re;


  // Return
  return lprod;
}


template<>
inline  
BinaryReturn< OLattice<TVec>, OLattice<TVec>, FnInnerProductReal>::Type_t
innerProductReal(const multi1d< OLattice<TVec> > &v1,
		 const multi1d< OLattice<TVec> > &v2, 
		 const OrderedSubset& s)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "BJ: multi1d innerProductReal s" << endl;
#endif

  // This BinaryReturn has Type_t
  // OScalar<OScalar<OScalar<RScalar<PScalar<REAL> > > > >
  BinaryReturn< OLattice<TVec>, OLattice<TVec>, FnInnerProductReal>::Type_t lprod;
  DOUBLE ip_re=0;

  unsigned long n_3vec = (s.end() - s.start() + 1)*Ns;

  for(int n=0; n < v1.size(); ++n)
  {
    DOUBLE iip_re=0;

    local_vcdot_real(&iip_re,
		     (REAL *)&(v1[n].elem(s.start()).elem(0).elem(0).real()),
		     (REAL *)&(v2[n].elem(s.start()).elem(0).elem(0).real()),
		     n_3vec);

    ip_re += iip_re;
  }

  Internal::globalSum(ip_re);
  lprod.elem().elem().elem().elem() = ip_re;
  
  return lprod;
}

// AX + BY 

// z = ax + b y
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpAdd,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< TScal, OScalar< TScal > > >,
	         Reference< QDPType< TVec, OLattice< TVec > > > 
                >,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< TScal, OScalar< TScal > > >,
	         Reference< QDPType< TVec, OLattice< TVec > > > 
                > 
               >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{

#ifdef DEBUG_BLAS_VAXPBY
  QDPIO::cout << "SSE_TEST: vaxpby" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN1;

  // get the binary node
  const BN1 &mulNode1 = static_cast<const BN1&> (rhs.expression().left());
  const BN1 &mulNode2 = static_cast<const BN1&> (rhs.expression().right());

  // get a and x out of the binary node
  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode1.left());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode1.right());
  
  // get b and y out of the binary node
  const OScalar< TScal >& b = static_cast<const OScalar< TScal >&>(mulNode2.left());
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode2.right());

  
  // Set pointers 
  REAL *aptr = (REAL *)&(a.elem().elem().elem().elem());
  REAL *bptr = (REAL *)&(b.elem().elem().elem().elem());
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_4vec = (s.end()-s.start()+1);
  axpbyz(zptr, aptr, xptr, bptr, yptr, n_4vec);
}

// z = ax - b y
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpSubtract,
	         BinaryNode<OpMultiply, 
	           Reference< QDPType< TScal, OScalar< TScal > > >,
	           Reference< QDPType< TVec, OLattice< TVec > > > 
	         >,
	         BinaryNode<OpMultiply, 
	           Reference< QDPType< TScal, OScalar< TScal > > >,
	           Reference< QDPType< TVec, OLattice< TVec > > > 
                 > 
               >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{

#ifdef DEBUG_BLAS_VAXMBY
  QDPIO::cout << "SSE_TEST: vaxmby" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN1;

  // get the binary node
  const BN1 &mulNode1 = static_cast<const BN1&> (rhs.expression().left());
  const BN1 &mulNode2 = static_cast<const BN1&> (rhs.expression().right());

  // get a and x out of the binary node
  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode1.left());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode1.right());
  
  // get b and y out of the binary node
  const OScalar< TScal >& b = static_cast<const OScalar< TScal >&>(mulNode2.left());
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode2.right());

  
  // Set pointers 
  REAL *aptr = (REAL *)&(a.elem().elem().elem().elem());
  REAL *bptr = (REAL *)&(b.elem().elem().elem().elem());
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_4vec = (s.end()-s.start()+1);
  axmbyz(zptr, aptr, xptr, bptr, yptr, n_4vec);
}

#if defined(QDP_SCALARSITE_DEBUG)
#undef QDP_SCALARSITE_DEBUG
#endif

#if defined(QDP_SCALARSITE_USE_EVALUATE)
#undef QDP_SCALARSITE_USE_EVALUATE
#endif

  
QDP_END_NAMESPACE();

#endif  // guard
 
