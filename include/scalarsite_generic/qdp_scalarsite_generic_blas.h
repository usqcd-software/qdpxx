// $Id: qdp_scalarsite_generic_blas.h,v 1.11 2004-04-01 12:56:39 bjoo Exp $

/*! @file
 * @brief Generic Scalarsite  optimization hooks
 * 
 * 
 */


#ifndef QDP_SCALARSITE_GENERIC_BLAS_H
#define QDP_SCALARSITE_GENERIC_BLAS_H

#include "scalarsite_generic/generic_blas_vaxpy3.h"
#include "scalarsite_generic/generic_blas_vaxmy3.h"
#include "scalarsite_generic/generic_blas_vaxpby3.h"
#include "scalarsite_generic/generic_blas_vaxmby3.h"
#include "scalarsite_generic/generic_blas_vscal.h"
#include "scalarsite_generic/generic_blas_local_sumsq.h"
#include "scalarsite_generic/generic_blas_vaxpy3_norm.h"
#include "scalarsite_generic/generic_blas_vaxmby3_norm.h"
#include "scalarsite_generic/generic_blas_local_vcdot.h"
#include "scalarsite_generic/generic_blas_local_vcdot_real.h"

QDP_BEGIN_NAMESPACE(QDP);

// Types needed for the expression templates. 
// TVec has outer Ns template so it ought to work for staggered as well
typedef PSpinVector<PColorVector<RComplex<PScalar<REAL> >, 3>, Ns> TVec;
typedef PScalar<PScalar<RScalar<PScalar<REAL> > > >  TScal;

// #define DEBUG_BLAS
// TVec is the LatticeFermion from qdp_dwdefs.h with the OLattice<> stripped
// from around it

// TScalar is the usual Real, with the OScalar<> stripped from it
//
// THis is simply to make the code more readable, and reduces < < s and > >s
// in the template arguments

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
  QDPIO::cout << "y += a*x" << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec > &>(rhs.expression().right());
  const OScalar< TScal >& a = static_cast<const OScalar< TScal > &> (rhs.expression().left());
  
  REAL ar = a.elem().elem().elem().elem().elem();
  REAL* aptr = &ar;
  REAL* xptr = (REAL *)&(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* yptr = &(d.elem(s.start()).elem(0).elem(0).real().elem());
  // cout << "Specialised axpy a ="<< ar << endl;
  
  int n_3vec = (s.end()-s.start()+1)*Ns;
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
  QDPIO::cout << "y -= a*x" << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec > &>(rhs.expression().right());
  const OScalar< TScal >& a = static_cast<const OScalar< TScal > &> (rhs.expression().left());

  // - sign as y -= ax <=> y = y-ax = -ax + y = axpy with -a 
  REAL ar = -( a.elem().elem().elem().elem().elem());
  REAL* aptr = &ar;
  REAL* xptr = (REAL *)&(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* yptr = &(d.elem(s.start()).elem(0).elem(0).real().elem());
  
  int n_3vec = (s.end()-s.start()+1)*Ns;
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
  QDPIO::cout << "z = a*x + y" << endl;
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
  REAL ar =  a.elem().elem().elem().elem().elem();
  REAL *aptr = (REAL *)&ar;
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real().elem());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
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
  QDPIO::cout << "z = y + a*x" << endl;
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
  REAL ar =  a.elem().elem().elem().elem().elem();
  REAL *aptr = (REAL *)&ar;
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real().elem());

  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
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
  QDPIO::cout << "z = a*x - y" << endl;
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
  REAL ar =  a.elem().elem().elem().elem().elem();
  REAL *aptr = (REAL *)&ar;
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real().elem());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
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
  QDPIO::cout << "z = y - a*x" << endl;
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
  REAL ar =  -a.elem().elem().elem().elem().elem();
  REAL *aptr = (REAL *)&ar;
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real().elem());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
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
  QDPIO::cout << "y += x*a" << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec > &>(rhs.expression().left());
  const OScalar< TScal >& a = static_cast<const OScalar< TScal > &> (rhs.expression().right());
  
  REAL ar = a.elem().elem().elem().elem().elem();
  REAL* aptr = &ar;
  REAL* xptr = (REAL *)&(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* yptr = &(d.elem(s.start()).elem(0).elem(0).real().elem());
  // cout << "Specialised axpy a ="<< ar << endl;
  
  int n_3vec = (s.end()-s.start()+1)*Ns;
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
  QDPIO::cout << "y -= x*a" << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec > &>(rhs.expression().left());
  const OScalar< TScal >& a = static_cast<const OScalar< TScal > &> (rhs.expression().right());

  // - sign as y -= ax <=> y = y-ax = -ax + y = axpy with -a 
  REAL ar = -( a.elem().elem().elem().elem().elem());
  REAL* aptr = &ar;
  REAL* xptr = (REAL *)&(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* yptr = &(d.elem(s.start()).elem(0).elem(0).real().elem());
  
  int n_3vec = (s.end()-s.start()+1)*Ns;
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
  QDPIO::cout << "z = x*a + y" << endl;
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
  REAL ar =  a.elem().elem().elem().elem().elem();
  REAL *aptr = (REAL *)&ar;
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real().elem());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
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
  QDPIO::cout << "z = y + x*a" << endl;
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
  REAL ar =  a.elem().elem().elem().elem().elem();
  REAL *aptr = (REAL *)&ar;
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real().elem());

  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
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
  QDPIO::cout << "z = x*a - y" << endl;
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
  REAL ar =  a.elem().elem().elem().elem().elem();
  REAL *aptr = (REAL *)&ar;
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real().elem());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
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
  QDPIO::cout << "z = y - x*a" << endl;
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
  REAL ar =  -a.elem().elem().elem().elem().elem();
  REAL *aptr = (REAL *)&ar;
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real().elem());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
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
  cout << "BJ: v+v " << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(rhs.expression().left());
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(rhs.expression().right());

  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real().elem());
  REAL one = 1;

  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
  vaxpy3(zptr,&one, xptr, yptr, n_3vec);
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
  cout << "BJ: v-v " << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(rhs.expression().left());
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(rhs.expression().right());

  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real().elem());

  
  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
  REAL one=1;
  vaxmy3(zptr,&one, xptr, yptr, n_3vec);
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
  cout << "BJ: v = a*v " << endl;
#endif
  const OLattice< TVec > &x = static_cast<const OLattice< TVec >&>(rhs.expression().right());
  const OScalar< TScal > &a = static_cast<const OScalar< TScal >&>(rhs.expression().left());

  REAL ar =  a.elem().elem().elem().elem().elem();
  REAL *aptr = &ar;  
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *zptr =  &(d.elem(s.start()).elem(0).elem(0).real().elem());
  int n_3vec = (s.end()-s.start()+1)*Ns;

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
  cout << "BJ: v = v*a " << endl;
#endif

  const OLattice< TVec > &x = static_cast<const OLattice< TVec >&>(rhs.expression().left());
  const OScalar< TScal > &a = static_cast<const OScalar< TScal >&>(rhs.expression().right());

  REAL ar =  a.elem().elem().elem().elem().elem();
  REAL *aptr = &ar;  
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *zptr =  &(d.elem(s.start()).elem(0).elem(0).real().elem());
  int n_3vec = (s.end()-s.start()+1)*Ns;

  vscal(zptr, aptr, xptr, n_3vec);
}

// v *= a
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpMultiplyAssign &op,
	       const QDPExpr< 
	       UnaryNode<OpIdentity,
	       Reference< QDPType< TScal, OScalar< TScal > > > >,
	       OScalar< TScal > > &rhs,
	       const OrderedSubset& s)
{
  const OScalar< TScal >& a = static_cast< const OScalar<TScal >&>(rhs.expression().child());


#ifdef DEBUG_BLAS
  QDPIO::cout << "BJ: v *= a, a = " << a << endl;
#endif
  
  REAL ar = a.elem().elem().elem().elem().elem();
  REAL* xptr = &(d.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* zptr = xptr;
  int n_3vec = (s.end()-s.start()+1)*Ns;
  vscal(zptr,&ar, xptr, n_3vec);
}

// v /= a
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpDivideAssign &op,
	       const QDPExpr< 
	       UnaryNode<OpIdentity,
	       Reference< QDPType< TScal, OScalar< TScal > > > >,
	       OScalar< TScal > > &rhs,
	       const OrderedSubset& s)
{
  const OScalar< TScal >& a = static_cast< const OScalar<TScal >&>(rhs.expression().child());


#ifdef DEBUG_BLAS
  QDPIO::cout << "BJ: v /= a, a = " << a << endl;
#endif
  
  REAL ar = (REAL)1/a.elem().elem().elem().elem().elem();
  REAL* xptr = &(d.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* zptr = xptr;
  int n_3vec = (s.end()-s.start()+1)*Ns;
  vscal(zptr,&ar, xptr, n_3vec);
}

// v += v
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAddAssign &op,
	       const QDPExpr< 
	       UnaryNode<OpIdentity,
	       Reference< QDPType< TVec, OLattice< TVec > > > >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{
  const OLattice< TVec >& x = static_cast< const OLattice<TVec >&>(rhs.expression().child());

 

#ifdef DEBUG_BLAS
  QDPIO::cout << "BJ: v += v" << endl;
#endif

  int n_3vec = (s.end() - s.start()+1)*Ns;
  REAL *xptr = (REAL *)(&x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *)(&d.elem(s.start()).elem(0).elem(0).real().elem());
  REAL one = 1;
  vaxpy3(yptr, &one, yptr, xptr,n_3vec);

}

// v -= v
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpSubtractAssign &op,
	       const QDPExpr< 
	       UnaryNode<OpIdentity,
	       Reference< QDPType< TVec, OLattice< TVec > > > >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{
  const OLattice< TVec >& x = static_cast< const OLattice<TVec >&>(rhs.expression().child());

 

#ifdef DEBUG_BLAS
  QDPIO::cout << "BJ: v -= v" << endl;
#endif

  int n_3vec = (s.end() - s.start()+1)*Ns;
  REAL *xptr = (REAL *)(&x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *)(&d.elem(s.start()).elem(0).elem(0).real().elem());
  REAL one = 1;
  vaxmy3(yptr, &one, yptr, xptr, n_3vec);

}


// z = ax + by
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpAdd,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< TScal, OScalar< TScal > > >,
	         Reference< QDPType< TVec, OLattice< TVec > > > >,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< TScal, OScalar< TScal > > >,
	         Reference< QDPType< TVec, OLattice< TVec > > > > >,
	        OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{

#ifdef DEBUG_BLAS
  QDPIO::cout << "z = a*x + b*y" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN;

  // get the binary node
  const BN &mulNode1 = static_cast<const BN&> (rhs.expression().left());
  const BN &mulNode2 = static_cast<const BN&> (rhs.expression().right());

  // get a and x out of the binary node
  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode1.left());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode1.right());
  
  // get b and y out of the binary node
  const OScalar< TScal >& b = static_cast<const OScalar< TScal >&>(mulNode2.left());
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode2.right());

  
  // Set pointers 
  REAL *aptr = (REAL *)&(a.elem().elem().elem().elem().elem());
  REAL *bptr = (REAL *)&(b.elem().elem().elem().elem().elem());
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real().elem());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
  vaxpby3(zptr, aptr, xptr, bptr, yptr, n_3vec);
}


// z = xa + by
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpAdd,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< TVec, OLattice< TVec > > >,
	         Reference< QDPType< TScal, OScalar< TScal > > > >,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< TScal, OScalar< TScal > > >,
	         Reference< QDPType< TVec, OLattice< TVec > > > > >,
	        OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{

#ifdef DEBUG_BLAS
  QDPIO::cout << "z = x*a + b*y" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TVec, OLattice< TVec > > >,
    Reference< QDPType< TScal, OScalar< TScal > > > > BN1;

  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN2;


  
  // get the binary node
  const BN1 &mulNode1 = static_cast<const BN1&> (rhs.expression().left());
  const BN2 &mulNode2 = static_cast<const BN2&> (rhs.expression().right());

  // get a and x out of the binary node
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode1.left());

  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode1.right());
  
  // get b and y out of the binary node
  const OScalar< TScal >& b = static_cast<const OScalar< TScal >&>(mulNode2.left());
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode2.right());

  
  // Set pointers 
  REAL *aptr = (REAL *)&(a.elem().elem().elem().elem().elem());
  REAL *bptr = (REAL *)&(b.elem().elem().elem().elem().elem());
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real().elem());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
  vaxpby3(zptr, aptr, xptr, bptr, yptr, n_3vec);
}

// z = ax + yb
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpAdd,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< TScal, OScalar< TScal > > >,
	         Reference< QDPType< TVec, OLattice< TVec > > > >,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< TVec, OLattice< TVec > > >,
	         Reference< QDPType< TScal, OScalar< TScal > > > > >,
	        OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{

#ifdef DEBUG_BLAS
  QDPIO::cout << "z = a*x + y*b" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs

  // type of a*x
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN1;

  // type of y*b
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TVec, OLattice< TVec > > >,
    Reference< QDPType< TScal, OScalar< TScal > > > > BN2;


  
  // get the binary nodes
  // a*x node
  const BN1 &mulNode1 = static_cast<const BN1&> (rhs.expression().left());

  // y*b node
  const BN2 &mulNode2 = static_cast<const BN2&> (rhs.expression().right());

  // get a and x out of the binary node
  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode1.left());

  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode1.right());

  
  // get b and y out of the binary node
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode2.left());

  const OScalar< TScal >& b = static_cast<const OScalar< TScal >&>(mulNode2.right());

  
  // Set pointers 
  REAL *aptr = (REAL *)&(a.elem().elem().elem().elem().elem());
  REAL *bptr = (REAL *)&(b.elem().elem().elem().elem().elem());
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real().elem());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
  vaxpby3(zptr, aptr, xptr, bptr, yptr, n_3vec);
}

// z = xa + yb
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpAdd,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< TVec, OLattice< TVec > > >,
	         Reference< QDPType< TScal, OScalar< TScal > > > >,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< TVec, OLattice< TVec > > >,
	         Reference< QDPType< TScal, OScalar< TScal > > > > >,
	        OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{

#ifdef DEBUG_BLAS
  QDPIO::cout << "z = x*a + y*b" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TVec, OLattice< TVec > > >,
    Reference< QDPType< TScal, OScalar< TScal > > > > BN;

  // get the binary node
  const BN &mulNode1 = static_cast<const BN&> (rhs.expression().left());
  const BN &mulNode2 = static_cast<const BN&> (rhs.expression().right());

  // get a and x out of the binary node
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode1.left());
  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode1.right());
  
  // get b and y out of the binary node
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode2.left());

  const OScalar< TScal >& b = static_cast<const OScalar< TScal >&>(mulNode2.right());
  
  // Set pointers 
  REAL *aptr = (REAL *)&(a.elem().elem().elem().elem().elem());
  REAL *bptr = (REAL *)&(b.elem().elem().elem().elem().elem());
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real().elem());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
  vaxpby3(zptr, aptr, xptr, bptr, yptr, n_3vec);
}

// z = ax - by
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpSubtract,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< TScal, OScalar< TScal > > >,
	         Reference< QDPType< TVec, OLattice< TVec > > > >,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< TScal, OScalar< TScal > > >,
	         Reference< QDPType< TVec, OLattice< TVec > > > > >,
	        OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{

#ifdef DEBUG_BLAS
  QDPIO::cout << "z = a*x - b*y" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN;

  // get the binary node
  const BN &mulNode1 = static_cast<const BN&> (rhs.expression().left());
  const BN &mulNode2 = static_cast<const BN&> (rhs.expression().right());

  // get a and x out of the binary node
  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode1.left());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode1.right());
  
  // get b and y out of the binary node
  const OScalar< TScal >& b = static_cast<const OScalar< TScal >&>(mulNode2.left());
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode2.right());

  
  // Set pointers 
  REAL *aptr = (REAL *)&(a.elem().elem().elem().elem().elem());
  REAL *bptr = (REAL *)&(b.elem().elem().elem().elem().elem());
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real().elem());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
  vaxmby3(zptr, aptr, xptr, bptr, yptr, n_3vec);
}


// z = xa - by
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpSubtract,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< TVec, OLattice< TVec > > >,
	         Reference< QDPType< TScal, OScalar< TScal > > > >,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< TScal, OScalar< TScal > > >,
	         Reference< QDPType< TVec, OLattice< TVec > > > > >,
	        OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{

#ifdef DEBUG_BLAS
  QDPIO::cout << "z = x*a - b*y" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TVec, OLattice< TVec > > >,
    Reference< QDPType< TScal, OScalar< TScal > > > > BN1;

  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN2;


  
  // get the binary node
  const BN1 &mulNode1 = static_cast<const BN1&> (rhs.expression().left());
  const BN2 &mulNode2 = static_cast<const BN2&> (rhs.expression().right());

  // get a and x out of the binary node
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode1.left());

  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode1.right());
  
  // get b and y out of the binary node
  const OScalar< TScal >& b = static_cast<const OScalar< TScal >&>(mulNode2.left());
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode2.right());

  
  // Set pointers 
  REAL *aptr = (REAL *)&(a.elem().elem().elem().elem().elem());
  REAL *bptr = (REAL *)&(b.elem().elem().elem().elem().elem());
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real().elem());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
  vaxmby3(zptr, aptr, xptr, bptr, yptr, n_3vec);
}

// z = ax - yb
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpSubtract,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< TScal, OScalar< TScal > > >,
	         Reference< QDPType< TVec, OLattice< TVec > > > >,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< TVec, OLattice< TVec > > >,
	         Reference< QDPType< TScal, OScalar< TScal > > > > >,
	        OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{

#ifdef DEBUG_BLAS
  QDPIO::cout << "z = a*x - y*b" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs

  // type of a*x
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN1;

  // type of y*b
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TVec, OLattice< TVec > > >,
    Reference< QDPType< TScal, OScalar< TScal > > > > BN2;


  
  // get the binary nodes
  // a*x node
  const BN1 &mulNode1 = static_cast<const BN1&> (rhs.expression().left());

  // y*b node
  const BN2 &mulNode2 = static_cast<const BN2&> (rhs.expression().right());

  // get a and x out of the binary node
  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode1.left());

  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode1.right());

  
  // get b and y out of the binary node
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode2.left());

  const OScalar< TScal >& b = static_cast<const OScalar< TScal >&>(mulNode2.right());

  
  // Set pointers 
  REAL *aptr = (REAL *)&(a.elem().elem().elem().elem().elem());
  REAL *bptr = (REAL *)&(b.elem().elem().elem().elem().elem());
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real().elem());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
  vaxmby3(zptr, aptr, xptr, bptr, yptr, n_3vec);
}

// z = xa - yb
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpSubtract,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< TVec, OLattice< TVec > > >,
	         Reference< QDPType< TScal, OScalar< TScal > > > >,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< TVec, OLattice< TVec > > >,
	         Reference< QDPType< TScal, OScalar< TScal > > > > >,
	        OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{

#ifdef DEBUG_BLAS
  QDPIO::cout << "z = x*a - y*b" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TVec, OLattice< TVec > > >,
    Reference< QDPType< TScal, OScalar< TScal > > > > BN;

  // get the binary node
  const BN &mulNode1 = static_cast<const BN&> (rhs.expression().left());
  const BN &mulNode2 = static_cast<const BN&> (rhs.expression().right());

  // get a and x out of the binary node
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode1.left());
  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode1.right());
  
  // get b and y out of the binary node
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode2.left());

  const OScalar< TScal >& b = static_cast<const OScalar< TScal >&>(mulNode2.right());
  
  // Set pointers 
  REAL *aptr = (REAL *)&(a.elem().elem().elem().elem().elem());
  REAL *bptr = (REAL *)&(b.elem().elem().elem().elem().elem());
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real().elem());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
  vaxmby3(zptr, aptr, xptr, bptr, yptr, n_3vec);
}

// Global norm squared of a vector...
template<>
inline UnaryReturn<OLattice< TVec >, FnNorm2>::Type_t
norm2(const QDPType<TVec ,OLattice< TVec > >& s1, const Subset& s)
{

#ifdef DEBUG_BLAS
  QDPIO::cout << "Using BJ sumsq" << endl;
#endif
  if ( s.hasOrderedRep() ) {

#ifdef DEBUG_BLAS
    QDPIO::cout << "BJ sumsq " << endl;
#endif
    int n_3vec = (s.end() - s.start() + 1)*Ns;
    const REAL *s1ptr =  &(s1.elem(s.start()).elem(0).elem(0).real().elem());
    
    // Has Type OScalar< PScalar < PScalar < RScalar < PScalar < REAL > > > > >
    UnaryReturn< OLattice< TVec >, FnNorm2>::Type_t  lsum;
    lsum = Double(0);
    
    local_sumsq((DOUBLE *)&(lsum.elem().elem().elem().elem().elem()),
		(REAL *)s1ptr, 
	       n_3vec); 

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
  QDPIO::cout << "Using BJ sumsq all" << endl;
#endif

  if ( all.hasOrderedRep() ) {

#ifdef DEBUG_BLAS
    QDPIO::cout << "BJ sumsq all" << endl;
#endif
    int n_3vec = (all.end() - all.start() + 1)*Ns;
    const REAL *s1ptr =  &(s1.elem(all.start()).elem(0).elem(0).real().elem());
    
    // I am relying on this being a Double here 
    UnaryReturn< OLattice< TVec >, FnNorm2>::Type_t  lsum;
    lsum = Double(0);
 
    local_sumsq((DOUBLE *)&(lsum.elem().elem().elem().elem().elem()),
		(REAL *)s1ptr, 
	       n_3vec); 
    Internal::globalSum(lsum);
    return lsum;
  }
  else {
   return sum(localNorm2(s1),all);
  }
}

template<>
inline  BinaryReturn< OLattice<TVec>, OLattice<TVec>, FnInnerProduct>::Type_t
innerProduct(const QDPType< TVec, OLattice<TVec> > &v1,
	     const QDPType< TVec, OLattice<TVec> > &v2)
{
  if( all.hasOrderedRep() ) {
#ifdef DEBUG_BLAS
    QDPIO::cout << "BJ: innerProduct all" << endl;
#endif

    // This BinaryReturn has Type_t
    // OScalar<OScalar<OScalar<RComplex<PScalar<REAL> > > > >
    BinaryReturn< OLattice<TVec>, OLattice<TVec>, FnInnerProduct>::Type_t lprod;
    // Inner product is accumulated internally in DOUBLE
    multi1d<DOUBLE> ip(2);
    ip[0]=0;
    ip[1]=0;

    // Length of subset 
    unsigned long n_3vec = (all.end() - all.start() + 1)*Ns;
    
    // Call My CDOT
    local_vcdot(&(ip[0]), &(ip[1]),
		(REAL *)&(v1.elem(all.start()).elem(0).elem(0).real().elem()),
		(REAL *)&(v2.elem(all.start()).elem(0).elem(0).real().elem()),
		n_3vec);


    // Global sum -- still on a vector of doubles
    Internal::globalSum(ip);

    // Downcast (and possibly lose precision) here 
    lprod.elem().elem().elem().real().elem() = (REAL)ip[0];
    lprod.elem().elem().elem().imag().elem() = (REAL)ip[1];

    // Return
    return lprod;
  }
  else {
    // Do localInnerProduct
    // then sum
    return sum(localInnerProduct(v1, v2),all);
  }
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
    multi1d<DOUBLE> ip(2);
    ip[0] = 0;
    ip[1] = 0;

    unsigned long n_3vec = (s.end() - s.start() + 1)*Ns;
    local_vcdot(&(ip[0]), &(ip[1]),
		(REAL *)&(v1.elem(s.start()).elem(0).elem(0).real().elem()),
		(REAL *)&(v2.elem(s.start()).elem(0).elem(0).real().elem()),
		n_3vec);

    Internal::globalSum(ip);

    lprod.elem().elem().elem().real().elem() = (REAL)ip[0];
    lprod.elem().elem().elem().imag().elem() = (REAL)ip[1];
    

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
  if( all.hasOrderedRep() ) {
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
		(REAL *)&(v1.elem(all.start()).elem(0).elem(0).real().elem()),
		(REAL *)&(v2.elem(all.start()).elem(0).elem(0).real().elem()),
		n_3vec);

    // Global sum
    Internal::globalSum(ip_re);

    // Whether CDOT did anything or not ip_re and ip_im should 
    // now be right. Assign them to the ReturnType
    lprod.elem().elem().elem().elem().elem() = (REAL)ip_re;


    // Return
    return lprod;
  }
  else {
    // Do localInnerProduct
    // then sum
    return sum(localInnerProductReal(v1, v2),all);
  }
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
		     (REAL *)&(v1.elem(s.start()).elem(0).elem(0).real().elem()),
		     (REAL *)&(v2.elem(s.start()).elem(0).elem(0).real().elem()),
		     n_3vec);

    Internal::globalSum(ip_re);
    lprod.elem().elem().elem().elem().elem() = (REAL)ip_re;


    return lprod;
  }
  else {
    return sum(localInnerProductReal(v1, v2),s);
  }
}


QDP_END_NAMESPACE();

#endif  // guard
 
