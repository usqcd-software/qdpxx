// $Id: qdp_scalarsite_generic_blas.h,v 1.8 2004-03-26 14:53:49 bjoo Exp $

/*! @file
 * @brief Intel SSE optimizations
 * 
 * SSE optimizations of basic operations
 */


#ifndef QDP_SCALARSITE_GENERIC_BLAS_H
#define QDP_SCALARSITE_GENERIC_BLAS_H


QDP_BEGIN_NAMESPACE(QDP);
// Forward declarations of BLAS routines

// Level 1 BLAS like operations. Vector operations all work 
// on 3 component vector -- reasonable prefetching on QCDOC on 
// whose assembler the C code is based.
// Hence typical value for n_3vec = ( s.end() - s.start() + 1 )*Ns
// where s is the subset under which the operation takes place.
// Scalars are always passed by address -- again for compatibility
// with assembler -- of course the QCDOC assembler is completely
// independent so this could be different.

// (Vector) out = (Scalar) (*scalep) * (Vector) InScale + (Vector) Add
void vaxpy3(REAL *Out, REAL *scalep,REAL *InScale, REAL *Add,int n_3vec);

// (Vector) Out = (Scalar) (*scalep) * (Vector) InScale - (Vector) Add
void vaxmy3(REAL *Out, REAL *scalep,REAL *InScale, REAL *Sub,int n_3vec);

// VAXPY with local norm accumulated in *norm
void vaxpy3_norm(REAL *Out,REAL *scalep,REAL *InScale, REAL *Add,int n_3vec, 
		 REAL *norm);

// (Vector) Out = (Scalar) (*ap) * (Vector)xp + (Scalar)(*bp) * (Vector)yp
void vaxpby3(REAL *Out, REAL *ap, REAL *xp, REAL *bp, REAL *yp, int n_3vec);

// (Vector) Out = (Scalar) (*ap) * (Vector)xp - (Scalar)(*bp) * (Vector)yp
void vaxmby3(REAL *Out, REAL *ap, REAL *xp, REAL *bp, REAL *yp, int n_3vec);

// VAXMY with local norm of result accumulated in *norm
void vaxpmy3_norm(REAL *Out,REAL *ap,REAL *xp, REAL *bp, REAL *yp, 
		  int n_3vec, REAL *norm);

// (Vector) Out = (Vector) In1 + (Vector) In2
void vadd(REAL *Out, REAL *In1, REAL *In2, int n_3vec);

// (Vector) Out = (Vector) In1 - (Vector) In2
void vsub(REAL *Out, REAL *In1, REAL *In2, int n_3vec);

// (Vector) out = (Scalar) (*scalep) * (Vector) In
void vscal(REAL *Out, REAL *scalep, REAL *In, int n_3vec);

// (Double) (*out) = || (Vector) In ||^2
void local_sumsq(DOUBLE *Out, REAL *In, int n_3vec);

typedef PSpinVector<PColorVector<RComplex<PScalar<REAL> >, 3>, 4> TVec;
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

// go for the throat -- 
// ax + by

// AXPY and AXMY routines
// Similar calling interface to assembler
// But with REAL instead of float
inline
void vaxpy3(REAL *Out,REAL *scalep,REAL *InScale, REAL *Add,int n_3vec)
{
  register double a;
  register double x0r;
  register double x0i;
  
  register double x1r;
  register double x1i;
  
  register double x2r;
  register double x2i;
  
  register double y0r;
  register double y0i;
  
  register double y1r;
  register double y1i;
  
  register double y2r;
  register double y2i;
  
  register double z0r;
  register double z0i;
  
  register double z1r;
  register double z1i;
  
  register double z2r;
  register double z2i;
  
  a = *scalep;
  
  register int index_x = 0;
  register int index_y = 0;
  register int index_z = 0;
  
  register int counter;
  
  for( counter = 0; counter < n_3vec; counter++) {
    x0r = (double)InScale[index_x++];
    y0r = (double)Add[index_y++];
    z0r = a*x0r + y0r;
    Out[index_z++] =(REAL) z0r;
    
    x0i = (double)InScale[index_x++];
    y0i = (double)Add[index_y++];
    z0i = a*x0i + y0i;
    Out[index_z++] =(REAL) z0i;
    
    x1r = (double)InScale[index_x++];
    y1r = (double)Add[index_y++];
    z1r = a*x1r + y1r;
    Out[index_z++] = (REAL)z1r;
    
    x1i = (double)InScale[index_x++];
    y1i = (double)Add[index_y++];
    z1i = a*x1i + y1i;
    Out[index_z++] = (REAL)z1i;
    
    x2r = (double)InScale[index_x++];     
    y2r = (double)Add[index_y++];
    z2r = a*x2r + y2r;
    Out[index_z++] = (REAL)z2r;
    
    x2i = (double)InScale[index_x++];
    y2i = (double)Add[index_y++];
    z2i = a*x2i + y2i;  
    Out[index_z++] = (REAL)z2i;
  }
}

inline
void vaxmy3(REAL *Out,REAL *scalep,REAL *InScale, REAL *Sub,int n_3vec)
{
  register double a;
  register double x0r;
  register double x0i;
  
  register double x1r;
  register double x1i;
  
  register double x2r;
  register double x2i;
  
  register double y0r;
  register double y0i;
  
  register double y1r;
  register double y1i;
  
  register double y2r;
  register double y2i;
  
  register double z0r;
  register double z0i;
  
  register double z1r;
  register double z1i;
  
  register double z2r;
  register double z2i;
  
  a = *scalep;
  
  register int index_x = 0;
  register int index_y = 0;
  register int index_z = 0;
  
  register int counter;
  
  for( counter = 0; counter < n_3vec; counter++) {
    x0r = (double)InScale[index_x++];
    y0r = (double)Sub[index_y++];
    z0r = a*x0r - y0r;
    Out[index_z++] = (REAL)z0r;
    
    x0i = (double)InScale[index_x++];
    y0i = (double)Sub[index_y++];
    z0i = a*x0i - y0i;
    Out[index_z++] = (REAL)z0i;
    
    x1r = (double)InScale[index_x++];
    y1r = (double)Sub[index_y++];
    z1r = a*x1r - y1r;
    Out[index_z++] = (REAL)z1r;
    
    x1i = (double)InScale[index_x++];
    y1i = (double)Sub[index_y++];
    z1i = a*x1i - y1i;
    Out[index_z++] = (REAL)z1i;
    
    x2r = (double)InScale[index_x++];     
    y2r = (double)Sub[index_y++];
    z2r = a*x2r - y2r;
    Out[index_z++] = (REAL)z2r;
    
    x2i = (double)InScale[index_x++];
    y2i = (double)Sub[index_y++];
    z2i = a*x2i - y2i;  
    Out[index_z++] = (REAL)z2i;
  }
}

// AXPY and AXMY routines
// Similar calling interface to assembler
// But with REAL instead of float
inline
void vaxpby3(REAL *Out,REAL *ap ,REAL *xp, REAL *bp,  REAL *yp ,int n_3vec)
{
  register double a;
  register double b;

  register double x0r;
  register double x0i;
  
  register double x1r;
  register double x1i;
  
  register double x2r;
  register double x2i;
  
  register double y0r;
  register double y0i;
  
  register double y1r;
  register double y1i;
  
  register double y2r;
  register double y2i;
  
  register double z0r;
  register double z0i;
  
  register double z1r;
  register double z1i;
  
  register double z2r;
  register double z2i;
  
  a = *ap;
  b = *bp;

  register int index_x = 0;
  register int index_y = 0;
  register int index_z = 0;
  
  register int counter;
  
  for( counter = 0; counter < n_3vec; counter++) {
    x0r = (double)xp[index_x++];
    y0r = (double)yp[index_y++];
    z0r = a*x0r;
    z0r = z0r + b*y0r;
    Out[index_z++] =(REAL) z0r;
    
    x0i = (double)xp[index_x++];
    y0i = (double)yp[index_y++];
    z0i = a*x0i;
    z0i = z0i + b*y0i;
    Out[index_z++] =(REAL) z0i;
    
    x1r = (double)xp[index_x++];
    y1r = (double)yp[index_y++];
    z1r = a*x1r;
    z1r = z1r + b*y1r;
    Out[index_z++] = (REAL)z1r;
    
    x1i = (double)xp[index_x++];
    y1i = (double)yp[index_y++];
    z1i = a*x1i;
    z1i = z1i + b*y1i;

    Out[index_z++] = (REAL)z1i;
    
    x2r = (double)xp[index_x++];     
    y2r = (double)yp[index_y++];
    z2r = a*x2r;
    z2r = z2r + b*y2r;
    Out[index_z++] = (REAL)z2r;
    
    x2i = (double)xp[index_x++];
    y2i = (double)yp[index_y++];
    z2i = a*x2i;
    z2i = z2i + b*y2i;
    Out[index_z++] = (REAL)z2i;
  }
}

inline
void vaxmby3(REAL *Out,REAL *ap,REAL *xp, REAL *bp, REAL *yp,int n_3vec)
{
  register double a;
  register double b;

  register double x0r;
  register double x0i;
  
  register double x1r;
  register double x1i;
  
  register double x2r;
  register double x2i;
  
  register double y0r;
  register double y0i;
  
  register double y1r;
  register double y1i;
  
  register double y2r;
  register double y2i;
  
  register double z0r;
  register double z0i;
  
  register double z1r;
  register double z1i;
  
  register double z2r;
  register double z2i;
  
  a = *ap;
  b = *bp;

  register int index_x = 0;
  register int index_y = 0;
  register int index_z = 0;
  
  register int counter;
  
  for( counter = 0; counter < n_3vec; counter++) {
    x0r = (double)xp[index_x++];
    y0r = (double)yp[index_y++];
    z0r = a*x0r;
    z0r = z0r - b*y0r;
    Out[index_z++] = (REAL)z0r;
    
    x0i = (double)xp[index_x++];
    y0i = (double)yp[index_y++];
    z0i = a*x0i;
    z0i = z0i - b*y0i;
    Out[index_z++] = (REAL)z0i;
    
    x1r = (double)xp[index_x++];
    y1r = (double)yp[index_y++];
    z1r = a*x1r;
    z1r = z1r - b*y1r;
    Out[index_z++] = (REAL)z1r;
    
    x1i = (double)xp[index_x++];
    y1i = (double)yp[index_y++];
    z1i = a*x1i;
    z1i = z1i - b*y1i;
    Out[index_z++] = (REAL)z1i;
    
    x2r = (double)xp[index_x++];     
    y2r = (double)yp[index_y++];
    z2r = a*x2r;
    z2r = z2r - b*y2r;
    Out[index_z++] = (REAL)z2r;
    
    x2i = (double)xp[index_x++];
    y2i = (double)yp[index_y++];
    z2i = a*x2i;
    z2i = z2i - b*y2i;  
    Out[index_z++] = (REAL)z2i;
  }
}


inline
void vadd(REAL *Out, REAL *In1, REAL *In2, int n_3vec)
{
  register double in10r;
  register double in10i;
  register double in11r;
  register double in11i;
  register double in12r;
  register double in12i;

  register double in20r;
  register double in20i;
  register double in21r;
  register double in21i;
  register double in22r;
  register double in22i;

  register double out0r;
  register double out0i;
  register double out1r;
  register double out1i;
  register double out2r;
  register double out2i;

  register int counter =0;
  register int in1ptr =0;
  register int in2ptr =0;
  register int outptr =0;

  if( n_3vec > 0 ) {
    in10r = (double)In1[in1ptr++];
    in20r = (double)In2[in2ptr++];
    in10i = (double)In1[in1ptr++];
    in20i = (double)In2[in2ptr++];
    for(counter = 0; counter < n_3vec-1; counter++) { 
      out0r = in10r + in20r;
      Out[outptr++] = (REAL)out0r;

      in11r = (double)In1[in1ptr++];
      in21r = (double)In2[in2ptr++];
      out0i = in10i + in20i;
      Out[outptr++] = (REAL)out0i;

      in11i = (double)In1[in1ptr++];
      in21i = (double)In2[in2ptr++];
      out1r = in11r + in21r;
      Out[outptr++] = (REAL)out1r;

      in12r = (double)In1[in1ptr++];
      in22r = (double)In2[in2ptr++];
      out1i = in11i + in21i;
      Out[outptr++] = (REAL)out1i;

      in12i = (double)In1[in1ptr++];
      in22i = (double)In2[in2ptr++];
      out2r = in12r + in22r;
      Out[outptr++] = (REAL)out2r;

      in10r = (double)In1[in1ptr++];
      in20r = (double)In2[in2ptr++];     
      out2i = in12i + in22i;
      Out[outptr++] = (REAL)out2i;

      in10i = (double)In1[in1ptr++];
      in20i = (double)In2[in2ptr++];
    }
    out0r = in10r + in20r;
    Out[outptr++] = (REAL)out0r;

    in11r = (double)In1[in1ptr++];
    in21r = (double)In2[in2ptr++];
    out0i = in10i + in20i;
    Out[outptr++] = (REAL)out0i;

    in11i = (double)In1[in1ptr++];
    in21i = (double)In2[in2ptr++];
    out1r = in11r + in21r;
    Out[outptr++] = (REAL)out1r;

    in12r = (double)In1[in1ptr++];
    in22r = (double)In2[in2ptr++];
    out1i = in11i + in21i;
    Out[outptr++] = (REAL)out1i;

    in12i = (double)In1[in1ptr++];
    in22i = (double)In2[in2ptr++];
    out2r = in12r + in22r;
    Out[outptr++] = (REAL)out2r;
    out2i = in12i + in22i;
    Out[outptr++] = (REAL)out2i;
  }
}

inline
void vsub(REAL *Out, REAL *In1, REAL *In2, int n_3vec)
{
  register double in10r;
  register double in10i;
  register double in11r;
  register double in11i;
  register double in12r;
  register double in12i;

  register double in20r;
  register double in20i;
  register double in21r;
  register double in21i;
  register double in22r;
  register double in22i;

  register double out0r;
  register double out0i;
  register double out1r;
  register double out1i;
  register double out2r;
  register double out2i;

  register int counter =0;
  register int in1ptr =0;
  register int in2ptr =0;
  register int outptr =0;

  if( n_3vec > 0 ) {
    in10r = (double)In1[in1ptr++];
    in20r = (double)In2[in2ptr++];
    in10i = (double)In1[in1ptr++];
    in20i = (double)In2[in2ptr++];
    for(counter = 0; counter < n_3vec-1; counter++) { 
      out0r = in10r - in20r;
      Out[outptr++] = (REAL)out0r;

      in11r = (double)In1[in1ptr++];
      in21r = (double)In2[in2ptr++];
      out0i = in10i - in20i;
      Out[outptr++] = (REAL)out0i;

      in11i = (double)In1[in1ptr++];
      in21i = (double)In2[in2ptr++];
      out1r = in11r - in21r;
      Out[outptr++] = (REAL)out1r;

      in12r = (double)In1[in1ptr++];
      in22r = (double)In2[in2ptr++];
      out1i = in11i - in21i;
      Out[outptr++] = (REAL)out1i;

      in12i = (double)In1[in1ptr++];
      in22i = (double)In2[in2ptr++];
      out2r = in12r - in22r;
      Out[outptr++] = (REAL)out2r;

      in10r = (double)In1[in1ptr++];
      in20r = (double)In2[in2ptr++];     
      out2i = in12i - in22i;
      Out[outptr++] = (REAL)out2i;

      in10i = (double)In1[in1ptr++];
      in20i = (double)In2[in2ptr++];
    }
    out0r = in10r - in20r;
    Out[outptr++] = (REAL)out0r;

    in11r = (double)In1[in1ptr++];
    in21r = (double)In2[in2ptr++];
    out0i = in10i - in20i;
    Out[outptr++] = (REAL)out0i;

    in11i = (double)In1[in1ptr++];
    in21i = (double)In2[in2ptr++];
    out1r = in11r - in21r;
    Out[outptr++] = (REAL)out1r;

    in12r = (double)In1[in1ptr++];
    in22r = (double)In2[in2ptr++];
    out1i = in11i - in21i;
    Out[outptr++] = (REAL)out1i;

    in12i = (double)In1[in1ptr++];
    in22i = (double)In2[in2ptr++];
    out2r = in12r - in22r;
    Out[outptr++] = (REAL)out2r;
    out2i = in12i - in22i;
    Out[outptr++] = (REAL)out2i;
  }
}

inline
void vscal(REAL *Out, REAL *scalep, REAL *In, int n_3vec)
{
  register double a = *scalep;

  register double i0r;
  register double i0i;
  register double i1r;
  register double i1i;
  register double i2r;
  register double i2i;

  register double o0r;
  register double o0i;
  register double o1r;
  register double o1i;
  register double o2r;
  register double o2i;

  register int counter=0;
  register int inptr=0;
  register int outptr=0;

  if( n_3vec > 0 ) {
    i0r = (double)In[inptr++];
    i0i = (double)In[inptr++];
    i1r = (double)In[inptr++];
    for(counter = 0; counter < n_3vec-1 ; counter++) {
      o0r = a*i0r;
      Out[outptr++] = (REAL)o0r;
      
      i1i = (double)In[inptr++];
      i2r = (double)In[inptr++];
      o0i = a*i0i;
      Out[outptr++] = (REAL)o0i;
      
      i2i = (double)In[inptr++];
      i0r = (double)In[inptr++];
      o1r = a*i1r;
      Out[outptr++] = (REAL)o1r;
      
      i0i = (double)In[inptr++];
      i1r = (double)In[inptr++]; // Last prefetched
      
      o1i = a*i1i;
      Out[outptr++] = (REAL)o1i;
      
      o2r= a*i2r;
      Out[outptr++] = (REAL)o2r;
      
      o2i= a*i2i;
      Out[outptr++] = (REAL)o2i;
    }

    o0r = a*i0r;
    Out[outptr++] =(REAL) o0r;
    
    i1i = (double)In[inptr++];
    i2r = (double)In[inptr++];
    o0i = a*i0i;
    Out[outptr++] = (REAL)o0i;
    
    i2i = (double)In[inptr++];
    o1r = a*i1r;
    Out[outptr++] = (REAL)o1r;
    
    o1i = a*i1i;
    Out[outptr++] = (REAL)o1i;
    
    o2r= a*i2r;
    Out[outptr++] = (REAL)o2r;
    
    o2i= a*i2i;
    Out[outptr++] = (REAL)o2i;
    
  }
}  

inline
void local_sumsq(DOUBLE *Out, REAL *In, int n_3vec)
{
  register double result;
  
  register double i1;
  register double i2;
  register double i3;
  register double i4;
  register double i5;
  register double i6;
  
  int counter;
  int vecptr=0;
  result = 0;

  if( n_3vec > 0 ) { 
    i1 = (double)In[vecptr++];
    i2 = (double)In[vecptr++];
    result = i1*i1 + result;
    for(counter=0; counter < n_3vec-1; counter++) {
      i3 = (double)In[vecptr++];
      result = i2*i2 + result;
      i4 = (double)In[vecptr++];
      result = i3*i3 + result;
      i5 = (double)In[vecptr++];
      result = i4*i4 + result;
      i6 = (double)In[vecptr++];
      result = i5*i5 + result;
      i1 = (double)In[vecptr++];
      result = i6*i6 + result;
      i2 = (double)In[vecptr++];
      result = i1*i1 + result;
    }
    
    i3 = (double)In[vecptr++];
    result = i2*i2 + result;
    i4 = (double)In[vecptr++];
    result = i3*i3 + result;
    i5 = (double)In[vecptr++];
    result = i4*i4 + result;
    i6 = (double)In[vecptr++];
    result = i5*i5 + result;
    result = i6*i6 + result;
  }
  
  *Out=(DOUBLE)result;
}

inline
void vaxpy3_norm(REAL *Out,REAL *scalep,REAL *InScale, REAL *Add,int n_3vec, REAL *norm)
{
  register double a;
  register double x0r;
  register double x0i;
  
  register double x1r;
  register double x1i;
  
  register double x2r;
  register double x2i;
  
  register double y0r;
  register double y0i;
  
  register double y1r;
  register double y1i;
  
  register double y2r;
  register double y2i;
  
  register double z0r;
  register double z0i;
  
  register double z1r;
  register double z1i;
  
  register double z2r;
  register double z2i;
  register double norm_out=0;

  a = *scalep;
  
  register int index_x = 0;
  register int index_y = 0;
  register int index_z = 0;
  
  register int counter;
  
  for( counter = 0; counter < n_3vec; counter++) {
    x0r = (double)InScale[index_x++];
    y0r = (double)Add[index_y++];
    z0r = a*x0r + y0r;
    norm_out += z0r*z0r;
    Out[index_z++] =(REAL) z0r;
    
    x0i = (double)InScale[index_x++];
    y0i = (double)Add[index_y++];
    z0i = a*x0i + y0i;
    norm_out += z0i*z0i;
    Out[index_z++] =(REAL) z0i;
    

    x1r = (double)InScale[index_x++];
    y1r = (double)Add[index_y++];
    z1r = a*x1r + y1r;
    norm_out += z1r * z1r; 
    Out[index_z++] = (REAL)z1r;
    
    x1i = (double)InScale[index_x++];
    y1i = (double)Add[index_y++];
    z1i = a*x1i + y1i;
    norm_out += z1i*z1i;
    Out[index_z++] = (REAL)z1i;
    
    x2r = (double)InScale[index_x++];     
    y2r = (double)Add[index_y++];
    z2r = a*x2r + y2r;
    norm_out += z2r*z2r;
    Out[index_z++] = (REAL)z2r;
    
    x2i = (double)InScale[index_x++];
    y2i = (double)Add[index_y++];
    z2i = a*x2i + y2i;  
    norm_out += z2i*z2i;
    Out[index_z++] = (REAL)z2i;
    
  }
  *norm=(REAL)norm_out;
}

// z = aX - bY, norm = || z ||^2 (local) 
inline
void vaxpmy3_norm(REAL *Out,REAL *ap,REAL *xp, REAL *bp, REAL *yp, 
		  int n_3vec, REAL *norm)
{
  register double a;
  register double b;
  register double x0r;
  register double x0i;
  
  register double x1r;
  register double x1i;
  
  register double x2r;
  register double x2i;
  
  register double y0r;
  register double y0i;
  
  register double y1r;
  register double y1i;
  
  register double y2r;
  register double y2i;
  
  register double z0r;
  register double z0i;
  
  register double z1r;
  register double z1i;
  
  register double z2r;
  register double z2i;
  register double norm_out=0;

  a = *ap;
  b = *bp;
  register int index_x = 0;
  register int index_y = 0;
  register int index_z = 0;
  
  register int counter;
  
  for( counter = 0; counter < n_3vec; counter++) {
    x0r = (double)xp[index_x++];
    y0r = (double)yp[index_y++];
    z0r = a*x0r;
    z0r = z0r - b*y0r;
    Out[index_z++] =(REAL) z0r;
    norm_out += z0r*z0r;

    x0i = (double)xp[index_x++];
    y0i = (double)yp[index_y++];
    z0i = a*x0i;
    z0i = z0i - b*y0i;
    Out[index_z++] =(REAL) z0i;
    norm_out += z0i*z0i;    

    x1r = (double)xp[index_x++];
    y1r = (double)yp[index_y++];
    z1r = a*x1r;
    z1r = z1r - b*y1r;
    Out[index_z++] = (REAL)z1r;
    norm_out += z1r * z1r; 
    
    x1i = (double)xp[index_x++];
    y1i = (double)yp[index_y++];
    z1i = a*x1i;
    z1i = z1i - b*y1i;
    Out[index_z++] = (REAL)z1i;
    norm_out += z1i*z1i;
    
    x2r = (double)xp[index_x++];     
    y2r = (double)yp[index_y++];
    z2r = a*x2r;
    z2r = z2r - b*y2r;
    Out[index_z++] = (REAL)z2r;
    norm_out += z2r*z2r;
    
    x2i = (double)xp[index_x++];
    y2i = (double)yp[index_y++];
    z2i = a*x2i;
    z2i = z2i - b*y2i;  
    Out[index_z++] = (REAL)z2i;
    norm_out += z2i*z2i;    
  }
  *norm=(REAL)norm_out;
}
  
QDP_END_NAMESPACE();

#endif  // guard
 
