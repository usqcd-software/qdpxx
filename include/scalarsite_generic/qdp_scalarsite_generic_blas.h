// $Id: qdp_scalarsite_generic_blas.h,v 1.1 2004-03-21 21:19:44 bjoo Exp $

/*! @file
 * @brief Intel SSE optimizations
 * 
 * SSE optimizations of basic operations
 */


#ifndef QDP_SCALARSITE_BLAS_H
#define QDP_SCALARSITE_BLAS_H


QDP_BEGIN_NAMESPACE(QDP);
// Forward declarations of BLAS routines
void vaxpy3(REAL *Out, REAL *scalep,REAL *InScale, REAL *Add,int n_3vec);
void vaxmy3(REAL *Out, REAL *scalep,REAL *InScale, REAL *Sub,int n_3vec);
void vadd(REAL *Out, REAL *In1, REAL *In2, int n_3vec);
void vsub(REAL *Out, REAL *In1, REAL *In2, int n_3vec);
void vscal(REAL *Out, REAL *scalep, REAL *In, int n_3vec);
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

  const LatticeFermion& x = static_cast<const LatticeFermion &>(rhs.expression().right());
  const Real& a = static_cast<const Real &> (rhs.expression().left());
  
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

  const LatticeFermion& x = static_cast<const LatticeFermion &>(rhs.expression().right());
  const Real& a = static_cast<const Real &> (rhs.expression().left());

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
  const LatticeFermion& y = static_cast<const LatticeFermion&> (rhs.expression().right());

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN;

  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().left());

  // get a and x out of the bynary node
  const Real& a = static_cast<const Real&>(mulNode.left());
  const LatticeFermion& x = static_cast<const LatticeFermion&>(mulNode.right());
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
  const LatticeFermion& y = static_cast<const LatticeFermion&> (rhs.expression().left());

  // ax is the right side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN;

  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().right());

  // get a and x out of the bynary node
  const Real& a = static_cast<const Real&>(mulNode.left());
  const LatticeFermion& x = static_cast<const LatticeFermion&>(mulNode.right());
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


  const LatticeFermion& y = static_cast<const LatticeFermion&> (rhs.expression().right());

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN;

  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().left());

  // get a and x out of the bynary node
  const Real& a = static_cast<const Real&>(mulNode.left());
  const LatticeFermion& x = static_cast<const LatticeFermion&>(mulNode.right());
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

  const LatticeFermion& y = static_cast<const LatticeFermion&> (rhs.expression().left());

  // ax is the right side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN;

  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().right());

  // get a and x out of the bynary node
  const Real& a = static_cast<const Real&>(mulNode.left());
  const LatticeFermion& x = static_cast<const LatticeFermion&>(mulNode.right());
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

  const LatticeFermion& x = static_cast<const LatticeFermion &>(rhs.expression().left());
  const Real& a = static_cast<const Real &> (rhs.expression().right());
  
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

  const LatticeFermion& x = static_cast<const LatticeFermion &>(rhs.expression().left());
  const Real& a = static_cast<const Real &> (rhs.expression().right());

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
  const LatticeFermion& y = static_cast<const LatticeFermion&> (rhs.expression().right());

  // ax is the right side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TVec, OLattice< TVec > > >,    
    Reference< QDPType< TScal, OScalar< TScal > > > > BN;

  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().left());

  // get a and x out of the bynary node
  const Real& a = static_cast<const Real&>(mulNode.right());
  const LatticeFermion& x = static_cast<const LatticeFermion&>(mulNode.left());
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
  const LatticeFermion& y = static_cast<const LatticeFermion&> (rhs.expression().left());

  // ax is the right side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TVec, OLattice< TVec > > >,    
    Reference< QDPType< TScal, OScalar< TScal > > > > BN;

  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().right());

  // get a and x out of the bynary node
  const Real& a = static_cast<const Real&>(mulNode.right());
  const LatticeFermion& x = static_cast<const LatticeFermion&>(mulNode.left());
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

  const LatticeFermion& y = static_cast<const LatticeFermion&> (rhs.expression().right());


  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TVec, OLattice< TVec > > >,    
    Reference< QDPType< TScal, OScalar< TScal > > > > BN;

  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().left());

  // get a and x out of the bynary node
  const Real& a = static_cast<const Real&>(mulNode.right());
  const LatticeFermion& x = static_cast<const LatticeFermion&>(mulNode.left());
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

  const LatticeFermion& y = static_cast<const LatticeFermion&> (rhs.expression().left());

  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TVec, OLattice< TVec > > >,    
    Reference< QDPType< TScal, OScalar< TScal > > > > BN;

  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().right());

  // get a and x out of the bynary node
  const Real& a = static_cast<const Real&>(mulNode.right());
  const LatticeFermion& x = static_cast<const LatticeFermion&>(mulNode.left());
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

  const LatticeFermion& x = static_cast<const LatticeFermion&>(rhs.expression().left());
  const LatticeFermion& y = static_cast<const LatticeFermion&>(rhs.expression().right());

  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real().elem());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
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
  cout << "BJ: v-v " << endl;
#endif

  const LatticeFermion& x = static_cast<const LatticeFermion&>(rhs.expression().left());
  const LatticeFermion& y = static_cast<const LatticeFermion&>(rhs.expression().right());

  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real().elem());

  
  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
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
  cout << "BJ: v = a*v " << endl;
#endif
  const LatticeFermion &x = static_cast<const LatticeFermion&>(rhs.expression().right());
  const Real &a = static_cast<const Real&>(rhs.expression().left());

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

  const LatticeFermion &x = static_cast<const LatticeFermion&>(rhs.expression().left());
  const Real &a = static_cast<const Real&>(rhs.expression().right());

  REAL ar =  a.elem().elem().elem().elem().elem();
  REAL *aptr = &ar;  
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *zptr =  &(d.elem(s.start()).elem(0).elem(0).real().elem());
  int n_3vec = (s.end()-s.start()+1)*Ns;

  vscal(zptr, aptr, xptr, n_3vec);
}
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

  
QDP_END_NAMESPACE();

#endif  // guard
 
