#ifndef QDP_SCALARSITE_GENERIC_CBLAS
#define QDP_SCALARSITE_GENERIC_CBLAS

// Complex BLAS routines

#include "scalarsite_generic/generic_blas_vcscal.h"
#include "scalarsite_generic/generic_blas_vcaxpy3.h"
#include "scalarsite_generic/generic_blas_vcaxmy3.h"

QDP_BEGIN_NAMESPACE(QDP);

typedef PScalar<PScalar<RComplex<PScalar<REAL> > > >  CScal;
typedef PSpinVector<PColorVector<RComplex<PScalar<REAL> >, 3>, Ns> TVec;

// vector z *= complex a
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpMultiplyAssign &op,
	       const QDPExpr< 
	       UnaryNode<OpIdentity,
	       Reference< QDPType< CScal, OScalar< CScal > > > >,
	       OScalar< CScal > > &rhs,
	       const OrderedSubset& s)
{
  const OScalar< CScal >& a = static_cast< const OScalar<CScal >&>(rhs.expression().child()); 

#ifdef DEBUG_BLAS  
  QDPIO::cout << "BJ: Complex v *= a " << a << endl;
#endif

  REAL *d_start = &(d.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *a_start = (REAL *) &(a.elem().elem().elem().real().elem());
  int n_3vec =( s.end() - s.start() + 1 )*Ns;

  vcscal(d_start, a_start, d_start, n_3vec);
}


// vector z = complex a * vector x
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpMultiply,
	       Reference< QDPType< CScal, OScalar< CScal > > >,
	       Reference< QDPType< TVec, OLattice< TVec > > > >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{
  const OScalar< CScal >& a = static_cast< const OScalar<CScal >&>(rhs.expression().left()); 
  
  const OLattice< TVec > &x = static_cast<const OLattice< TVec >&>(rhs.expression().right());

#ifdef DEBUG_BLAS
  QDPIO::cout << "BJ: Complex v = a*x " << a << endl;
#endif

  REAL *d_start = &(d.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *x_start = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());

  REAL *a_start = (REAL *) &(a.elem().elem().elem().real().elem());
  int n_3vec =( s.end() - s.start() + 1 )*Ns;

  vcscal(d_start, a_start, x_start, n_3vec);
}

// vector z = vector x * complex a
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpMultiply,
	       Reference< QDPType< TVec, OLattice< TVec > > >,
	       Reference< QDPType< CScal, OScalar< CScal > > > >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{
  const OScalar< CScal >& a = static_cast< const OScalar<CScal >&>(rhs.expression().right()); 
  
  const OLattice< TVec > &x = static_cast<const OLattice< TVec >&>(rhs.expression().left());

#ifdef DEBUG_BLAS
  QDPIO::cout << "BJ: Complex v = x*a " << a << endl;
#endif

  REAL *d_start = &(d.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *x_start = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());

  REAL *a_start = (REAL *) &(a.elem().elem().elem().real().elem());
  int n_3vec =( s.end() - s.start() + 1 )*Ns;

  vcscal(d_start, a_start, x_start, n_3vec);
}

//
// AXPYs.
//
//  y += a*x
template<>
inline
void evaluate(OLattice< TVec >& d, 
	      const OpAddAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	      Reference< QDPType< CScal, OScalar < CScal > > >,
	      Reference< QDPType< TVec, OLattice< TVec > > > >,
	      OLattice< TVec > > &rhs,
	      const OrderedSubset& s)
{

#ifdef DEBUG_BLAS
  QDPIO::cout << "y += a*x" << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec > &>(rhs.expression().right());
  const OScalar< CScal >& a = static_cast<const OScalar< CScal > &> (rhs.expression().left());
  
  REAL* ar   = (REAL *)&(a.elem().elem().elem().real().elem());
  REAL* xptr = (REAL *)&(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* yptr = &(d.elem(s.start()).elem(0).elem(0).real().elem());
  // cout << "Specialised axpy a ="<< ar << endl;
  
  int n_3vec = (s.end()-s.start()+1)*Ns;
  vcaxpy3(yptr, ar, xptr, yptr, n_3vec);
}

//  y += x*a
template<>
inline
void evaluate(OLattice< TVec >& d, 
	      const OpAddAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	      Reference< QDPType< TVec, OLattice< TVec > > >,
	      Reference< QDPType< CScal, OScalar < CScal > > > >,
	      OLattice< TVec > > &rhs,
	      const OrderedSubset& s)
{

#ifdef DEBUG_BLAS
  QDPIO::cout << "y += x*a" << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec > &>(rhs.expression().left());
  const OScalar< CScal >& a = static_cast<const OScalar< CScal > &> (rhs.expression().right());
  
  REAL* ar   = (REAL *)&(a.elem().elem().elem().real().elem());
  REAL* xptr = (REAL *)&(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* yptr = &(d.elem(s.start()).elem(0).elem(0).real().elem());
  // cout << "Specialised axpy a ="<< ar << endl;
  
  int n_3vec = (s.end()-s.start()+1)*Ns;
  vcaxpy3(yptr, ar, xptr, yptr, n_3vec);
}

//  y -= a*x
template<>
inline
void evaluate(OLattice< TVec >& d, 
	      const OpSubtractAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	      Reference< QDPType< CScal, OScalar < CScal > > >,
	      Reference< QDPType< TVec, OLattice< TVec > > > >,
	      OLattice< TVec > > &rhs,
	      const OrderedSubset& s)
{

#ifdef DEBUG_BLAS
  QDPIO::cout << "y -= a*x" << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec > &>(rhs.expression().right());
  const OScalar< CScal >& a = static_cast<const OScalar< CScal > &> (rhs.expression().left());

  // Get minus a
  OScalar<CScal> m_a = -a;

  REAL* ar   = (REAL *)&(m_a.elem().elem().elem().real().elem());
  REAL* xptr = (REAL *)&(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* yptr = &(d.elem(s.start()).elem(0).elem(0).real().elem());
  // cout << "Specialised axpy a ="<< ar << endl;
  
  int n_3vec = (s.end()-s.start()+1)*Ns;
  vcaxpy3(yptr, ar, xptr, yptr, n_3vec);
}

//  y -= x*a
template<>
inline
void evaluate(OLattice< TVec >& d, 
	      const OpSubtractAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	      Reference< QDPType< TVec, OLattice< TVec > > >,
	      Reference< QDPType< CScal, OScalar < CScal > > > >,
	      OLattice< TVec > > &rhs,
	      const OrderedSubset& s)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "y -= x*a" << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec > &>(rhs.expression().left());
  const OScalar< CScal >& a = static_cast<const OScalar< CScal > &> (rhs.expression().right());

  // Get minus a
  OScalar<CScal> m_a = -a;
  
  REAL* ar   = (REAL *)&(m_a.elem().elem().elem().real().elem());
  REAL* xptr = (REAL *)&(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL* yptr = &(d.elem(s.start()).elem(0).elem(0).real().elem());
  // cout << "Specialised axpy a ="<< ar << endl;
  
  int n_3vec = (s.end()-s.start()+1)*Ns;
  vcaxpy3(yptr, ar, xptr, yptr, n_3vec);
}

// z = a*x + y
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpAdd,
	        BinaryNode<OpMultiply, 
	        Reference< QDPType< CScal, OScalar< CScal > > >,
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
    Reference< QDPType< CScal, OScalar< CScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN;

  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().left());

  // get a and x out of the bynary node
  const OScalar< CScal >& a = static_cast<const OScalar< CScal >&>(mulNode.left());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode.right());
  // Set pointers 
  REAL *ar   = (REAL *) &(a.elem().elem().elem().real().elem());
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *zptr =          &(d.elem(s.start()).elem(0).elem(0).real().elem());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
  vcaxpy3(zptr, ar, xptr, yptr, n_3vec);
}

// z = x*a + y
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpAdd,
	        BinaryNode<OpMultiply, 
	        Reference< QDPType< TVec, OLattice< TVec > > >,
	        Reference< QDPType< CScal, OScalar< CScal > > > >,
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

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TVec, OLattice< TVec > > >,
    Reference< QDPType< CScal, OScalar< CScal > > > > BN; 


  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().left());

  // get a and x out of the bynary node
  const OScalar< CScal >& a = static_cast<const OScalar< CScal >&>(mulNode.right());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode.left());
  // Set pointers 
  REAL *ar   = (REAL *) &(a.elem().elem().elem().real().elem());
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *zptr =          &(d.elem(s.start()).elem(0).elem(0).real().elem());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
  vcaxpy3(zptr, ar, xptr, yptr, n_3vec);
}

// z = a*x - y
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpSubtract,
	        BinaryNode<OpMultiply, 
	        Reference< QDPType< CScal, OScalar< CScal > > >,
	        Reference< QDPType< TVec, OLattice< TVec > > > >,
	        Reference< QDPType< TVec, OLattice< TVec > > > >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{

#ifdef DEBUG_BLAS
  QDPIO::cout << "z = a*x - y" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&> (rhs.expression().right());

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< CScal, OScalar< CScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN;

  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().left());

  // get a and x out of the bynary node
  const OScalar< CScal >& a = static_cast<const OScalar< CScal >&>(mulNode.left());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode.right());
  // Set pointers 
  REAL *ar   = (REAL *) &(a.elem().elem().elem().real().elem());
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *zptr =          &(d.elem(s.start()).elem(0).elem(0).real().elem());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
  vcaxmy3(zptr, ar, xptr, yptr, n_3vec);
}

// z = x*a - y
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpSubtract,
	        BinaryNode<OpMultiply, 
	        Reference< QDPType< TVec, OLattice< TVec > > >,
	        Reference< QDPType< CScal, OScalar< CScal > > > >,
	        Reference< QDPType< TVec, OLattice< TVec > > > >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{

#ifdef DEBUG_BLAS
  QDPIO::cout << "z = x*a - y" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&> (rhs.expression().right());

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TVec, OLattice< TVec > > >,
    Reference< QDPType< CScal, OScalar< CScal > > > > BN; 


  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().left());

  // get a and x out of the bynary node
  const OScalar< CScal >& a = static_cast<const OScalar< CScal >&>(mulNode.right());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode.left());
  // Set pointers 
  REAL *ar   = (REAL *) &(a.elem().elem().elem().real().elem());
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *zptr =          &(d.elem(s.start()).elem(0).elem(0).real().elem());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
  vcaxmy3(zptr, ar, xptr, yptr, n_3vec);
}


// z = y + a*x
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpAdd,
	        Reference< QDPType< TVec, OLattice< TVec > > >,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< CScal, OScalar< CScal > > >,
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
    Reference< QDPType< CScal, OScalar< CScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN;

  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().right());

  // get a and x out of the bynary node
  const OScalar< CScal >& a = static_cast<const OScalar< CScal >&>(mulNode.left());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode.right());
  // Set pointers 
  REAL *ar   = (REAL *) &(a.elem().elem().elem().real().elem());
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *zptr =          &(d.elem(s.start()).elem(0).elem(0).real().elem());

  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
  vcaxpy3(zptr, ar, xptr, yptr, n_3vec);

}


// z = y + x*a
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpAdd,
	        Reference< QDPType< TVec, OLattice< TVec > > >,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< TVec, OLattice< TVec > > >,
	         Reference< QDPType< CScal, OScalar< CScal > > > > >,
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
    Reference< QDPType< CScal, OScalar< CScal > > > > BN;


  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().right());

  // get a and x out of the bynary node
  const OScalar< CScal >& a = static_cast<const OScalar< CScal >&>(mulNode.right());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode.left());
  // Set pointers 
  REAL *ar   = (REAL *) &(a.elem().elem().elem().real().elem());
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *zptr =          &(d.elem(s.start()).elem(0).elem(0).real().elem());

  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
  vcaxpy3(zptr, ar, xptr, yptr, n_3vec);

}

// z = y - a*x
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpSubtract,
	        Reference< QDPType< TVec, OLattice< TVec > > >,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< CScal, OScalar< CScal > > >,
	         Reference< QDPType< TVec, OLattice< TVec > > > > >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{

#ifdef DEBUG_BLAS
  QDPIO::cout << "z = y - a*x" << endl;
#endif

  // Peel the stuff out of the expression

  // y is the left side of rhs
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&> (rhs.expression().left());

  // ax is the right side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< CScal, OScalar< CScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN;

  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().right());

  // get a and x out of the bynary node
  const OScalar< CScal >& a = static_cast<const OScalar< CScal >&>(mulNode.left());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode.right());

  OScalar<CScal> m_a = -a;

  // Set pointers 
  REAL *ar   = (REAL *) &(m_a.elem().elem().elem().real().elem());
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *zptr =          &(d.elem(s.start()).elem(0).elem(0).real().elem());

  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
  vcaxpy3(zptr, ar, xptr, yptr, n_3vec);

}


// z = y - x*a
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpSubtract,
	        Reference< QDPType< TVec, OLattice< TVec > > >,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< TVec, OLattice< TVec > > >,
	         Reference< QDPType< CScal, OScalar< CScal > > > > >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{

#ifdef DEBUG_BLAS
  QDPIO::cout << "z = y - x*a" << endl;
#endif

  // Peel the stuff out of the expression

  // y is the left side of rhs
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&> (rhs.expression().left());

  // ax is the right side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TVec, OLattice< TVec > > >,
    Reference< QDPType< CScal, OScalar< CScal > > > > BN;


  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().right());

  // get a and x out of the bynary node
  const OScalar< CScal >& a = static_cast<const OScalar< CScal >&>(mulNode.right());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode.left());

  OScalar<CScal> m_a = -a;

  // Set pointers 
  REAL *ar   = (REAL *) &(m_a.elem().elem().elem().real().elem());
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real().elem());
  REAL *zptr =          &(d.elem(s.start()).elem(0).elem(0).real().elem());

  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_3vec = (s.end()-s.start()+1)*Ns;
  vcaxpy3(zptr, ar, xptr, yptr, n_3vec);

}

  
QDP_END_NAMESPACE();
#endif
