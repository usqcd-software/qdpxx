// $Id: qdp_scalarsite_sse_blas_g5.h,v 1.1 2005-03-17 18:52:32 bjoo Exp $

/*! @file
 * @brief Generic Scalarsite  optimization hooks
 * 
 * 
 */


#ifndef QDP_SCALARSITE_SSE_BLAS_G5_H
#define QDP_SCALARSITE_SSE_BLAS_G5_H

#include "scalarsite_sse/qdp_scalarsite_sse_blas_g5_includes.h"

using namespace QDP;

QDP_BEGIN_NAMESPACE(QDP);

// Types needed for the expression templates. 
// TVec has outer Ns template so it ought to work for staggered as well
typedef PSpinVector<PColorVector<RComplex<REAL>, 3>, Ns> TVec;
typedef PScalar<PScalar<RScalar<REAL> > >  TScal;

// #define DEBUG_BLAS_G6
// TVec is the LatticeFermion from qdp_dwdefs.h with the OLattice<> stripped
// from around it

// TScalar is the usual Real, with the OScalar<> stripped from it
//
// THis is simply to make the code more readable, and reduces < < s and > >s
// in the template arguments

// d += Scalar*ChiralProjPlus(Vec);
template<>
inline
void evaluate(OLattice< TVec >& d, 
	      const OpAddAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	                     Reference< QDPType< TScal, OScalar < TScal > > >,
	                     UnaryNode< FnChiralProjectPlus, Reference< QDPType<TVec,OLattice<TVec> > > >
>, 
	                     OLattice< TVec > > &rhs,
	      const OrderedSubset& s)
{

#ifdef DEBUG_BLAS_G5
  QDPIO::cout << "y += a*P{+}x" << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec > &>(rhs.expression().right().child());
  const OScalar< TScal >& a = static_cast<const OScalar< TScal > &> (rhs.expression().left());

  
  REAL ar = a.elem().elem().elem().elem();
  REAL* aptr = &ar;
  REAL* xptr = (REAL *)&(x.elem(s.start()).elem(0).elem(0).real());
  REAL* yptr = (REAL *)&(d.elem(s.start()).elem(0).elem(0).real());
  
  
  int n_4vec = (s.end()-s.start()+1);
  xpayz_g5ProjPlus(yptr, aptr,yptr, xptr, n_4vec);
  
}

// d += Scalar*ChiralProjMinus(Vec);
template<>
inline
void evaluate(OLattice< TVec >& d, 
	      const OpAddAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	                     Reference< QDPType< TScal, OScalar < TScal > > >,
	                     UnaryNode< FnChiralProjectMinus, Reference< QDPType<TVec,OLattice<TVec> > > >
>, 
	                     OLattice< TVec > > &rhs,
	      const OrderedSubset& s)
{

#ifdef DEBUG_BLAS_G5
  QDPIO::cout << "y += a*P{-}x" << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec > &>(rhs.expression().right().child());
  const OScalar< TScal >& a = static_cast<const OScalar< TScal > &> (rhs.expression().left());

  
  REAL ar = a.elem().elem().elem().elem();
  REAL* aptr = &ar;
  REAL* xptr = (REAL *)&(x.elem(s.start()).elem(0).elem(0).real());
  REAL* yptr = (REAL *)&(d.elem(s.start()).elem(0).elem(0).real());
  
  
  int n_4vec = (s.end()-s.start()+1);
  xpayz_g5ProjMinus(yptr, aptr,yptr, xptr, n_4vec);
  
}


// d -= Scalar*ChiralProjPlus(Vec);
template<>
inline
void evaluate(OLattice< TVec >& d, 
	      const OpSubtractAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	                     Reference< QDPType< TScal, OScalar < TScal > > >,
	                     UnaryNode< FnChiralProjectPlus, Reference< QDPType<TVec,OLattice<TVec> > > >
>, 
	                     OLattice< TVec > > &rhs,
	      const OrderedSubset& s)
{

#ifdef DEBUG_BLAS_G5
  QDPIO::cout << "y -= a*P{+}x" << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec > &>(rhs.expression().right().child());
  const OScalar< TScal >& a = static_cast<const OScalar< TScal > &> (rhs.expression().left());

  
  REAL ar = a.elem().elem().elem().elem();
  REAL* aptr = &ar;
  REAL* xptr = (REAL *)&(x.elem(s.start()).elem(0).elem(0).real());
  REAL* yptr = (REAL *)&(d.elem(s.start()).elem(0).elem(0).real());
  
  
  int n_4vec = (s.end()-s.start()+1);
  xmayz_g5ProjPlus(yptr, aptr,yptr, xptr, n_4vec);
}

// d -= Scalar*ChiralProjMinus(Vec);
template<>
inline
void evaluate(OLattice< TVec >& d, 
	      const OpSubtractAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	                     Reference< QDPType< TScal, OScalar < TScal > > >,
	                     UnaryNode< FnChiralProjectMinus, Reference< QDPType<TVec,OLattice<TVec> > > >
>, 
	                     OLattice< TVec > > &rhs,
	      const OrderedSubset& s)
{

#ifdef DEBUG_BLAS_G5
  QDPIO::cout << "y -= a*P{-}x" << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec > &>(rhs.expression().right().child());
  const OScalar< TScal >& a = static_cast<const OScalar< TScal > &> (rhs.expression().left());

  
  REAL ar = a.elem().elem().elem().elem();
  REAL* aptr = &ar;
  REAL* xptr = (REAL *)&(x.elem(s.start()).elem(0).elem(0).real());
  REAL* yptr = (REAL *)&(d.elem(s.start()).elem(0).elem(0).real());
  
  
  int n_4vec = (s.end()-s.start()+1);
  xmayz_g5ProjMinus(yptr, aptr,yptr, xptr, n_4vec);
  
}


// d += ChiralProjPlus(Vec);
template<>
inline
void evaluate(OLattice< TVec >& d, 
	      const OpAddAssign& op, 
	      const QDPExpr<
	                    UnaryNode< FnChiralProjectPlus, Reference< QDPType<TVec,OLattice<TVec> > > >, 
	                    OLattice< TVec > > &rhs,
	      const OrderedSubset& s)
{

#ifdef DEBUG_BLAS_G5
  QDPIO::cout << "y += P{+}x" << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec > &>(rhs.expression().child());

  
  REAL* xptr = (REAL *)&(x.elem(s.start()).elem(0).elem(0).real());
  REAL* yptr = (REAL *)&(d.elem(s.start()).elem(0).elem(0).real());
  
  
  int n_4vec = (s.end()-s.start()+1);
  add_g5ProjPlus(yptr, yptr, xptr, n_4vec);
  
}


// d += ChiralProjMinus(Vec);
template<>
inline
void evaluate(OLattice< TVec >& d, 
	      const OpAddAssign& op, 
	      const QDPExpr<
	                    UnaryNode< FnChiralProjectMinus, Reference< QDPType<TVec,OLattice<TVec> > > >, 
	                    OLattice< TVec > > &rhs,
	      const OrderedSubset& s)
{

#ifdef DEBUG_BLAS_G5
  QDPIO::cout << "y += P{-}x" << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec > &>(rhs.expression().child());

  
  REAL* xptr = (REAL *)&(x.elem(s.start()).elem(0).elem(0).real());
  REAL* yptr = (REAL *)&(d.elem(s.start()).elem(0).elem(0).real());
  
  
  int n_4vec = (s.end()-s.start()+1);
  add_g5ProjMinus(yptr, yptr, xptr, n_4vec);
  
}


// d -= ChiralProjPlus(Vec);
template<>
inline
void evaluate(OLattice< TVec >& d, 
	      const OpSubtractAssign& op, 
	      const QDPExpr<
	                    UnaryNode< FnChiralProjectPlus, Reference< QDPType<TVec,OLattice<TVec> > > >, 
	                    OLattice< TVec > > &rhs,
	      const OrderedSubset& s)
{

#ifdef DEBUG_BLAS_G5
  QDPIO::cout << "y -= P{+}x" << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec > &>(rhs.expression().child());

  
  REAL* xptr = (REAL *)&(x.elem(s.start()).elem(0).elem(0).real());
  REAL* yptr = (REAL *)&(d.elem(s.start()).elem(0).elem(0).real());
  
  
  int n_4vec = (s.end()-s.start()+1);
  sub_g5ProjPlus(yptr, yptr, xptr, n_4vec);
  
}


// d += ChiralProjMinus(Vec);
template<>
inline
void evaluate(OLattice< TVec >& d, 
	      const OpSubtractAssign& op, 
	      const QDPExpr<
	                    UnaryNode< FnChiralProjectMinus, Reference< QDPType<TVec,OLattice<TVec> > > >, 
	                    OLattice< TVec > > &rhs,
	      const OrderedSubset& s)
{

#ifdef DEBUG_BLAS_G5
  QDPIO::cout << "y -= P{-}x" << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec > &>(rhs.expression().child());

  
  REAL* xptr = (REAL *)&(x.elem(s.start()).elem(0).elem(0).real());
  REAL* yptr = (REAL *)&(d.elem(s.start()).elem(0).elem(0).real());
  
  
  int n_4vec = (s.end()-s.start()+1);
  sub_g5ProjMinus(yptr, yptr, xptr, n_4vec);
  
}

// d = x + a P_{+} y
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpAdd,
	                  Reference< QDPType< TVec, OLattice< TVec > > >,
	                  BinaryNode<OpMultiply, 
	                             Reference< QDPType< TScal, OScalar< TScal > > >,
	                  UnaryNode< FnChiralProjectPlus, Reference< QDPType< TVec, OLattice<TVec> > > >
                                    > 
                          >,
	                  OLattice< TVec > 
               > &rhs,
	       const OrderedSubset& s)
{
#ifdef DEBUG_BLAS_G5
  QDPIO::cout << "z = x + a*P{+} y" << endl;
#endif


  // Peel the stuff out of the expression

  // y is the left side of rhs
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&> (rhs.expression().left());

  // ax is the right side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,    
    UnaryNode< FnChiralProjectPlus, Reference< QDPType< TVec, OLattice< TVec > > > > > BN;

  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().right());

  // get a and x out of the bynary node
  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode.left());
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode.right().child());
  // Set pointers 
  REAL ar =  a.elem().elem().elem().elem();
  REAL *aptr = (REAL *)&ar;
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());

  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_4vec = (s.end()-s.start()+1);
  xpayz_g5ProjPlus(zptr, aptr, xptr, yptr, n_4vec);

}

// d = x + a P_{-} y
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpAdd,
	                  Reference< QDPType< TVec, OLattice< TVec > > >,
	                  BinaryNode<OpMultiply, 
	                             Reference< QDPType< TScal, OScalar< TScal > > >,
	                  UnaryNode< FnChiralProjectMinus, Reference< QDPType< TVec, OLattice<TVec> > > >
                                    > 
                          >,
	                  OLattice< TVec > 
               > &rhs,
	       const OrderedSubset& s)
{
#ifdef DEBUG_BLAS_G5
  QDPIO::cout << "z = x + a*P{-} y" << endl;
#endif


  // Peel the stuff out of the expression

  // y is the left side of rhs
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&> (rhs.expression().left());

  // ax is the right side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,    
    UnaryNode< FnChiralProjectMinus, Reference< QDPType< TVec, OLattice< TVec > > > > > BN;

  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().right());

  // get a and x out of the bynary node
  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode.left());
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode.right().child());
  // Set pointers 
  REAL ar =  a.elem().elem().elem().elem();
  REAL *aptr = (REAL *)&ar;
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());

  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_4vec = (s.end()-s.start()+1);
  xpayz_g5ProjMinus(zptr, aptr, xptr, yptr, n_4vec);

}

// d = x - a P_{+} y
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpSubtract,
	                  Reference< QDPType< TVec, OLattice< TVec > > >,
	                  BinaryNode<OpMultiply, 
	                             Reference< QDPType< TScal, OScalar< TScal > > >,
	                  UnaryNode< FnChiralProjectPlus, Reference< QDPType< TVec, OLattice<TVec> > > >
                                    > 
                          >,
	                  OLattice< TVec > 
               > &rhs,
	       const OrderedSubset& s)
{
#ifdef DEBUG_BLAS_G5
  QDPIO::cout << "z = x - a*P{+} y" << endl;
#endif


  // Peel the stuff out of the expression

  // y is the left side of rhs
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&> (rhs.expression().left());

  // ax is the right side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,    
    UnaryNode< FnChiralProjectPlus, Reference< QDPType< TVec, OLattice< TVec > > > > > BN;

  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().right());

  // get a and x out of the bynary node
  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode.left());
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode.right().child());
  // Set pointers 
  REAL ar =  a.elem().elem().elem().elem();
  REAL *aptr = (REAL *)&ar;
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());

  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_4vec = (s.end()-s.start()+1);
  xmayz_g5ProjPlus(zptr, aptr, xptr, yptr, n_4vec);

}

// d = x - a P_{-} y
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpSubtract,
	                  Reference< QDPType< TVec, OLattice< TVec > > >,
	                  BinaryNode<OpMultiply, 
	                             Reference< QDPType< TScal, OScalar< TScal > > >,
	                  UnaryNode< FnChiralProjectMinus, Reference< QDPType< TVec, OLattice<TVec> > > >
                                    > 
                          >,
	                  OLattice< TVec > 
               > &rhs,
	       const OrderedSubset& s)
{
#ifdef DEBUG_BLAS_G5
  QDPIO::cout << "z = x - a*P{-} y" << endl;
#endif


  // Peel the stuff out of the expression

  // y is the left side of rhs
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&> (rhs.expression().left());

  // ax is the right side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,    
    UnaryNode< FnChiralProjectMinus, Reference< QDPType< TVec, OLattice< TVec > > > > > BN;

  // get the binary node
  const BN &mulNode = static_cast<const BN&> (rhs.expression().right());

  // get a and x out of the bynary node
  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode.left());
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode.right().child());
  // Set pointers 
  REAL ar =  a.elem().elem().elem().elem();
  REAL *aptr = (REAL *)&ar;
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());

  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_4vec = (s.end()-s.start()+1);
  xmayz_g5ProjMinus(zptr, aptr, xptr, yptr, n_4vec);

}

// d = ax + P+ y
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	         BinaryNode<OpAdd,
	           BinaryNode<OpMultiply, 
	              Reference< QDPType< TScal, OScalar< TScal > > >,
	              Reference< QDPType< TVec, OLattice< TVec  > > > 
                   >,
	           UnaryNode<FnChiralProjectPlus, Reference< QDPType<TVec, OLattice<TVec> > > >
                 >,
	        OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{
#ifdef DEBUG_BLAS_G5
  QDPIO::cout << "z = a*x + P{+} y" << endl;
#endif


  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&> (rhs.expression().right().child());

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
  REAL ar =  a.elem().elem().elem().elem();
  REAL *aptr = (REAL *)&ar;
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_4vec = (s.end()-s.start()+1);
  axpyz_g5ProjPlus(zptr, aptr, xptr, yptr, n_4vec);
}

// d = ax + P- y
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	         BinaryNode<OpAdd,
	           BinaryNode<OpMultiply, 
	              Reference< QDPType< TScal, OScalar< TScal > > >,
	              Reference< QDPType< TVec, OLattice< TVec  > > > 
                   >,
	           UnaryNode<FnChiralProjectMinus, Reference< QDPType<TVec, OLattice<TVec> > > >
                 >,
	        OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{
#ifdef DEBUG_BLAS_G5
  QDPIO::cout << "z = a*x + P{-} y" << endl;
#endif


  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&> (rhs.expression().right().child());

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
  REAL ar =  a.elem().elem().elem().elem();
  REAL *aptr = (REAL *)&ar;
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_4vec = (s.end()-s.start()+1);
  axpyz_g5ProjMinus(zptr, aptr, xptr, yptr, n_4vec);
}


// d = ax - P+ y
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	         BinaryNode<OpSubtract,
	           BinaryNode<OpMultiply, 
	              Reference< QDPType< TScal, OScalar< TScal > > >,
	              Reference< QDPType< TVec, OLattice< TVec  > > > 
                   >,
	           UnaryNode<FnChiralProjectPlus, Reference< QDPType<TVec, OLattice<TVec> > > >
                 >,
	        OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{
#ifdef DEBUG_BLAS_G5
  QDPIO::cout << "z = a*x + P{+} y" << endl;
#endif


  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&> (rhs.expression().right().child());

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
  REAL ar =  a.elem().elem().elem().elem();
  REAL *aptr = (REAL *)&ar;
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_4vec = (s.end()-s.start()+1);
  axmyz_g5ProjPlus(zptr, aptr, xptr, yptr, n_4vec);
}

// d = ax - P- y
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	         BinaryNode<OpSubtract,
	           BinaryNode<OpMultiply, 
	              Reference< QDPType< TScal, OScalar< TScal > > >,
	              Reference< QDPType< TVec, OLattice< TVec  > > > 
                   >,
	           UnaryNode<FnChiralProjectMinus, Reference< QDPType<TVec, OLattice<TVec> > > >
                 >,
	        OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{
#ifdef DEBUG_BLAS_G5
  QDPIO::cout << "z = a*x + P{-} y" << endl;
#endif


  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&> (rhs.expression().right().child());

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
  REAL ar =  a.elem().elem().elem().elem();
  REAL *aptr = (REAL *)&ar;
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_4vec = (s.end()-s.start()+1);
  axmyz_g5ProjMinus(zptr, aptr, xptr, yptr, n_4vec);
}

// Vec = Scal * P_{+} Vec
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpMultiply,
	       Reference< QDPType< TScal, OScalar< TScal > > >,
	       UnaryNode< FnChiralProjectPlus, Reference< QDPType< TVec, OLattice< TVec > > > >
               >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{

#ifdef DEBUG_BLAS_G5
  cout << "BJ: v = a*P{+}v " << endl;
#endif

  const OLattice< TVec > &x = static_cast<const OLattice< TVec >&>(rhs.expression().right().child());
  const OScalar< TScal > &a = static_cast<const OScalar< TScal >&>(rhs.expression().left());

  REAL ar =  a.elem().elem().elem().elem();
  REAL *aptr = &ar;  
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL *zptr =  &(d.elem(s.start()).elem(0).elem(0).real());
  int n_4vec = (s.end()-s.start()+1);

  scal_g5ProjPlus(zptr, aptr, xptr, n_4vec);
}

// Vec = Scal * P_{-} Vec
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpMultiply,
	       Reference< QDPType< TScal, OScalar< TScal > > >,
	       UnaryNode< FnChiralProjectMinus, Reference< QDPType< TVec, OLattice< TVec > > > >
               >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{

#ifdef DEBUG_BLAS_G5
  cout << "BJ: v = a*P{-}v " << endl;
#endif

  const OLattice< TVec > &x = static_cast<const OLattice< TVec >&>(rhs.expression().right().child());
  const OScalar< TScal > &a = static_cast<const OScalar< TScal >&>(rhs.expression().left());

  REAL ar =  a.elem().elem().elem().elem();
  REAL *aptr = &ar;  
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL *zptr =  &(d.elem(s.start()).elem(0).elem(0).real());
  int n_4vec = (s.end()-s.start()+1);

  scal_g5ProjMinus(zptr, aptr, xptr, n_4vec);
}

// z = ax + bP+ y
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
	         UnaryNode< FnChiralProjectPlus, Reference< QDPType< TVec, OLattice< TVec > > > >
                > 
               >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{

#ifdef DEBUG_BLAS_G5
  QDPIO::cout << "z = a*x + b*P+y" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN1;

  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,
    UnaryNode< FnChiralProjectPlus, Reference< QDPType< TVec, OLattice< TVec > > > > > BN2;

  // get the binary node
  const BN1 &mulNode1 = static_cast<const BN1&> (rhs.expression().left());
  const BN2 &mulNode2 = static_cast<const BN2&> (rhs.expression().right());

  // get a and x out of the binary node
  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode1.left());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode1.right());
  
  // get b and y out of the binary node
  const OScalar< TScal >& b = static_cast<const OScalar< TScal >&>(mulNode2.left());
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode2.right().child());

  
  // Set pointers 
  REAL *aptr = (REAL *)&(a.elem().elem().elem().elem());
  REAL *bptr = (REAL *)&(b.elem().elem().elem().elem());
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_4vec = (s.end()-s.start()+1);
  axpbyz_g5ProjPlus(zptr, aptr, xptr, bptr, yptr, n_4vec);
}

// z = ax + bP- y
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
	         UnaryNode< FnChiralProjectMinus, Reference< QDPType< TVec, OLattice< TVec > > > >
                > 
               >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{

#ifdef DEBUG_BLAS_G5
  QDPIO::cout << "z = a*x + b*P-y" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN1;

  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,
    UnaryNode< FnChiralProjectMinus, Reference< QDPType< TVec, OLattice< TVec > > > > > BN2;

  // get the binary node
  const BN1 &mulNode1 = static_cast<const BN1&> (rhs.expression().left());
  const BN2 &mulNode2 = static_cast<const BN2&> (rhs.expression().right());

  // get a and x out of the binary node
  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode1.left());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode1.right());
  
  // get b and y out of the binary node
  const OScalar< TScal >& b = static_cast<const OScalar< TScal >&>(mulNode2.left());
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode2.right().child());

  
  // Set pointers 
  REAL *aptr = (REAL *)&(a.elem().elem().elem().elem());
  REAL *bptr = (REAL *)&(b.elem().elem().elem().elem());
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_4vec = (s.end()-s.start()+1);
  axpbyz_g5ProjMinus(zptr, aptr, xptr, bptr, yptr, n_4vec);
}

// z = ax - bP+ y
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
	         UnaryNode< FnChiralProjectPlus, Reference< QDPType< TVec, OLattice< TVec > > > >
                > 
               >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{

#ifdef DEBUG_BLAS_G5
  QDPIO::cout << "z = a*x - b*P+y" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN1;

  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,
    UnaryNode< FnChiralProjectPlus, Reference< QDPType< TVec, OLattice< TVec > > > > > BN2;

  // get the binary node
  const BN1 &mulNode1 = static_cast<const BN1&> (rhs.expression().left());
  const BN2 &mulNode2 = static_cast<const BN2&> (rhs.expression().right());

  // get a and x out of the binary node
  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode1.left());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode1.right());
  
  // get b and y out of the binary node
  const OScalar< TScal >& b = static_cast<const OScalar< TScal >&>(mulNode2.left());
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode2.right().child());

  
  // Set pointers 
  REAL *aptr = (REAL *)&(a.elem().elem().elem().elem());
  REAL *bptr = (REAL *)&(b.elem().elem().elem().elem());
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_4vec = (s.end()-s.start()+1);
  axmbyz_g5ProjPlus(zptr, aptr, xptr, bptr, yptr, n_4vec);
}

// z = ax - bP- y
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
	         UnaryNode< FnChiralProjectMinus, Reference< QDPType< TVec, OLattice< TVec > > > >
                > 
               >,
	       OLattice< TVec > > &rhs,
	       const OrderedSubset& s)
{

#ifdef DEBUG_BLAS_G5
  QDPIO::cout << "z = a*x - b*P-y" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN1;

  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TScal, OScalar< TScal > > >,
    UnaryNode< FnChiralProjectMinus, Reference< QDPType< TVec, OLattice< TVec > > > > > BN2;

  // get the binary node
  const BN1 &mulNode1 = static_cast<const BN1&> (rhs.expression().left());
  const BN2 &mulNode2 = static_cast<const BN2&> (rhs.expression().right());

  // get a and x out of the binary node
  const OScalar< TScal >& a = static_cast<const OScalar< TScal >&>(mulNode1.left());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode1.right());
  
  // get b and y out of the binary node
  const OScalar< TScal >& b = static_cast<const OScalar< TScal >&>(mulNode2.left());
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode2.right().child());

  
  // Set pointers 
  REAL *aptr = (REAL *)&(a.elem().elem().elem().elem());
  REAL *bptr = (REAL *)&(b.elem().elem().elem().elem());
  REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
  REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
  REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());


  // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
  int n_4vec = (s.end()-s.start()+1);
  axmbyz_g5ProjMinus(zptr, aptr, xptr, bptr, yptr, n_4vec);
}


QDP_END_NAMESPACE();

#endif  // guard
 
