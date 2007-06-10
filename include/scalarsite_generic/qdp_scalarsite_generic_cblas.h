#ifndef QDP_SCALARSITE_GENERIC_CBLAS
#define QDP_SCALARSITE_GENERIC_CBLAS

// Complex BLAS routines

#include "scalarsite_generic/generic_blas_vcscal.h"
#include "scalarsite_generic/generic_blas_vcaxpy3.h"
#include "scalarsite_generic/generic_blas_vcaxmy3.h"
#include "scalarsite_generic/generic_blas_vcaxpby3.h"
#include "scalarsite_generic/generic_blas_vcaxmby3.h"

namespace QDP {

typedef PScalar<PScalar<RComplex<REAL> > >  CScal;
typedef PSpinVector<PColorVector<RComplex<REAL>, 3>, 4> TVec;

// vector z *= complex a
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpMultiplyAssign &op,
	       const QDPExpr< 
	       UnaryNode<OpIdentity,
	       Reference< QDPType< CScal, OScalar< CScal > > > >,
	       OScalar< CScal > > &rhs,
	       const Subset& s)
{
  const OScalar< CScal >& a = static_cast< const OScalar<CScal >&>(rhs.expression().child()); 

#ifdef DEBUG_CBLAS  
  QDPIO::cout << "BJ: Complex v *= a " << a << endl;
#endif
  
  REAL *a_start = (REAL *) &(a.elem().elem().elem().real());
  
  if( s.hasOrderedRep() ) { 
    REAL *d_start = &(d.elem(s.start()).elem(0).elem(0).real());

    int n_3vec =( s.end() - s.start() + 1 )*Ns;
    vcscal(d_start, a_start, d_start, n_3vec);
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];
      REAL *d_start = &(d.elem(i).elem(0).elem(0).real());

      vcscal(d_start, a_start, d_start, 4);
    }
  }

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
	       const Subset& s)
{
  const OScalar< CScal >& a = static_cast< const OScalar<CScal >&>(rhs.expression().left()); 
  
  const OLattice< TVec > &x = static_cast<const OLattice< TVec >&>(rhs.expression().right());

#ifdef DEBUG_CBLAS
  QDPIO::cout << "BJ: Complex v = a*x " << a << endl;
#endif

  REAL *a_start = (REAL *) &(a.elem().elem().elem().real());

  if( s.hasOrderedRep() ) {
  
    REAL *d_start = &(d.elem(s.start()).elem(0).elem(0).real());
    REAL *x_start = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());

  
    int n_3vec =( s.end() - s.start() + 1 )*Ns;

    vcscal(d_start, a_start, x_start, n_3vec);
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];
   
      REAL *d_start = &(d.elem(i).elem(0).elem(0).real());
      REAL *x_start = (REAL *) &(x.elem(i).elem(0).elem(0).real());
      
  
      vcscal(d_start, a_start, x_start, 4);
    }
  }
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
	       const Subset& s)
{
  const OScalar< CScal >& a = static_cast< const OScalar<CScal >&>(rhs.expression().right()); 
  
  const OLattice< TVec > &x = static_cast<const OLattice< TVec >&>(rhs.expression().left());

#ifdef DEBUG_CBLAS
  QDPIO::cout << "BJ: Complex v = x*a " << a << endl;
#endif


  REAL *a_start = (REAL *) &(a.elem().elem().elem().real());

  if( s.hasOrderedRep() ) { 
    REAL *d_start = &(d.elem(s.start()).elem(0).elem(0).real());
    REAL *x_start = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
    
    int n_3vec =( s.end() - s.start() + 1 )*Ns;
    
    vcscal(d_start, a_start, x_start, n_3vec);
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];
      REAL *d_start = &(d.elem(i).elem(0).elem(0).real());
      REAL *x_start = (REAL *) &(x.elem(i).elem(0).elem(0).real());
      
      vcscal(d_start, a_start, x_start, 4);
    }
  }
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
	      const Subset& s)
{

#ifdef DEBUG_CBLAS
  QDPIO::cout << "y += a*x" << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec > &>(rhs.expression().right());
  const OScalar< CScal >& a = static_cast<const OScalar< CScal > &> (rhs.expression().left());
  
  REAL* ar   = (REAL *)&(a.elem().elem().elem().real());
  if( s.hasOrderedRep() ) { 

    REAL* xptr = (REAL *)&(x.elem(s.start()).elem(0).elem(0).real());
    REAL* yptr = &(d.elem(s.start()).elem(0).elem(0).real());

    int n_3vec = (s.end()-s.start()+1)*Ns;
    vcaxpy3(yptr, ar, xptr, yptr, n_3vec);
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];

      REAL* xptr = (REAL *)&(x.elem(i).elem(0).elem(0).real());
      REAL* yptr = &(d.elem(i).elem(0).elem(0).real());

      vcaxpy3(yptr, ar, xptr, yptr, 4);

    }
  }
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
	      const Subset& s)
{

#ifdef DEBUG_CBLAS
  QDPIO::cout << "y += x*a" << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec > &>(rhs.expression().left());
  const OScalar< CScal >& a = static_cast<const OScalar< CScal > &> (rhs.expression().right());
  
  REAL* ar   = (REAL *)&(a.elem().elem().elem().real());

  if( s.hasOrderedRep()) { 
    REAL* xptr = (REAL *)&(x.elem(s.start()).elem(0).elem(0).real());
    REAL* yptr = &(d.elem(s.start()).elem(0).elem(0).real());
  
    int n_3vec = (s.end()-s.start()+1)*Ns;
    vcaxpy3(yptr, ar, xptr, yptr, n_3vec);
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];

      REAL* xptr = (REAL *)&(x.elem(i).elem(0).elem(0).real());
      REAL* yptr = &(d.elem(i).elem(0).elem(0).real());

      vcaxpy3(yptr, ar, xptr, yptr, 4);

    }
  }
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
	      const Subset& s)
{

#ifdef DEBUG_CBLAS
  QDPIO::cout << "y -= a*x" << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec > &>(rhs.expression().right());
  const OScalar< CScal >& a = static_cast<const OScalar< CScal > &> (rhs.expression().left());

  // Get minus a
  OScalar<CScal> m_a = -a;

  REAL* ar   = (REAL *)&(m_a.elem().elem().elem().real());
  if( s.hasOrderedRep() ) { 
    REAL* xptr = (REAL *)&(x.elem(s.start()).elem(0).elem(0).real());
    REAL* yptr = &(d.elem(s.start()).elem(0).elem(0).real());
    // cout << "Specialised axpy a ="<< ar << endl;
    
    int n_3vec = (s.end()-s.start()+1)*Ns;
    vcaxpy3(yptr, ar, xptr, yptr, n_3vec);
  }  
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];

      REAL* xptr = (REAL *)&(x.elem(i).elem(0).elem(0).real());
      REAL* yptr = &(d.elem(i).elem(0).elem(0).real());

      vcaxpy3(yptr, ar, xptr, yptr, 4);

    }
  }
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
	      const Subset& s)
{
#ifdef DEBUG_CBLAS
  QDPIO::cout << "y -= x*a" << endl;
#endif

  const OLattice< TVec >& x = static_cast<const OLattice< TVec > &>(rhs.expression().left());
  const OScalar< CScal >& a = static_cast<const OScalar< CScal > &> (rhs.expression().right());

  // Get minus a
  OScalar<CScal> m_a = -a;
  
  REAL* ar   = (REAL *)&(m_a.elem().elem().elem().real());

  if( s.hasOrderedRep()  ) { 
    REAL* xptr = (REAL *)&(x.elem(s.start()).elem(0).elem(0).real());
    REAL* yptr = &(d.elem(s.start()).elem(0).elem(0).real());
    // cout << "Specialised axpy a ="<< ar << endl;
    
    int n_3vec = (s.end()-s.start()+1)*Ns;
    vcaxpy3(yptr, ar, xptr, yptr, n_3vec);
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];
      
      REAL* xptr = (REAL *)&(x.elem(i).elem(0).elem(0).real());
      REAL* yptr = &(d.elem(i).elem(0).elem(0).real());

      vcaxpy3(yptr, ar, xptr, yptr, 4);

    }
  }
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
	       const Subset& s)
{

#ifdef DEBUG_CBLAS
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
  REAL *ar   = (REAL *) &(a.elem().elem().elem().real());

  if( s.hasOrderedRep() ) { 

    REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
    REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
    REAL *zptr =          &(d.elem(s.start()).elem(0).elem(0).real());


    // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
    int n_3vec = (s.end()-s.start()+1)*Ns;
    vcaxpy3(zptr, ar, xptr, yptr, n_3vec);
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];

      REAL *xptr = (REAL *) &(x.elem(i).elem(0).elem(0).real());
      REAL *yptr = (REAL *) &(y.elem(i).elem(0).elem(0).real());
      REAL *zptr =          &(d.elem(i).elem(0).elem(0).real());


      // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
      vcaxpy3(zptr, ar, xptr, yptr, 4);

    }
  }
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
	       const Subset& s)
{

#ifdef DEBUG_CBLAS
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
  REAL *ar   = (REAL *) &(a.elem().elem().elem().real());
  
  if( s.hasOrderedRep() ) { 
    REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
    REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
    REAL *zptr =          &(d.elem(s.start()).elem(0).elem(0).real());
    
    
    // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
    int n_3vec = (s.end()-s.start()+1)*Ns;
    vcaxpy3(zptr, ar, xptr, yptr, n_3vec);
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];

      REAL *xptr = (REAL *) &(x.elem(i).elem(0).elem(0).real());
      REAL *yptr = (REAL *) &(y.elem(i).elem(0).elem(0).real());
      REAL *zptr =          &(d.elem(i).elem(0).elem(0).real());


      // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
      vcaxpy3(zptr, ar, xptr, yptr, 4);

    }
  }
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
	       const Subset& s)
{

#ifdef DEBUG_CBLAS
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
  REAL *ar   = (REAL *) &(a.elem().elem().elem().real());
  
  if( s.hasOrderedRep() ) { 
    REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
    REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
    REAL *zptr =          &(d.elem(s.start()).elem(0).elem(0).real());
    
    
    // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
    int n_3vec = (s.end()-s.start()+1)*Ns;
    vcaxmy3(zptr, ar, xptr, yptr, n_3vec);
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];

      REAL *xptr = (REAL *) &(x.elem(i).elem(0).elem(0).real());
      REAL *yptr = (REAL *) &(y.elem(i).elem(0).elem(0).real());
      REAL *zptr =          &(d.elem(i).elem(0).elem(0).real());


      // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
      vcaxmy3(zptr, ar, xptr, yptr, 4);

    }
  }
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
	       const Subset& s)
{

#ifdef DEBUG_CBLAS
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
  REAL *ar   = (REAL *) &(a.elem().elem().elem().real());

  if( s.hasOrderedRep() ) { 
    REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
    REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
    REAL *zptr =          &(d.elem(s.start()).elem(0).elem(0).real());
    

    // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
    int n_3vec = (s.end()-s.start()+1)*Ns;
    vcaxmy3(zptr, ar, xptr, yptr, n_3vec);
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];
      
      REAL *xptr = (REAL *) &(x.elem(i).elem(0).elem(0).real());
      REAL *yptr = (REAL *) &(y.elem(i).elem(0).elem(0).real());
      REAL *zptr =          &(d.elem(i).elem(0).elem(0).real());
      
      
      // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
      vcaxmy3(zptr, ar, xptr, yptr, 4);

    }
  }
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
	       const Subset& s)
{

#ifdef DEBUG_CBLAS
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
  REAL *ar   = (REAL *) &(a.elem().elem().elem().real());
  if( s.hasOrderedRep()) { 
    REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
    REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
    REAL *zptr =          &(d.elem(s.start()).elem(0).elem(0).real());
    
    // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
    int n_3vec = (s.end()-s.start()+1)*Ns;
    vcaxpy3(zptr, ar, xptr, yptr, n_3vec);
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];
      
      REAL *xptr = (REAL *) &(x.elem(i).elem(0).elem(0).real());
      REAL *yptr = (REAL *) &(y.elem(i).elem(0).elem(0).real());
      REAL *zptr =          &(d.elem(i).elem(0).elem(0).real());
      
      
      // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
      vcaxpy3(zptr, ar, xptr, yptr, 4);
      
    }
  }
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
	       const Subset& s)
{

#ifdef DEBUG_CBLAS
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
  REAL *ar   = (REAL *) &(a.elem().elem().elem().real());

  if( s.hasOrderedRep() ) { 
    REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
    REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
    REAL *zptr =          &(d.elem(s.start()).elem(0).elem(0).real());
    
    // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
    int n_3vec = (s.end()-s.start()+1)*Ns;
    vcaxpy3(zptr, ar, xptr, yptr, n_3vec);
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];
      
      REAL *xptr = (REAL *) &(x.elem(i).elem(0).elem(0).real());
      REAL *yptr = (REAL *) &(y.elem(i).elem(0).elem(0).real());
      REAL *zptr =          &(d.elem(i).elem(0).elem(0).real());
      
      
      // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
      vcaxpy3(zptr, ar, xptr, yptr, 4);
      
    }
  }
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
	       const Subset& s)
{

#ifdef DEBUG_CBLAS
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
  REAL *ar   = (REAL *) &(m_a.elem().elem().elem().real());
  
  if( s.hasOrderedRep() ) { 
    REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
    REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
    REAL *zptr =          &(d.elem(s.start()).elem(0).elem(0).real());
    
    // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
    int n_3vec = (s.end()-s.start()+1)*Ns;
    vcaxpy3(zptr, ar, xptr, yptr, n_3vec);
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];
      
      REAL *xptr = (REAL *) &(x.elem(i).elem(0).elem(0).real());
      REAL *yptr = (REAL *) &(y.elem(i).elem(0).elem(0).real());
      REAL *zptr =          &(d.elem(i).elem(0).elem(0).real());
      
      
      // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
      vcaxpy3(zptr, ar, xptr, yptr, 4); 
            
    }
  }
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
	       const Subset& s)
{

#ifdef DEBUG_CBLAS
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
  REAL *ar   = (REAL *) &(m_a.elem().elem().elem().real());
  
  if( s.hasOrderedRep() ) { 
    REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
    REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
    REAL *zptr =          &(d.elem(s.start()).elem(0).elem(0).real());
    
    // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
    int n_3vec = (s.end()-s.start()+1)*Ns;
    vcaxpy3(zptr, ar, xptr, yptr, n_3vec);
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];
      
      REAL *xptr = (REAL *) &(x.elem(i).elem(0).elem(0).real());
      REAL *yptr = (REAL *) &(y.elem(i).elem(0).elem(0).real());
      REAL *zptr =          &(d.elem(i).elem(0).elem(0).real());
      
      
      // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
      vcaxpy3(zptr, ar, xptr, yptr, 4);
            
    }
  }
}


// z = ax + by
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpAdd,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< CScal, OScalar< CScal > > >,
	         Reference< QDPType< TVec, OLattice< TVec > > > >,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< CScal, OScalar< CScal > > >,
	         Reference< QDPType< TVec, OLattice< TVec > > > > >,
	        OLattice< TVec > > &rhs,
	       const Subset& s)
{

#ifdef DEBUG_CBLAS
  QDPIO::cout << "z = a*x + b*y" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< CScal, OScalar< CScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN;

  // get the binary node
  const BN &mulNode1 = static_cast<const BN&> (rhs.expression().left());
  const BN &mulNode2 = static_cast<const BN&> (rhs.expression().right());

  // get a and x out of the binary node
  const OScalar< CScal >& a = static_cast<const OScalar< CScal >&>(mulNode1.left());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode1.right());
  
  // get b and y out of the binary node
  const OScalar< CScal >& b = static_cast<const OScalar< CScal >&>(mulNode2.left());
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode2.right());

  
  // Set pointers 
  REAL *aptr = (REAL *)&(a.elem().elem().elem().real());
  REAL *bptr = (REAL *)&(b.elem().elem().elem().real());

  if( s.hasOrderedRep() ) { 

    REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
    REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
    REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());
    
    
    // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
    int n_3vec = (s.end()-s.start() + 1)*Ns;
    vcaxpby3(zptr, aptr, xptr, bptr, yptr, n_3vec);
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];
      
      REAL *xptr = (REAL *) &(x.elem(i).elem(0).elem(0).real());
      REAL *yptr = (REAL *) &(y.elem(i).elem(0).elem(0).real());
      REAL *zptr =          &(d.elem(i).elem(0).elem(0).real());
      
      
      // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
      vcaxpby3(zptr, aptr, xptr, bptr, yptr, 4);  
            
    }
  }
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
	         Reference< QDPType< CScal, OScalar< CScal > > > >,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< CScal, OScalar< CScal > > >,
	         Reference< QDPType< TVec, OLattice< TVec > > > > >,
	        OLattice< TVec > > &rhs,
	       const Subset& s)
{

#ifdef DEBUG_CBLAS
  QDPIO::cout << "z = x*a + b*y" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TVec, OLattice< TVec > > >,
    Reference< QDPType< CScal, OScalar< CScal > > > > BN1;

  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< CScal, OScalar< CScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN2;


  
  // get the binary node
  const BN1 &mulNode1 = static_cast<const BN1&> (rhs.expression().left());
  const BN2 &mulNode2 = static_cast<const BN2&> (rhs.expression().right());

  // get a and x out of the binary node
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode1.left());

  const OScalar< CScal >& a = static_cast<const OScalar< CScal >&>(mulNode1.right());
  
  // get b and y out of the binary node
  const OScalar< CScal >& b = static_cast<const OScalar< CScal >&>(mulNode2.left());
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode2.right());

  
  // Set pointers 
  REAL *aptr = (REAL *)&(a.elem().elem().elem().real());
  REAL *bptr = (REAL *)&(b.elem().elem().elem().real());
  if( s.hasOrderedRep() ) { 

    REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
    REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
    REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());
    
    
    // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
    int n_3vec = (s.end()-s.start()+1)*Ns;
    vcaxpby3(zptr, aptr, xptr, bptr, yptr, n_3vec);
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];
      
      REAL *xptr = (REAL *) &(x.elem(i).elem(0).elem(0).real());
      REAL *yptr = (REAL *) &(y.elem(i).elem(0).elem(0).real());
      REAL *zptr =          &(d.elem(i).elem(0).elem(0).real());
      
      
      // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
      vcaxpby3(zptr, aptr, xptr, bptr, yptr, 4);  
            
    }
  }
}

// z = ax + yb
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpAdd,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< CScal, OScalar< CScal > > >,
	         Reference< QDPType< TVec, OLattice< TVec > > > >,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< TVec, OLattice< TVec > > >,
	         Reference< QDPType< CScal, OScalar< CScal > > > > >,
	        OLattice< TVec > > &rhs,
	       const Subset& s)
{

#ifdef DEBUG_CBLAS
  QDPIO::cout << "z = a*x + y*b" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs

  // type of a*x
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< CScal, OScalar< CScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN1;

  // type of y*b
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TVec, OLattice< TVec > > >,
    Reference< QDPType< CScal, OScalar< CScal > > > > BN2;


  
  // get the binary nodes
  // a*x node
  const BN1 &mulNode1 = static_cast<const BN1&> (rhs.expression().left());

  // y*b node
  const BN2 &mulNode2 = static_cast<const BN2&> (rhs.expression().right());

  // get a and x out of the binary node
  const OScalar< CScal >& a = static_cast<const OScalar< CScal >&>(mulNode1.left());

  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode1.right());

  
  // get b and y out of the binary node
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode2.left());

  const OScalar< CScal >& b = static_cast<const OScalar< CScal >&>(mulNode2.right());

  
  // Set pointers 
  REAL *aptr = (REAL *)&(a.elem().elem().elem().real());
  REAL *bptr = (REAL *)&(b.elem().elem().elem().real());

  if( s.hasOrderedRep() ) { 
    REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
    REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
    REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());
    
    
    // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
    int n_3vec = (s.end()-s.start()+1)*Ns;
    vcaxpby3(zptr, aptr, xptr, bptr, yptr, n_3vec);
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];
      
      REAL *xptr = (REAL *) &(x.elem(i).elem(0).elem(0).real());
      REAL *yptr = (REAL *) &(y.elem(i).elem(0).elem(0).real());
      REAL *zptr =          &(d.elem(i).elem(0).elem(0).real());
      
      
      // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
      vcaxpby3(zptr, aptr, xptr, bptr, yptr, 4);  
            
    }
  }
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
	         Reference< QDPType< CScal, OScalar< CScal > > > >,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< TVec, OLattice< TVec > > >,
	         Reference< QDPType< CScal, OScalar< CScal > > > > >,
	        OLattice< TVec > > &rhs,
	       const Subset& s)
{

#ifdef DEBUG_CBLAS
  QDPIO::cout << "z = x*a + y*b" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TVec, OLattice< TVec > > >,
    Reference< QDPType< CScal, OScalar< CScal > > > > BN;

  // get the binary node
  const BN &mulNode1 = static_cast<const BN&> (rhs.expression().left());
  const BN &mulNode2 = static_cast<const BN&> (rhs.expression().right());

  // get a and x out of the binary node
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode1.left());
  const OScalar< CScal >& a = static_cast<const OScalar< CScal >&>(mulNode1.right());
  
  // get b and y out of the binary node
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode2.left());

  const OScalar< CScal >& b = static_cast<const OScalar< CScal >&>(mulNode2.right());
  
  // Set pointers 
  REAL *aptr = (REAL *)&(a.elem().elem().elem().real());
  REAL *bptr = (REAL *)&(b.elem().elem().elem().real());
  
  if( s.hasOrderedRep() ) { 
    REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
    REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
    REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());
    
    
    // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
    int n_3vec = (s.end()-s.start()+1)*Ns;
    vcaxpby3(zptr, aptr, xptr, bptr, yptr, n_3vec);
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];
      
      REAL *xptr = (REAL *) &(x.elem(i).elem(0).elem(0).real());
      REAL *yptr = (REAL *) &(y.elem(i).elem(0).elem(0).real());
      REAL *zptr =          &(d.elem(i).elem(0).elem(0).real());
      
      
      // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
      vcaxpby3(zptr, aptr, xptr, bptr, yptr, 4);  
            
    }
  }
}

// z = ax - by
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpSubtract,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< CScal, OScalar< CScal > > >,
	         Reference< QDPType< TVec, OLattice< TVec > > > >,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< CScal, OScalar< CScal > > >,
	         Reference< QDPType< TVec, OLattice< TVec > > > > >,
	        OLattice< TVec > > &rhs,
	       const Subset& s)
{

#ifdef DEBUG_CBLAS
  QDPIO::cout << "z = a*x - b*y" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< CScal, OScalar< CScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN;

  // get the binary node
  const BN &mulNode1 = static_cast<const BN&> (rhs.expression().left());
  const BN &mulNode2 = static_cast<const BN&> (rhs.expression().right());

  // get a and x out of the binary node
  const OScalar< CScal >& a = static_cast<const OScalar< CScal >&>(mulNode1.left());
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode1.right());
  
  // get b and y out of the binary node
  const OScalar< CScal >& b = static_cast<const OScalar< CScal >&>(mulNode2.left());
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode2.right());

  
  // Set pointers 
  REAL *aptr = (REAL *)&(a.elem().elem().elem().real());
  REAL *bptr = (REAL *)&(b.elem().elem().elem().real());
  
  if( s.hasOrderedRep() ) { 
    REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
    REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
    REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());
    
    
    // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
    int n_3vec = (s.end()-s.start()+1)*Ns;
    vcaxmby3(zptr, aptr, xptr, bptr, yptr, n_3vec);
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];
      
      REAL *xptr = (REAL *) &(x.elem(i).elem(0).elem(0).real());
      REAL *yptr = (REAL *) &(y.elem(i).elem(0).elem(0).real());
      REAL *zptr =          &(d.elem(i).elem(0).elem(0).real());
      
      
      // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
      vcaxmby3(zptr, aptr, xptr, bptr, yptr, 4);
            
    }
  } 
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
	         Reference< QDPType< CScal, OScalar< CScal > > > >,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< CScal, OScalar< CScal > > >,
	         Reference< QDPType< TVec, OLattice< TVec > > > > >,
	        OLattice< TVec > > &rhs,
	       const Subset& s)
{

#ifdef DEBUG_CBLAS
  QDPIO::cout << "z = x*a - b*y" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TVec, OLattice< TVec > > >,
    Reference< QDPType< CScal, OScalar< CScal > > > > BN1;

  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< CScal, OScalar< CScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN2;


  
  // get the binary node
  const BN1 &mulNode1 = static_cast<const BN1&> (rhs.expression().left());
  const BN2 &mulNode2 = static_cast<const BN2&> (rhs.expression().right());

  // get a and x out of the binary node
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode1.left());

  const OScalar< CScal >& a = static_cast<const OScalar< CScal >&>(mulNode1.right());
  
  // get b and y out of the binary node
  const OScalar< CScal >& b = static_cast<const OScalar< CScal >&>(mulNode2.left());
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode2.right());

  
  // Set pointers 
  REAL *aptr = (REAL *)&(a.elem().elem().elem().real());
  REAL *bptr = (REAL *)&(b.elem().elem().elem().real());

  if( s.hasOrderedRep() ) { 
    REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
    REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
    REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());
    
    
    // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
    int n_3vec = (s.end()-s.start()+1)*Ns;
    vcaxmby3(zptr, aptr, xptr, bptr, yptr, n_3vec);
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];
      
      REAL *xptr = (REAL *) &(x.elem(i).elem(0).elem(0).real());
      REAL *yptr = (REAL *) &(y.elem(i).elem(0).elem(0).real());
      REAL *zptr =          &(d.elem(i).elem(0).elem(0).real());
      
      
      // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
      vcaxmby3(zptr, aptr, xptr, bptr, yptr, 4);
            
    }
  }
}

// z = ax - yb
template<>
inline
void evaluate( OLattice< TVec > &d,
	       const OpAssign &op,
	       const QDPExpr< 
	       BinaryNode<OpSubtract,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< CScal, OScalar< CScal > > >,
	         Reference< QDPType< TVec, OLattice< TVec > > > >,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< TVec, OLattice< TVec > > >,
	         Reference< QDPType< CScal, OScalar< CScal > > > > >,
	        OLattice< TVec > > &rhs,
	       const Subset& s)
{

#ifdef DEBUG_CBLAS
  QDPIO::cout << "z = a*x - y*b" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs

  // type of a*x
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< CScal, OScalar< CScal > > >,
    Reference< QDPType< TVec, OLattice< TVec > > > > BN1;

  // type of y*b
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TVec, OLattice< TVec > > >,
    Reference< QDPType< CScal, OScalar< CScal > > > > BN2;


  
  // get the binary nodes
  // a*x node
  const BN1 &mulNode1 = static_cast<const BN1&> (rhs.expression().left());

  // y*b node
  const BN2 &mulNode2 = static_cast<const BN2&> (rhs.expression().right());

  // get a and x out of the binary node
  const OScalar< CScal >& a = static_cast<const OScalar< CScal >&>(mulNode1.left());

  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode1.right());

  
  // get b and y out of the binary node
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode2.left());

  const OScalar< CScal >& b = static_cast<const OScalar< CScal >&>(mulNode2.right());

  
  // Set pointers 
  REAL *aptr = (REAL *)&(a.elem().elem().elem().real());
  REAL *bptr = (REAL *)&(b.elem().elem().elem().real());
  
  if( s.hasOrderedRep() ) { 
    REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
    REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
    REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());
    
    
    // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
    int n_3vec = (s.end()-s.start()+1)*Ns;
    vcaxmby3(zptr, aptr, xptr, bptr, yptr, n_3vec);
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];
      
      REAL *xptr = (REAL *) &(x.elem(i).elem(0).elem(0).real());
      REAL *yptr = (REAL *) &(y.elem(i).elem(0).elem(0).real());
      REAL *zptr =          &(d.elem(i).elem(0).elem(0).real());
      
      
      // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
      vcaxmby3(zptr, aptr, xptr, bptr, yptr, 4);
            
    }
  }
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
	         Reference< QDPType< CScal, OScalar< CScal > > > >,
	        BinaryNode<OpMultiply, 
	         Reference< QDPType< TVec, OLattice< TVec > > >,
	         Reference< QDPType< CScal, OScalar< CScal > > > > >,
	        OLattice< TVec > > &rhs,
	       const Subset& s)
{

#ifdef DEBUG_CBLAS
  QDPIO::cout << "z = x*a - y*b" << endl;
#endif

  // Peel the stuff out of the expression
  // y is the right side of rhs

  // ax is the left side of rhs and is in a binary node
  typedef BinaryNode<OpMultiply, 
    Reference< QDPType< TVec, OLattice< TVec > > >,
    Reference< QDPType< CScal, OScalar< CScal > > > > BN;

  // get the binary node
  const BN &mulNode1 = static_cast<const BN&> (rhs.expression().left());
  const BN &mulNode2 = static_cast<const BN&> (rhs.expression().right());

  // get a and x out of the binary node
  const OLattice< TVec >& x = static_cast<const OLattice< TVec >&>(mulNode1.left());
  const OScalar< CScal >& a = static_cast<const OScalar< CScal >&>(mulNode1.right());
  
  // get b and y out of the binary node
  const OLattice< TVec >& y = static_cast<const OLattice< TVec >&>(mulNode2.left());

  const OScalar< CScal >& b = static_cast<const OScalar< CScal >&>(mulNode2.right());
  
  // Set pointers 
  REAL *aptr = (REAL *)&(a.elem().elem().elem().real());
  REAL *bptr = (REAL *)&(b.elem().elem().elem().real());

  if( s.hasOrderedRep() ) { 
    REAL *xptr = (REAL *) &(x.elem(s.start()).elem(0).elem(0).real());
    REAL *yptr = (REAL *) &(y.elem(s.start()).elem(0).elem(0).real());
    REAL* zptr =  &(d.elem(s.start()).elem(0).elem(0).real());
    
    
    // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
    int n_3vec = (s.end()-s.start()+1)*Ns;
    vcaxmby3(zptr, aptr, xptr, bptr, yptr, n_3vec);
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];
      
      REAL *xptr = (REAL *) &(x.elem(i).elem(0).elem(0).real());
      REAL *yptr = (REAL *) &(y.elem(i).elem(0).elem(0).real());
      REAL *zptr =          &(d.elem(i).elem(0).elem(0).real());
      
      
      // Get the no of 3vecs. s.start() and s.end() are inclusive so add +1
      vcaxmby3(zptr, aptr, xptr, bptr, yptr, 4);
            
    }
  }
}

  
} // namespace QDP;
#endif
