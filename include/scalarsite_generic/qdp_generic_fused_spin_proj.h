#ifndef QDP_SCALARSITE_FUSED_SPIN_PROJ_H
#define QDP_SCALARSITE_FUSED_SPIN_PROJ_H

#if QDP_USE_SSE == 1
#include "scalarsite_sse/sse_mv_switchbox.h"
#else
#include "scalarsite_generic/generic_mv_switchbox.h"
#endif

/* Evaluates for things like adj(u)*spinProjectDir0Plus(y) */
using namespace QDP;
QDP_BEGIN_NAMESPACE(QDP);

typedef PScalar< PColorMatrix< RComplex<REAL>, Nc> > SU3Mat;


// HalfVec = adj(u)*SpinProjectDir0Plus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& d,
              const OpAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                OpAdjMultiply,
	               
	                UnaryNode< OpIdentity, 
	                  Reference< QDPType< SU3Mat, OLattice< SU3Mat > > > >,
	               
	                UnaryNode< FnSpinProjectDir0Plus, 
                          Reference< QDPType< FVec,   OLattice< FVec > > > >
                      >,
	              OLattice< HVec > 
                    >&rhs,
	      const OrderedSubset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left().child());
  const OLattice< FVec >& a = static_cast< const OLattice< FVec >& >(rhs.expression().right().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  unsigned long aoffset=0;
  const int re=0;
  const int im=1;
  /* 1 + \gamma_0 =  1  0  0  i 
                     0  1  i  0
                     0 -i  1  0
                    -i  0  0  1 
 
   *      ( d0r + i d0i )  =  ( {x0r - x3i} + i{x0i + x3r} )
   *      ( d1r + i d1i )     ( {x1r - x2i} + i{x1i + x2r} )
   */
  for(int site=s.start(); site <= s.end(); ++site) {
    SpinColFull a_tmp;
    
    // Stream in a 
    for(int spin=0; spin < 4; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }
 
    // Do the projection
    PSpinVector<PColorVector<RComplex<REAL>, 3>, 2> b;
    REAL* bptr = (REAL*)&(b.elem(0).elem(0).real());
    int boffset = 0;

    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[0][col][re] - a_tmp[3][col][im];
      bptr[ boffset++ ] = a_tmp[0][col][im] + a_tmp[3][col][re];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[1][col][re] - a_tmp[2][col][im];
      bptr[ boffset++ ] = a_tmp[1][col][im] + a_tmp[2][col][re];
    }

    // Now do the multiply by the adjoint
    _inline_mult_adj_su3_mat_vec( u.elem(site).elem(), 
				  b.elem(0),
				  d.elem(site).elem(0) );

    _inline_mult_adj_su3_mat_vec( u.elem(site).elem(), 
				  b.elem(1),
				  d.elem(site).elem(1) );

  }

}

// HalfVec = adj(u)*SpinProjectDir0Minus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& d,
              const OpAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                OpAdjMultiply,
	               
	                UnaryNode< OpIdentity, 
	                  Reference< QDPType< SU3Mat, OLattice< SU3Mat > > > >,
	               
	                UnaryNode< FnSpinProjectDir0Minus, 
                          Reference< QDPType< FVec,   OLattice< FVec > > > >
                      >,
	              OLattice< HVec > 
                    >&rhs,
	      const OrderedSubset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left().child());
  const OLattice< FVec >& a = static_cast< const OLattice< FVec >& >(rhs.expression().right().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  unsigned long aoffset=0;
  const int re=0;
  const int im=1;

  /*                              ( 1  0  0 -i)  ( a0 )    ( a0 - i a3 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1 -i  0)  ( a1 )  = ( a1 - i a2 )
   *                    0         ( 0  i  1  0)  ( a2 )    ( a2 + i a1 )
   *                              ( i  0  0  1)  ( a3 )    ( a3 + i a0 )

   * The bottom components of be may be reconstructed using the formula

   *   ( b2r + i b2i )  =  ( {a2r - a1i} + i{a2i + a1r} )  =  ( - b1i + i b1r )
   *   ( b3r + i b3i )     ( {a3r - a0i} + i{a3i + a0r} )     ( - b0i + i b0r ) 
   */

  for(int site=s.start(); site <= s.end(); ++site) {
    SpinColFull a_tmp;
    
    // Stream in a 
    for(int spin=0; spin < 4; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }
 
    // Do the projection
    PSpinVector<PColorVector<RComplex<REAL>, 3>, 2> b;
    REAL* bptr = (REAL*)&(b.elem(0).elem(0).real());
    int boffset = 0;

     // Output Component 0
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[0][col][re] + a_tmp[3][col][im];
      bptr[ boffset++ ] = a_tmp[0][col][im] - a_tmp[3][col][re];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[1][col][re] + a_tmp[2][col][im];
      bptr[ boffset++ ] = a_tmp[1][col][im] - a_tmp[2][col][re];
    }

    // Now do the multiply by the adjoint
    _inline_mult_adj_su3_mat_vec( u.elem(site).elem(), 
				  b.elem(0),
				  d.elem(site).elem(0) );

    _inline_mult_adj_su3_mat_vec( u.elem(site).elem(), 
				  b.elem(1),
				  d.elem(site).elem(1) );

  }

}

// HalfVec = adj(u)*SpinProjectDir1Plus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& d,
              const OpAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                OpAdjMultiply,
	               
	                UnaryNode< OpIdentity, 
	                  Reference< QDPType< SU3Mat, OLattice< SU3Mat > > > >,
	               
	                UnaryNode< FnSpinProjectDir1Plus, 
                          Reference< QDPType< FVec,   OLattice< FVec > > > >
                      >,
	              OLattice< HVec > 
                    >&rhs,
	      const OrderedSubset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left().child());
  const OLattice< FVec >& a = static_cast< const OLattice< FVec >& >(rhs.expression().right().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  unsigned long aoffset=0;
  const int re=0;
  const int im=1;

  /* 1 + \gamma_1 =  1  0  0 -1 
                     0  1  1  0
                     0  1  1  0
                    -1  0  0  1 
 
   *      ( b0r + i b0i )  =  ( {a0r - a3r} + i{a0i - a3i} )
   *      ( b1r + i b1i )     ( {a1r + a2r} + i{a1i + a2i} )
   */
  for(int site=s.start(); site <= s.end(); ++site) {
    SpinColFull a_tmp;
    
    // Stream in a 
    for(int spin=0; spin < 4; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }
 
    // Do the projection
    PSpinVector<PColorVector<RComplex<REAL>, 3>, 2> b;
    REAL* bptr = (REAL*)&(b.elem(0).elem(0).real());
    int boffset = 0;
   // Output Component 0
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[0][col][re] - a_tmp[3][col][re];
      bptr[ boffset++ ] = a_tmp[0][col][im] - a_tmp[3][col][im];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[1][col][re] + a_tmp[2][col][re];
      bptr[ boffset++ ] = a_tmp[1][col][im] + a_tmp[2][col][im];
    }

    // Now do the multiply by the adjoint
    _inline_mult_adj_su3_mat_vec( u.elem(site).elem(), 
				  b.elem(0),
				  d.elem(site).elem(0) );

    _inline_mult_adj_su3_mat_vec( u.elem(site).elem(), 
				  b.elem(1),
				  d.elem(site).elem(1) );

  }

}

// HalfVec = adj(u)*SpinProjectDir1Minus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& d,
              const OpAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                OpAdjMultiply,
	               
	                UnaryNode< OpIdentity, 
	                  Reference< QDPType< SU3Mat, OLattice< SU3Mat > > > >,
	               
	                UnaryNode< FnSpinProjectDir1Minus, 
                          Reference< QDPType< FVec,   OLattice< FVec > > > >
                      >,
	              OLattice< HVec > 
                    >&rhs,
	      const OrderedSubset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left().child());
  const OLattice< FVec >& a = static_cast< const OLattice< FVec >& >(rhs.expression().right().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  unsigned long aoffset=0;
  const int re=0;
  const int im=1;


  /* 1 - \gamma_1 =  1  0  0 +1 
                     0  1 -1  0
                     0 -1  1  0
                    +1  0  0  1 
 
   *      ( b0r + i b0i )  =  ( {a0r + a3r} + i{a0i + a3i} )
   *      ( b1r + i b1i )     ( {a1r - a2r} + i{a1i - a2i} )
   */


  for(int site=s.start(); site <= s.end(); ++site) {
    SpinColFull a_tmp;
    
    // Stream in a 
    for(int spin=0; spin < 4; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }
 
    // Do the projection
    PSpinVector<PColorVector<RComplex<REAL>, 3>, 2> b;
    REAL* bptr = (REAL*)&(b.elem(0).elem(0).real());
    int boffset = 0;

    // Output Component 0
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[0][col][re] + a_tmp[3][col][re];
      bptr[ boffset++ ] = a_tmp[0][col][im] + a_tmp[3][col][im];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[1][col][re] - a_tmp[2][col][re];
      bptr[ boffset++ ] = a_tmp[1][col][im] - a_tmp[2][col][im];
    }

    // Now do the multiply by the adjoint
    _inline_mult_adj_su3_mat_vec( u.elem(site).elem(), 
				  b.elem(0),
				  d.elem(site).elem(0) );

    _inline_mult_adj_su3_mat_vec( u.elem(site).elem(), 
				  b.elem(1),
				  d.elem(site).elem(1) );

  }

}


// HalfVec = adj(u)*SpinProjectDir2Plus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& d,
              const OpAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                OpAdjMultiply,
	               
	                UnaryNode< OpIdentity, 
	                  Reference< QDPType< SU3Mat, OLattice< SU3Mat > > > >,
	               
	                UnaryNode< FnSpinProjectDir2Plus, 
                          Reference< QDPType< FVec,   OLattice< FVec > > > >
                      >,
	              OLattice< HVec > 
                    >&rhs,
	      const OrderedSubset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left().child());
  const OLattice< FVec >& a = static_cast< const OLattice< FVec >& >(rhs.expression().right().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  unsigned long aoffset=0;
  const int re=0;
  const int im=1;


  for(int site=s.start(); site <= s.end(); ++site) {
    SpinColFull a_tmp;
    /* 1 + \gamma_2 =  1  0  i  0 
                       0  1  0 -i
                      -i  0  1  0
                       0  i  0  1 


   *      ( b0r + i b0i )  =  ( {a0r - a2i} + i{a0i + a2r} )
   *      ( b1r + i b1i )     ( {a1r + a3i} + i{a1i - a3r} )
   */

    // Stream in a 
    for(int spin=0; spin < 4; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }
 
    // Do the projection
    PSpinVector<PColorVector<RComplex<REAL>, 3>, 2> b;
    REAL* bptr = (REAL*)&(b.elem(0).elem(0).real());
    int boffset = 0;

    // Output Component 0
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[0][col][re] - a_tmp[2][col][im];
      bptr[ boffset++ ] = a_tmp[0][col][im] + a_tmp[2][col][re];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[1][col][re] + a_tmp[3][col][im];
      bptr[ boffset++ ] = a_tmp[1][col][im] - a_tmp[3][col][re];
    }


    // Now do the multiply by the adjoint
    _inline_mult_adj_su3_mat_vec( u.elem(site).elem(), 
				  b.elem(0),
				  d.elem(site).elem(0) );

    _inline_mult_adj_su3_mat_vec( u.elem(site).elem(), 
				  b.elem(1),
				  d.elem(site).elem(1) );

  }

}

// HalfVec = adj(u)*SpinProjectDir2Minus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& d,
              const OpAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                OpAdjMultiply,
	               
	                UnaryNode< OpIdentity, 
	                  Reference< QDPType< SU3Mat, OLattice< SU3Mat > > > >,
	               
	                UnaryNode< FnSpinProjectDir2Minus, 
                          Reference< QDPType< FVec,   OLattice< FVec > > > >
                      >,
	              OLattice< HVec > 
                    >&rhs,
	      const OrderedSubset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left().child());
  const OLattice< FVec >& a = static_cast< const OLattice< FVec >& >(rhs.expression().right().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  unsigned long aoffset=0;
  const int re=0;
  const int im=1;

  /* 1 - \gamma_2 =  1  0  -i  0 
                     0  1  0  +i
                    +i  0  1   0
                     0 -i  0   1 


   *      ( b0r + i b0i )  =  ( {a0r + a2i} + i{a0i - a2r} )
   *      ( b1r + i b1i )     ( {a1r - a3i} + i{a1i + a3r} )
   */
  for(int site=s.start(); site <= s.end(); ++site) {
    SpinColFull a_tmp;
    
    // Stream in a 
    for(int spin=0; spin < 4; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }
 
    // Do the projection
    PSpinVector<PColorVector<RComplex<REAL>, 3>, 2> b;
    REAL* bptr = (REAL*)&(b.elem(0).elem(0).real());
    int boffset = 0;

    // Output Component 0
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[0][col][re] + a_tmp[2][col][im];
      bptr[ boffset++ ] = a_tmp[0][col][im] - a_tmp[2][col][re];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[1][col][re] - a_tmp[3][col][im];
      bptr[ boffset++ ] = a_tmp[1][col][im] + a_tmp[3][col][re];
    }


    // Now do the multiply by the adjoint
    _inline_mult_adj_su3_mat_vec( u.elem(site).elem(), 
				  b.elem(0),
				  d.elem(site).elem(0) );

    _inline_mult_adj_su3_mat_vec( u.elem(site).elem(), 
				  b.elem(1),
				  d.elem(site).elem(1) );

  }

}

// HalfVec = adj(u)*SpinProjectDir3Plus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& d,
              const OpAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                OpAdjMultiply,
	               
	                UnaryNode< OpIdentity, 
	                  Reference< QDPType< SU3Mat, OLattice< SU3Mat > > > >,
	               
	                UnaryNode< FnSpinProjectDir3Plus, 
                          Reference< QDPType< FVec,   OLattice< FVec > > > >
                      >,
	              OLattice< HVec > 
                    >&rhs,
	      const OrderedSubset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left().child());
  const OLattice< FVec >& a = static_cast< const OLattice< FVec >& >(rhs.expression().right().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  unsigned long aoffset=0;
  const int re=0;
  const int im=1;


  for(int site=s.start(); site <= s.end(); ++site) {
    SpinColFull a_tmp;
  /* 1 + \gamma_3 =  1  0  1  0 
                     0  1  0  1
                     1  0  1  0
                     0  1  0  1 

   *      ( b0r + i b0i )  =  ( {a0r + a2r} + i{a0i + a2i} )
   *      ( b1r + i b1i )     ( {a1r + a3r} + i{a1i + a3i} )
   */

    // Stream in a 
    for(int spin=0; spin < 4; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }
 
    // Do the projection
    PSpinVector<PColorVector<RComplex<REAL>, 3>, 2> b;
    REAL* bptr = (REAL*)&(b.elem(0).elem(0).real());
    int boffset = 0;

     // Output Component 0
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[0][col][re] + a_tmp[2][col][re];
      bptr[ boffset++ ] = a_tmp[0][col][im] + a_tmp[2][col][im];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[1][col][re] + a_tmp[3][col][re];
      bptr[ boffset++ ] = a_tmp[1][col][im] + a_tmp[3][col][im];
    }

    // Now do the multiply by the adjoint
    _inline_mult_adj_su3_mat_vec( u.elem(site).elem(), 
				  b.elem(0),
				  d.elem(site).elem(0) );

    _inline_mult_adj_su3_mat_vec( u.elem(site).elem(), 
				  b.elem(1),
				  d.elem(site).elem(1) );

  }

}

// HalfVec = adj(u)*SpinProjectDir3Minus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& d,
              const OpAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                OpAdjMultiply,
	               
	                UnaryNode< OpIdentity, 
	                  Reference< QDPType< SU3Mat, OLattice< SU3Mat > > > >,
	               
	                UnaryNode< FnSpinProjectDir3Minus, 
                          Reference< QDPType< FVec,   OLattice< FVec > > > >
                      >,
	              OLattice< HVec > 
                    >&rhs,
	      const OrderedSubset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left().child());
  const OLattice< FVec >& a = static_cast< const OLattice< FVec >& >(rhs.expression().right().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  unsigned long aoffset=0;
  const int re=0;
  const int im=1;

  /* 1 - \gamma_3 =  1  0  -1  0 
                     0  1  0  -1
                    -1  0  1  0
                     0 -1  0  1 

   *      ( b0r + i b0i )  =  ( {a0r - a2r} + i{a0i - a2i} )
   *      ( b1r + i b1i )     ( {a1r - a3r} + i{a1i - a3i} )
   */
  for(int site=s.start(); site <= s.end(); ++site) {
    SpinColFull a_tmp;
    
    // Stream in a 
    for(int spin=0; spin < 4; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }
 
    // Do the projection
    PSpinVector<PColorVector<RComplex<REAL>, 3>, 2> b;
    REAL* bptr = (REAL*)&(b.elem(0).elem(0).real());
    int boffset = 0;


    // Output Component 0
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[0][col][re] - a_tmp[2][col][re];
      bptr[ boffset++ ] = a_tmp[0][col][im] - a_tmp[2][col][im];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[1][col][re] - a_tmp[3][col][re];
      bptr[ boffset++ ] = a_tmp[1][col][im] - a_tmp[3][col][im];
    }

    // Now do the multiply by the adjoint
    _inline_mult_adj_su3_mat_vec( u.elem(site).elem(), 
				  b.elem(0),
				  d.elem(site).elem(0) );

    _inline_mult_adj_su3_mat_vec( u.elem(site).elem(), 
				  b.elem(1),
				  d.elem(site).elem(1) );

  }

}


// REconstruct(Upsi)
template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAssign& op,
              const QDPExpr<
	            UnaryNode< FnSpinReconstructDir0Plus,
	              BinaryNode< 
	                OpMultiply,
	             	Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
	              
                      > 
	             >,
	             OLattice< FVec >  >&rhs,
	      const OrderedSubset& s) 
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().child().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().child().right());

  for(int site=s.start(); site <= s.end(); ++site) {

    
    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(0),
			      d.elem(site).elem(0) );

    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(1),
			      d.elem(site).elem(1) );



   /* 1 + \gamma_0 =  1  0  0  i 
                      0  1  i  0
                      0 -i  1  0
                     -i  0  0  1 
 
    *  ( b2r + i b2i )  =  ( {a2r + a1i} + i{a2i - a1r} )  =  ( b1i - i b1r )
    *  ( b3r + i b3i )     ( {a3r + a0i} + i{a3i - a0r} )     ( b0i - i b0r ) 
    */
    
    REAL* bptr=(REAL *)&(d.elem(site).elem(2).elem(0).real());
    int boffset=0;
    int re=0;
    int im=0;

    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] =  d.elem(site).elem(1).elem(col).imag();
      bptr[ boffset++ ] = -d.elem(site).elem(1).elem(col).real();
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] =  d.elem(site).elem(0).elem(col).imag();
      bptr[ boffset++ ] = -d.elem(site).elem(0).elem(col).real(); 
    }			      

  }
}

template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAssign& op,
              const QDPExpr<
	            UnaryNode< FnSpinReconstructDir0Minus,
	              BinaryNode< 
	                OpMultiply,
	             	Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
	              
                      > 
	             >,
	             OLattice< FVec >  >&rhs,
	      const OrderedSubset& s) 
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().child().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().child().right());

  for(int site=s.start(); site <= s.end(); ++site) {

    
    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(0),
			      d.elem(site).elem(0) );

    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(1),
			      d.elem(site).elem(1) );



   /*                              ( 1  0  0 -i)  ( a0 )    ( a0 - i a3 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1 -i  0)  ( a1 )  = ( a1 - i a2 )
   *                    0         ( 0  i  1  0)  ( a2 )    ( a2 + i a1 )
   *                              ( i  0  0  1)  ( a3 )    ( a3 + i a0 )

   * The bottom components of be may be reconstructed using the formula

   *   ( b2r + i b2i )  =  ( {a2r - a1i} + i{a2i + a1r} )  =  ( - b1i + i b1r )
   *   ( b3r + i b3i )     ( {a3r - a0i} + i{a3i + a0r} )     ( - b0i + i b0r ) 
   */
    REAL* bptr=(REAL *)&(d.elem(site).elem(2).elem(0).real());
    int boffset=0;
    int re=0;
    int im=0;

    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = -d.elem(site).elem(1).elem(col).imag();
      bptr[ boffset++ ] =  d.elem(site).elem(1).elem(col).real();
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = -d.elem(site).elem(0).elem(col).imag();
      bptr[ boffset++ ] =  d.elem(site).elem(0).elem(col).real(); 
    }			      

  }
}


template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAssign& op,
              const QDPExpr<
	            UnaryNode< FnSpinReconstructDir1Plus,
	              BinaryNode< 
	                OpMultiply,
	             	Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
	              
                      > 
	             >,
	             OLattice< FVec >  >&rhs,
	      const OrderedSubset& s) 
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().child().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().child().right());

  for(int site=s.start(); site <= s.end(); ++site) {

    
    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(0),
			      d.elem(site).elem(0) );

    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(1),
			      d.elem(site).elem(1) );


  /* 1 + \gamma_1 =  1  0  0 -1 
                     0  1  1  0
                     0  1  1  0
                    -1  0  0  1 
 

   *   ( b2r + i b2i )  =  ( {a2r + a1r} + i{a2i + a1i} )  =  (   b1r + i b1i )
   *   ( b3r + i b3i )     ( {a3r - a0r} + i{a3i - a0i} )     ( - b0r - i b0i ) 
  
  */
    
    REAL* bptr=(REAL *)&(d.elem(site).elem(2).elem(0).real());
    int boffset=0;
    int re=0;
    int im=0;

    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] =  d.elem(site).elem(1).elem(col).real();
      bptr[ boffset++ ] =  d.elem(site).elem(1).elem(col).imag();
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = -d.elem(site).elem(0).elem(col).real();
      bptr[ boffset++ ] = -d.elem(site).elem(0).elem(col).imag(); 
    }			      

  }
}

template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAssign& op,
              const QDPExpr<
	            UnaryNode< FnSpinReconstructDir1Minus,
	              BinaryNode< 
	                OpMultiply,
	             	Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
	              
                      > 
	             >,
	             OLattice< FVec >  >&rhs,
	      const OrderedSubset& s) 
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().child().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().child().right());

  for(int site=s.start(); site <= s.end(); ++site) {

    
    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(0),
			      d.elem(site).elem(0) );

    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(1),
			      d.elem(site).elem(1) );


  /*                              ( 1  0  0  1)  ( a0 )    ( a0 + a3 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1 -1  0)  ( a1 )  = ( a1 - a2 )
   *                    1         ( 0 -1  1  0)  ( a2 )    ( a2 - a1 )
   *                              ( 1  0  0  1)  ( a3 )    ( a3 + a0 )
	 
   * The bottom components of be may be reconstructed using the formula

   *  ( b2r + i b2i )  =  ( {a2r - a1r} + i{a2i - a1i} )  =  ( - b1r - i b1i )
   *  ( b3r + i b3i )     ( {a3r + a0r} + i{a3i + a0i} )     (   b0r + i b0i ) 
   */
 
    REAL* bptr=(REAL *)&(d.elem(site).elem(2).elem(0).real());
    int boffset=0;
    int re=0;
    int im=0;

    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = -d.elem(site).elem(1).elem(col).real();
      bptr[ boffset++ ] = -d.elem(site).elem(1).elem(col).imag();
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] =  d.elem(site).elem(0).elem(col).real();
      bptr[ boffset++ ] =  d.elem(site).elem(0).elem(col).imag(); 
    }			      

  }
}

template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAssign& op,
              const QDPExpr<
	            UnaryNode< FnSpinReconstructDir2Plus,
	              BinaryNode< 
	                OpMultiply,
	             	Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
	              
                      > 
	             >,
	             OLattice< FVec >  >&rhs,
	      const OrderedSubset& s) 
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().child().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().child().right());

  for(int site=s.start(); site <= s.end(); ++site) {

    
    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(0),
			      d.elem(site).elem(0) );

    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(1),
			      d.elem(site).elem(1) );


  /* 1 + \gamma_2 =  1  0  i  0 
                     0  1  0 -i
                    -i  0  1  0
                     0  i  0  1 
		     
   *  ( b2r + i b2i )  =  ( {a2r + a0i} + i{a2i - a0r} )  =  (   b0i - i b0r )
   *  ( b3r + i b3i )     ( {a3r - a1i} + i{a3i + a1r} )     ( - b1i + i b1r ) 
  */
    REAL* bptr=(REAL *)&(d.elem(site).elem(2).elem(0).real());
    int boffset=0;
    int re=0;
    int im=0;

    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] =  d.elem(site).elem(0).elem(col).imag();
      bptr[ boffset++ ] = -d.elem(site).elem(0).elem(col).real();
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = -d.elem(site).elem(1).elem(col).imag();
      bptr[ boffset++ ] =  d.elem(site).elem(1).elem(col).real(); 
    }			      

  }
}

template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAssign& op,
              const QDPExpr<
	            UnaryNode< FnSpinReconstructDir2Minus,
	              BinaryNode< 
	                OpMultiply,
	             	Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
	              
                      > 
	             >,
	             OLattice< FVec >  >&rhs,
	      const OrderedSubset& s) 
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().child().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().child().right());

  for(int site=s.start(); site <= s.end(); ++site) {

    
    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(0),
			      d.elem(site).elem(0) );

    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(1),
			      d.elem(site).elem(1) );


 /*                               ( 1  0 -i  0)  ( a0 )    ( a0 - i a2 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1  0  i)  ( a1 )  = ( a1 + i a3 )
   *                    2         ( i  0  1  0)  ( a2 )    ( a2 + i a0 )
   *                              ( 0 -i  0  1)  ( a3 )    ( a3 - i a1 )

   * The bottom components of be may be reconstructed using the formula
   *  ( b2r + i b2i )  =  ( {a2r - a0i} + i{a2i + a0r} )  =  ( - b0i + i b0r )
   *  ( b3r + i b3i )     ( {a3r + a1i} + i{a3i - a1r} )     (   b1i - i b1r )
   */
    REAL* bptr=(REAL *)&(d.elem(site).elem(2).elem(0).real());
    int boffset=0;
    int re=0;
    int im=0;

    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = -d.elem(site).elem(0).elem(col).imag();
      bptr[ boffset++ ] =  d.elem(site).elem(0).elem(col).real();
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] =  d.elem(site).elem(1).elem(col).imag();
      bptr[ boffset++ ] = -d.elem(site).elem(1).elem(col).real(); 
    }			      

  }
}

template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAssign& op,
              const QDPExpr<
	            UnaryNode< FnSpinReconstructDir3Plus,
	              BinaryNode< 
	                OpMultiply,
	             	Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
	              
                      > 
	             >,
	             OLattice< FVec >  >&rhs,
	      const OrderedSubset& s) 
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().child().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().child().right());

  for(int site=s.start(); site <= s.end(); ++site) {

    
    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(0),
			      d.elem(site).elem(0) );

    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(1),
			      d.elem(site).elem(1) );


  /*                              ( 1  0  1  0)  ( a0 )    ( a0 + a2 )
   *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  0  1)  ( a1 )  = ( a1 + a3 )
   *                    3         ( 1  0  1  0)  ( a2 )    ( a2 + a0 )
   *                              ( 0  1  0  1)  ( a3 )    ( a3 + a1 )
   
   * The bottom components of be may be reconstructed using the formula
   
   *   ( b2r + i b2i )  =  ( {a2r + a0r} + i{a2i + a0i} )  =  ( b0r + i b0i )
   *   ( b3r + i b3i )     ( {a3r + a1r} + i{a3i + a1i} )     ( b1r + i b1i ) 
   */

    REAL* bptr=(REAL *)&(d.elem(site).elem(2).elem(0).real());
    int boffset=0;
    int re=0;
    int im=0;

    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] =  d.elem(site).elem(0).elem(col).real();
      bptr[ boffset++ ] =  d.elem(site).elem(0).elem(col).imag();
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] =  d.elem(site).elem(1).elem(col).real();
      bptr[ boffset++ ] =  d.elem(site).elem(1).elem(col).imag(); 
    }			      

  }
}

template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAssign& op,
              const QDPExpr<
	            UnaryNode< FnSpinReconstructDir3Minus,
	              BinaryNode< 
	                OpMultiply,
	             	Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
	              
                      > 
	             >,
	             OLattice< FVec >  >&rhs,
	      const OrderedSubset& s) 
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().child().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().child().right());

  for(int site=s.start(); site <= s.end(); ++site) {

    
    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(0),
			      d.elem(site).elem(0) );

    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(1),
			      d.elem(site).elem(1) );


 /*                              ( 1  0 -1  0)  ( a0 )    ( a0 - a2 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1  0 -1)  ( a1 )  = ( a1 - a3 )
   *                    3         (-1  0  1  0)  ( a2 )    ( a2 - a0 )
   *                              ( 0 -1  0  1)  ( a3 )    ( a3 - a1 )
      
   * The bottom components of be may be reconstructed using the formula
   *  ( b2r + i b2i )  =  ( {a2r - a0r} + i{a2i - a0i} )  =  ( - b0r - i b0i )
   *  ( b3r + i b3i )     ( {a3r - a1r} + i{a3i - a1i} )     ( - b1r - i b1i ) 
   */

    REAL* bptr=(REAL *)&(d.elem(site).elem(2).elem(0).real());
    int boffset=0;
    int re=0;
    int im=0;

    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = -d.elem(site).elem(0).elem(col).real();
      bptr[ boffset++ ] = -d.elem(site).elem(0).elem(col).imag();
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = -d.elem(site).elem(1).elem(col).real();
      bptr[ boffset++ ] = -d.elem(site).elem(1).elem(col).imag(); 
    }			      

  }
}



// +=REconstruct(Upsi)
template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAddAssign& op,
              const QDPExpr<
	            UnaryNode< FnSpinReconstructDir0Plus,
	              BinaryNode< 
	                OpMultiply,
	             	Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
	              
                      > 
	             >,
	             OLattice< FVec >  >&rhs,
	      const OrderedSubset& s) 
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().child().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().child().right());

  for(int site=s.start(); site <= s.end(); ++site) {

    PSpinVector<PColorVector<RComplex<REAL>,3>, 2> h;

    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(0),
			      h.elem(0) );

    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(1),
			      h.elem(1) );



   /* 1 + \gamma_0 =  1  0  0  i 
                      0  1  i  0
                      0 -i  1  0
                     -i  0  0  1 
 
    *  ( b2r + i b2i )  =  ( {a2r + a1i} + i{a2i - a1r} )  =  ( b1i - i b1r )
    *  ( b3r + i b3i )     ( {a3r + a0i} + i{a3i - a0r} )     ( b0i - i b0r ) 
    */
    
    REAL* bptr=(REAL *)&(d.elem(site).elem(0).elem(0).real());
    int boffset=0;
    int re=0;
    int im=0;
    for(int col=0; col < 3; col++) { 
      bptr[ boffset++ ] += h.elem(0).elem(col).real();
      bptr[ boffset++ ] += h.elem(0).elem(col).imag();
    }

    for(int col=0; col < 3; col++) { 
      bptr[ boffset++ ] += h.elem(1).elem(col).real();
      bptr[ boffset++ ] += h.elem(1).elem(col).imag();
    }

    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] +=  h.elem(1).elem(col).imag();
      bptr[ boffset++ ] -=  h.elem(1).elem(col).real();
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] +=  h.elem(0).elem(col).imag();
      bptr[ boffset++ ] -=  h.elem(0).elem(col).real(); 
    }			      

  }
}

template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAddAssign& op,
              const QDPExpr<
	            UnaryNode< FnSpinReconstructDir0Minus,
	              BinaryNode< 
	                OpMultiply,
	             	Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
	              
                      > 
	             >,
	             OLattice< FVec >  >&rhs,
	      const OrderedSubset& s) 
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().child().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().child().right());

  for(int site=s.start(); site <= s.end(); ++site) {
    PSpinVector<PColorVector<RComplex<REAL>,3>, 2> h;
    
    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(0),
			      h.elem(0) );

    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(1),
			      h.elem(1) );



   /*                              ( 1  0  0 -i)  ( a0 )    ( a0 - i a3 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1 -i  0)  ( a1 )  = ( a1 - i a2 )
   *                    0         ( 0  i  1  0)  ( a2 )    ( a2 + i a1 )
   *                              ( i  0  0  1)  ( a3 )    ( a3 + i a0 )

   * The bottom components of be may be reconstructed using the formula

   *   ( b2r + i b2i )  =  ( {a2r - a1i} + i{a2i + a1r} )  =  ( - b1i + i b1r )
   *   ( b3r + i b3i )     ( {a3r - a0i} + i{a3i + a0r} )     ( - b0i + i b0r ) 
   */
    REAL* bptr=(REAL *)&(d.elem(site).elem(0).elem(0).real());
    int boffset=0;
    int re=0;
    int im=0;
    for(int col=0; col < 3; col++) { 
      bptr[ boffset++ ] += h.elem(0).elem(col).real();
      bptr[ boffset++ ] += h.elem(0).elem(col).imag();
    }

    for(int col=0; col < 3; col++) { 
      bptr[ boffset++ ] += h.elem(1).elem(col).real();
      bptr[ boffset++ ] += h.elem(1).elem(col).imag();
    }

    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] -=  h.elem(1).elem(col).imag();
      bptr[ boffset++ ] +=  h.elem(1).elem(col).real();
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] -=  h.elem(0).elem(col).imag();
      bptr[ boffset++ ] +=  h.elem(0).elem(col).real(); 
    }			      

  }
}


template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAddAssign& op,
              const QDPExpr<
	            UnaryNode< FnSpinReconstructDir1Plus,
	              BinaryNode< 
	                OpMultiply,
	             	Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
	              
                      > 
	             >,
	             OLattice< FVec >  >&rhs,
	      const OrderedSubset& s) 
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().child().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().child().right());

  for(int site=s.start(); site <= s.end(); ++site) {

    PSpinVector<PColorVector<RComplex<REAL>,3>, 2> h;    
    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(0),
			      h.elem(0) );

    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(1),
			      h.elem(1) );


  /* 1 + \gamma_1 =  1  0  0 -1 
                     0  1  1  0
                     0  1  1  0
                    -1  0  0  1 
 

   *   ( b2r + i b2i )  =  ( {a2r + a1r} + i{a2i + a1i} )  =  (   b1r + i b1i )
   *   ( b3r + i b3i )     ( {a3r - a0r} + i{a3i - a0i} )     ( - b0r - i b0i ) 
  
  */
    

    REAL* bptr=(REAL *)&(d.elem(site).elem(0).elem(0).real());
    int boffset=0;
    int re=0;
    int im=0;
    for(int col=0; col < 3; col++) { 
      bptr[ boffset++ ] += h.elem(0).elem(col).real();
      bptr[ boffset++ ] += h.elem(0).elem(col).imag();
    }

    for(int col=0; col < 3; col++) { 
      bptr[ boffset++ ] += h.elem(1).elem(col).real();
      bptr[ boffset++ ] += h.elem(1).elem(col).imag();
    }


    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] +=  h.elem(1).elem(col).real();
      bptr[ boffset++ ] +=  h.elem(1).elem(col).imag();
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] -=  h.elem(0).elem(col).real();
      bptr[ boffset++ ] -=  h.elem(0).elem(col).imag(); 
    }			      

  }
}

template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAddAssign& op,
              const QDPExpr<
	            UnaryNode< FnSpinReconstructDir1Minus,
	              BinaryNode< 
	                OpMultiply,
	             	Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
	              
                      > 
	             >,
	             OLattice< FVec >  >&rhs,
	      const OrderedSubset& s) 
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().child().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().child().right());

  for(int site=s.start(); site <= s.end(); ++site) {
 
    PSpinVector<PColorVector<RComplex<REAL>,3>, 2> h;   
    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(0),
			      h.elem(0) );

    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(1),
			      h.elem(1) );


  /*                              ( 1  0  0  1)  ( a0 )    ( a0 + a3 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1 -1  0)  ( a1 )  = ( a1 - a2 )
   *                    1         ( 0 -1  1  0)  ( a2 )    ( a2 - a1 )
   *                              ( 1  0  0  1)  ( a3 )    ( a3 + a0 )
	 
   * The bottom components of be may be reconstructed using the formula

   *  ( b2r + i b2i )  =  ( {a2r - a1r} + i{a2i - a1i} )  =  ( - b1r - i b1i )
   *  ( b3r + i b3i )     ( {a3r + a0r} + i{a3i + a0i} )     (   b0r + i b0i ) 
   */
 
    REAL* bptr=(REAL *)&(d.elem(site).elem(0).elem(0).real());
    int boffset=0;
    int re=0;
    int im=0;
    for(int col=0; col < 3; col++) { 
      bptr[ boffset++ ] += h.elem(0).elem(col).real();
      bptr[ boffset++ ] += h.elem(0).elem(col).imag();
    }

    for(int col=0; col < 3; col++) { 
      bptr[ boffset++ ] += h.elem(1).elem(col).real();
      bptr[ boffset++ ] += h.elem(1).elem(col).imag();
    }

    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] -= h.elem(1).elem(col).real();
      bptr[ boffset++ ] -= h.elem(1).elem(col).imag();
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] +=  h.elem(0).elem(col).real();
      bptr[ boffset++ ] +=  h.elem(0).elem(col).imag(); 
    }			      

  }
}

template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAddAssign& op,
              const QDPExpr<
	            UnaryNode< FnSpinReconstructDir2Plus,
	              BinaryNode< 
	                OpMultiply,
	             	Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
	              
                      > 
	             >,
	             OLattice< FVec >  >&rhs,
	      const OrderedSubset& s) 
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().child().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().child().right());

  for(int site=s.start(); site <= s.end(); ++site) {
    PSpinVector<PColorVector<RComplex<REAL>,3>, 2> h;
    
    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(0),
			      h.elem(0) );

    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(1),
			      h.elem(1) );


  /* 1 + \gamma_2 =  1  0  i  0 
                     0  1  0 -i
                    -i  0  1  0
                     0  i  0  1 
		     
   *  ( b2r + i b2i )  =  ( {a2r + a0i} + i{a2i - a0r} )  =  (   b0i - i b0r )
   *  ( b3r + i b3i )     ( {a3r - a1i} + i{a3i + a1r} )     ( - b1i + i b1r ) 
  */
    REAL* bptr=(REAL *)&(d.elem(site).elem(0).elem(0).real());
    int boffset=0;
    int re=0;
    int im=0;
    for(int col=0; col < 3; col++) { 
      bptr[ boffset++ ] += h.elem(0).elem(col).real();
      bptr[ boffset++ ] += h.elem(0).elem(col).imag();
    }

    for(int col=0; col < 3; col++) { 
      bptr[ boffset++ ] += h.elem(1).elem(col).real();
      bptr[ boffset++ ] += h.elem(1).elem(col).imag();
    }

    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] +=  h.elem(0).elem(col).imag();
      bptr[ boffset++ ] -=  h.elem(0).elem(col).real();
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] -= h.elem(1).elem(col).imag();
      bptr[ boffset++ ] += h.elem(1).elem(col).real(); 
    }			      

  }
}

template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAddAssign& op,
              const QDPExpr<
	            UnaryNode< FnSpinReconstructDir2Minus,
	              BinaryNode< 
	                OpMultiply,
	             	Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
	              
                      > 
	             >,
	             OLattice< FVec >  >&rhs,
	      const OrderedSubset& s) 
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().child().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().child().right());

  for(int site=s.start(); site <= s.end(); ++site) {

    PSpinVector<PColorVector<RComplex<REAL>,3>, 2> h;    
    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(0),
			      h.elem(0) );

    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(1),
			      h.elem(1) );


 /*                               ( 1  0 -i  0)  ( a0 )    ( a0 - i a2 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1  0  i)  ( a1 )  = ( a1 + i a3 )
   *                    2         ( i  0  1  0)  ( a2 )    ( a2 + i a0 )
   *                              ( 0 -i  0  1)  ( a3 )    ( a3 - i a1 )

   * The bottom components of be may be reconstructed using the formula
   *  ( b2r + i b2i )  =  ( {a2r - a0i} + i{a2i + a0r} )  =  ( - b0i + i b0r )
   *  ( b3r + i b3i )     ( {a3r + a1i} + i{a3i - a1r} )     (   b1i - i b1r )
   */

    REAL* bptr=(REAL *)&(d.elem(site).elem(0).elem(0).real());
    int boffset=0;
    int re=0;
    int im=0;
    for(int col=0; col < 3; col++) { 
      bptr[ boffset++ ] += h.elem(0).elem(col).real();
      bptr[ boffset++ ] += h.elem(0).elem(col).imag();
    }

    for(int col=0; col < 3; col++) { 
      bptr[ boffset++ ] += h.elem(1).elem(col).real();
      bptr[ boffset++ ] += h.elem(1).elem(col).imag();
    }

    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] -=  h.elem(0).elem(col).imag();
      bptr[ boffset++ ] +=  h.elem(0).elem(col).real();
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] +=  h.elem(1).elem(col).imag();
      bptr[ boffset++ ] -=  h.elem(1).elem(col).real(); 
    }			      

  }
}

template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAddAssign& op,
              const QDPExpr<
	            UnaryNode< FnSpinReconstructDir3Plus,
	              BinaryNode< 
	                OpMultiply,
	             	Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
	              
                      > 
	             >,
	             OLattice< FVec >  >&rhs,
	      const OrderedSubset& s) 
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().child().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().child().right());

  for(int site=s.start(); site <= s.end(); ++site) {

    PSpinVector<PColorVector<RComplex<REAL>,3>, 2> h;    
    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(0),
			      h.elem(0) );

    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(1),
			      h.elem(1) );


  /*                              ( 1  0  1  0)  ( a0 )    ( a0 + a2 )
   *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  0  1)  ( a1 )  = ( a1 + a3 )
   *                    3         ( 1  0  1  0)  ( a2 )    ( a2 + a0 )
   *                              ( 0  1  0  1)  ( a3 )    ( a3 + a1 )
   
   * The bottom components of be may be reconstructed using the formula
   
   *   ( b2r + i b2i )  =  ( {a2r + a0r} + i{a2i + a0i} )  =  ( b0r + i b0i )
   *   ( b3r + i b3i )     ( {a3r + a1r} + i{a3i + a1i} )     ( b1r + i b1i ) 
   */


    REAL* bptr=(REAL *)&(d.elem(site).elem(0).elem(0).real());
    int boffset=0;
    int re=0;
    int im=0;
    for(int col=0; col < 3; col++) { 
      bptr[ boffset++ ] += h.elem(0).elem(col).real();
      bptr[ boffset++ ] += h.elem(0).elem(col).imag();
    }

    for(int col=0; col < 3; col++) { 
      bptr[ boffset++ ] += h.elem(1).elem(col).real();
      bptr[ boffset++ ] += h.elem(1).elem(col).imag();
    }

    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] +=  h.elem(0).elem(col).real();
      bptr[ boffset++ ] +=  h.elem(0).elem(col).imag();
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] +=  h.elem(1).elem(col).real();
      bptr[ boffset++ ] +=  h.elem(1).elem(col).imag(); 
    }			      

  }
}

template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAddAssign& op,
              const QDPExpr<
	            UnaryNode< FnSpinReconstructDir3Minus,
	              BinaryNode< 
	                OpMultiply,
	             	Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
	              
                      > 
	             >,
	             OLattice< FVec >  >&rhs,
	      const OrderedSubset& s) 
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().child().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().child().right());

  for(int site=s.start(); site <= s.end(); ++site) {
    PSpinVector<PColorVector<RComplex<REAL>,3>, 2> h;
    
    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(0),
			      h.elem(0) );

    _inline_mult_su3_mat_vec( u.elem(site).elem(),
			      a.elem(site).elem(1),
			      h.elem(1) );


 /*                              ( 1  0 -1  0)  ( a0 )    ( a0 - a2 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1  0 -1)  ( a1 )  = ( a1 - a3 )
   *                    3         (-1  0  1  0)  ( a2 )    ( a2 - a0 )
   *                              ( 0 -1  0  1)  ( a3 )    ( a3 - a1 )
      
   * The bottom components of be may be reconstructed using the formula
   *  ( b2r + i b2i )  =  ( {a2r - a0r} + i{a2i - a0i} )  =  ( - b0r - i b0i )
   *  ( b3r + i b3i )     ( {a3r - a1r} + i{a3i - a1i} )     ( - b1r - i b1i ) 
   */


    REAL* bptr=(REAL *)&(d.elem(site).elem(0).elem(0).real());
    int boffset=0;
    int re=0;
    int im=0;
    for(int col=0; col < 3; col++) { 
      bptr[ boffset++ ] += h.elem(0).elem(col).real();
      bptr[ boffset++ ] += h.elem(0).elem(col).imag();
    }

    for(int col=0; col < 3; col++) { 
      bptr[ boffset++ ] += h.elem(1).elem(col).real();
      bptr[ boffset++ ] += h.elem(1).elem(col).imag();
    }

    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] -= h.elem(0).elem(col).real();
      bptr[ boffset++ ] -= h.elem(0).elem(col).imag();
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] -= h.elem(1).elem(col).real();
      bptr[ boffset++ ] -= h.elem(1).elem(col).imag(); 
    }			      

  }
}

QDP_END_NAMESPACE();

#endif
