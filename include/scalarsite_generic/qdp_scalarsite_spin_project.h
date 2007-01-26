#ifndef QDP_SCALARSITE_SPIN_PROJECT_H
#define QDP_SCALARSITE_SPIN_PROJECT_H


using namespace QDP;
QDP_BEGIN_NAMESPACE(QDP);

// Typedefs
typedef PSpinVector< PColorVector< RComplex<REAL>, Nc>, Ns>>1 > HVec;
typedef PSpinVector< PColorVector< RComplex<REAL>, Nc>, 4> FVec;

// Four spinor (Ns * Nc * Ncomplex ) Ncomplex fastest
typedef REAL SpinColFull[4][3][2];

// Half spinor (Ns/2 * Nc * Ncomplex ) Ncomplex fastest
typedef REAL SpinColHalf[2][3][2];
// d = SpinProjectDir0Plus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinProjectDir0Plus, 
	      Reference< QDPType<FVec,OLattice< FVec > > > >,
	      OLattice< HVec > > &rhs,
	      const OrderedSubset& s) 
{

  //  Get at pointer for 4 vec
  const OLattice< FVec >& a = static_cast<const OLattice< FVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned long aoffset=0;
  unsigned long boffset=0;
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
    // Temporary to hold all of a
    SpinColFull a_tmp; 
    
    // Stream in a 
    for(int spin=0; spin < 4; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }

    // Output Component 0
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[0][col][re] - a_tmp[3][col][im];
      bptr[ boffset++ ] = a_tmp[0][col][im] + a_tmp[3][col][re];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[1][col][re] - a_tmp[2][col][im];
      bptr[ boffset++ ] = a_tmp[1][col][im] + a_tmp[2][col][re];
    }
  }  
}

// d = SpinProjectDir1Plus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinProjectDir1Plus, 
	      Reference< QDPType<FVec,OLattice< FVec > > > >,
	      OLattice< HVec > > &rhs,
	      const OrderedSubset& s) 
{

  
  const OLattice< FVec >& a = static_cast<const OLattice< FVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned long aoffset=0;
  unsigned long boffset=0;
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
    // Temporary to hold all of a
    SpinColFull a_tmp; 
    
    // Stream in a 
    for(int spin=0; spin < 4; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }

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
  }  
 


}

// d = SpinProjectDir2Plus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinProjectDir2Plus, 
	      Reference< QDPType<FVec,OLattice< FVec > > > >,
	      OLattice< HVec > > &rhs,
	      const OrderedSubset& s) 
{
  
  const OLattice< FVec >& a = static_cast<const OLattice< FVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned long aoffset=0;
  unsigned long boffset=0;
  const int re=0;
  const int im=1;
  /* 1 + \gamma_2 =  1  0  i  0 
                     0  1  0 -i
                    -i  0  1  0
                     0  i  0  1 


   *      ( b0r + i b0i )  =  ( {a0r - a2i} + i{a0i + a2r} )
   *      ( b1r + i b1i )     ( {a1r + a3i} + i{a1i - a3r} )
   */

  for(int site=s.start(); site <= s.end(); ++site) {
    // Temporary to hold all of a
    SpinColFull a_tmp; 
    
    // Stream in a 
    for(int spin=0; spin < 4; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }

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
  }  
}

// d = SpinProjectDir3Plus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinProjectDir3Plus, 
	      Reference< QDPType<FVec,OLattice< FVec > > > >,
	      OLattice< HVec > > &rhs,
	      const OrderedSubset& s) 
{
  const OLattice< FVec >& a = static_cast<const OLattice< FVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned long aoffset=0;
  unsigned long boffset=0;
  const int re=0;
  const int im=1;
  /* 1 + \gamma_3 =  1  0  1  0 
                     0  1  0  1
                     1  0  1  0
                     0  1  0  1 

   *      ( b0r + i b0i )  =  ( {a0r + a2r} + i{a0i + a2i} )
   *      ( b1r + i b1i )     ( {a1r + a3r} + i{a1i + a3i} )
   */
  
  for(int site=s.start(); site <= s.end(); ++site) {
    // Temporary to hold all of a
    SpinColFull a_tmp; 
    
    // Stream in a 
    for(int spin=0; spin < 4; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }

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
  }  
}

// d = SpinProjectDir0Minus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinProjectDir0Minus, 
	      Reference< QDPType<FVec,OLattice< FVec > > > >,
	      OLattice< HVec > > &rhs,
	      const OrderedSubset& s) 
{

  //  Get at pointer for 4 vec
  const OLattice< FVec >& a = static_cast<const OLattice< FVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned long aoffset=0;
  unsigned long boffset=0;
  const int re=0;
  const int im=1;
  /* 1 + \gamma_0 =  1  0  0 -i 
                     0  1 -i  0
                     0 +i  1  0
                    +i  0  0  1 

   *      ( b0r + i b0i )  =  ( {a0r + a3i} + i{a0i - a3r} )
   *      ( b1r + i b1i )     ( {a1r + a2i} + i{a1i - a2r} )
		    
  */
  for(int site=s.start(); site <= s.end(); ++site) {
    // Temporary to hold all of a
    SpinColFull a_tmp; 
    
    // Stream in a 
    for(int spin=0; spin < 4; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }

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
  }  
}

// d = SpinProjectDir1Minus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinProjectDir1Minus, 
	      Reference< QDPType<FVec,OLattice< FVec > > > >,
	      OLattice< HVec > > &rhs,
	      const OrderedSubset& s) 
{

  const OLattice< FVec >& a = static_cast<const OLattice< FVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned long aoffset=0;
  unsigned long boffset=0;
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
    // Temporary to hold all of a
    SpinColFull a_tmp; 
    
    // Stream in a 
    for(int spin=0; spin < 4; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }

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
  }  
 

}

// d = SpinProjectDir2Minus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinProjectDir2Minus, 
	      Reference< QDPType<FVec,OLattice< FVec > > > >,
	      OLattice< HVec > > &rhs,
	      const OrderedSubset& s) 
{

  
  const OLattice< FVec >& a = static_cast<const OLattice< FVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned long aoffset=0;
  unsigned long boffset=0;
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
    // Temporary to hold all of a
    SpinColFull a_tmp; 
    
    // Stream in a 
    for(int spin=0; spin < 4; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }

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
  }  
}

// d = SpinProjectDir3Minus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinProjectDir3Minus, 
	      Reference< QDPType<FVec,OLattice< FVec > > > >,
	      OLattice< HVec > > &rhs,
	      const OrderedSubset& s) 
{
  const OLattice< FVec >& a = static_cast<const OLattice< FVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned long aoffset=0;
  unsigned long boffset=0;
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
    // Temporary to hold all of a
    SpinColFull a_tmp; 
    
    // Stream in a 
    for(int spin=0; spin < 4; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }

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
  }  
 

}




// d = SpinReconstructDir0Plus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir0Plus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{

  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned long aoffset=0;
  unsigned long boffset=0;
  const int re=0;
  const int im=1;

  /* 1 + \gamma_0 =  1  0  0  i 
                     0  1  i  0
                     0 -i  1  0
                    -i  0  0  1 
 
    *  ( b2r + i b2i )  =  ( {a2r + a1i} + i{a2i - a1r} )  =  ( b1i - i b1r )
    *  ( b3r + i b3i )     ( {a3r + a0i} + i{a3i - a0r} )     ( b0i - i b0r ) 
   */
  for(int site=s.start(); site <= s.end(); ++site) {
    // Temporary to hold all of a
    SpinColHalf a_tmp; 
    
    // Stream in a 
    for(int spin=0; spin < 2; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }

    // Output Component 0
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[0][col][re];
      bptr[ boffset++ ] = a_tmp[0][col][im];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[1][col][re];
      bptr[ boffset++ ] = a_tmp[1][col][im];
    }

    // Output Component 2
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] =  a_tmp[1][col][im];
      bptr[ boffset++ ] = -a_tmp[1][col][re];
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] =  a_tmp[0][col][im];
      bptr[ boffset++ ] = -a_tmp[0][col][re];
    }


  }  
  
}

// d = SpinReconstructDir1Plus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir1Plus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{

  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned long aoffset=0;
  unsigned long boffset=0;
  const int re=0;
  const int im=1;

  /* 1 + \gamma_1 =  1  0  0 -1 
                     0  1  1  0
                     0  1  1  0
                    -1  0  0  1 
 

   *   ( b2r + i b2i )  =  ( {a2r + a1r} + i{a2i + a1i} )  =  (   b1r + i b1i )
   *   ( b3r + i b3i )     ( {a3r - a0r} + i{a3i - a0i} )     ( - b0r - i b0i ) 
  
  */
  for(int site=s.start(); site <= s.end(); ++site) {
    // Temporary to hold all of a
    SpinColHalf a_tmp; 
    
    // Stream in a 
    for(int spin=0; spin < 2; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }

    // Output Component 0
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[0][col][re];
      bptr[ boffset++ ] = a_tmp[0][col][im];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[1][col][re];
      bptr[ boffset++ ] = a_tmp[1][col][im];
    }

    // Output Component 2
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] =  a_tmp[1][col][re];
      bptr[ boffset++ ] =  a_tmp[1][col][im];
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = -a_tmp[0][col][re];
      bptr[ boffset++ ] = -a_tmp[0][col][im];
    }

  }  

}

// d = SpinReconstructDir2Plus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir2Plus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{


  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned long aoffset=0;
  unsigned long boffset=0;
  const int re=0;
  const int im=1;

  /* 1 + \gamma_2 =  1  0  i  0 
                     0  1  0 -i
                    -i  0  1  0
                     0  i  0  1 
		     
   *  ( b2r + i b2i )  =  ( {a2r + a0i} + i{a2i - a0r} )  =  (   b0i - i b0r )
   *  ( b3r + i b3i )     ( {a3r - a1i} + i{a3i + a1r} )     ( - b1i + i b1r ) 
  */
  for(int site=s.start(); site <= s.end(); ++site) {
    // Temporary to hold all of a
    SpinColHalf a_tmp; 
    
    // Stream in a 
    for(int spin=0; spin < 2; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }

    // Output Component 0
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[0][col][re];
      bptr[ boffset++ ] = a_tmp[0][col][im];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[1][col][re];
      bptr[ boffset++ ] = a_tmp[1][col][im];
    }

    // Output Component 2
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] =   a_tmp[0][col][im];
      bptr[ boffset++ ] =  -a_tmp[0][col][re];
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = -a_tmp[1][col][im];
      bptr[ boffset++ ] =  a_tmp[1][col][re];
    }

  }  

}

// d = SpinReconstructDir3Plus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir3Plus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{

  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned long aoffset=0;
  unsigned long boffset=0;
  const int re=0;
  const int im=1;

  /*                              ( 1  0  1  0)  ( a0 )    ( a0 + a2 )
   *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  0  1)  ( a1 )  = ( a1 + a3 )
   *                    3         ( 1  0  1  0)  ( a2 )    ( a2 + a0 )
   *                              ( 0  1  0  1)  ( a3 )    ( a3 + a1 )
   
   * The bottom components of be may be reconstructed using the formula
   
   *   ( b2r + i b2i )  =  ( {a2r + a0r} + i{a2i + a0i} )  =  ( b0r + i b0i )
   *   ( b3r + i b3i )     ( {a3r + a1r} + i{a3i + a1i} )     ( b1r + i b1i ) 
   */

  for(int site=s.start(); site <= s.end(); ++site) {
    // Temporary to hold all of a
    SpinColHalf a_tmp; 
    
    // Stream in a 
    for(int spin=0; spin < 2; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }

    // Output Component 0
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[0][col][re];
      bptr[ boffset++ ] = a_tmp[0][col][im];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[1][col][re];
      bptr[ boffset++ ] = a_tmp[1][col][im];
    }

    // Output Component 2
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] =   a_tmp[0][col][re];
      bptr[ boffset++ ] =   a_tmp[0][col][im];
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] =  a_tmp[1][col][re];
      bptr[ boffset++ ] =  a_tmp[1][col][im];
    }

  }  


}

// d = SpinReconstructDir0Minus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir0Minus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{

  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned long aoffset=0;
  unsigned long boffset=0;
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
    // Temporary to hold all of a
    SpinColHalf a_tmp; 
    
    // Stream in a 
    for(int spin=0; spin < 2; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }

    // Output Component 0
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[0][col][re];
      bptr[ boffset++ ] = a_tmp[0][col][im];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[1][col][re];
      bptr[ boffset++ ] = a_tmp[1][col][im];
    }

    // Output Component 2
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = -a_tmp[1][col][im];
      bptr[ boffset++ ] =  a_tmp[1][col][re];
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = -a_tmp[0][col][im];
      bptr[ boffset++ ] =  a_tmp[0][col][re];
    }


  }  
  
}

// d = SpinReconstructDir1Minus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir1Minus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{

  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned long aoffset=0;
  unsigned long boffset=0;
  const int re=0;
  const int im=1;

  /*                              ( 1  0  0  1)  ( a0 )    ( a0 + a3 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1 -1  0)  ( a1 )  = ( a1 - a2 )
   *                    1         ( 0 -1  1  0)  ( a2 )    ( a2 - a1 )
   *                              ( 1  0  0  1)  ( a3 )    ( a3 + a0 )
	 
   * The bottom components of be may be reconstructed using the formula

   *  ( b2r + i b2i )  =  ( {a2r - a1r} + i{a2i - a1i} )  =  ( - b1r - i b1i )
   *  ( b3r + i b3i )     ( {a3r + a0r} + i{a3i + a0i} )     (   b0r + i b0i ) 
   */

  for(int site=s.start(); site <= s.end(); ++site) {
    // Temporary to hold all of a
    SpinColHalf a_tmp; 
    
    // Stream in a 
    for(int spin=0; spin < 2; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }

    // Output Component 0
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[0][col][re];
      bptr[ boffset++ ] = a_tmp[0][col][im];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[1][col][re];
      bptr[ boffset++ ] = a_tmp[1][col][im];
    }

    // Output Component 2
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] =  -a_tmp[1][col][re];
      bptr[ boffset++ ] =  -a_tmp[1][col][im];
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] =  a_tmp[0][col][re];
      bptr[ boffset++ ] =  a_tmp[0][col][im];
    }

  }  

}

// d = SpinReconstructDir2Minus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir2Minus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{


  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned long aoffset=0;
  unsigned long boffset=0;
  const int re=0;
  const int im=1;

  /*                              ( 1  0 -i  0)  ( a0 )    ( a0 - i a2 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1  0  i)  ( a1 )  = ( a1 + i a3 )
   *                    2         ( i  0  1  0)  ( a2 )    ( a2 + i a0 )
   *                              ( 0 -i  0  1)  ( a3 )    ( a3 - i a1 )

   * The bottom components of be may be reconstructed using the formula
   *  ( b2r + i b2i )  =  ( {a2r - a0i} + i{a2i + a0r} )  =  ( - b0i + i b0r )
   *  ( b3r + i b3i )     ( {a3r + a1i} + i{a3i - a1r} )     (   b1i - i b1r )
   */

  for(int site=s.start(); site <= s.end(); ++site) {
    // Temporary to hold all of a
    SpinColHalf a_tmp; 
    
    // Stream in a 
    for(int spin=0; spin < 2; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }

    // Output Component 0
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[0][col][re];
      bptr[ boffset++ ] = a_tmp[0][col][im];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[1][col][re];
      bptr[ boffset++ ] = a_tmp[1][col][im];
    }

    // Output Component 2
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] =  -a_tmp[0][col][im];
      bptr[ boffset++ ] =   a_tmp[0][col][re];
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] =  a_tmp[1][col][im];
      bptr[ boffset++ ] = -a_tmp[1][col][re];
    }

  }  

}

// d = SpinReconstructDir3Minus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir3Minus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{

  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned long aoffset=0;
  unsigned long boffset=0;
  const int re=0;
  const int im=1;
  /*                              ( 1  0 -1  0)  ( a0 )    ( a0 - a2 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1  0 -1)  ( a1 )  = ( a1 - a3 )
   *                    3         (-1  0  1  0)  ( a2 )    ( a2 - a0 )
   *                              ( 0 -1  0  1)  ( a3 )    ( a3 - a1 )
      
   * The bottom components of be may be reconstructed using the formula
   *  ( b2r + i b2i )  =  ( {a2r - a0r} + i{a2i - a0i} )  =  ( - b0r - i b0i )
   *  ( b3r + i b3i )     ( {a3r - a1r} + i{a3i - a1i} )     ( - b1r - i b1i ) 
   */

  for(int site=s.start(); site <= s.end(); ++site) {
    // Temporary to hold all of a
    SpinColHalf a_tmp; 
    
    // Stream in a 
    for(int spin=0; spin < 2; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }

    // Output Component 0
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[0][col][re];
      bptr[ boffset++ ] = a_tmp[0][col][im];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] = a_tmp[1][col][re];
      bptr[ boffset++ ] = a_tmp[1][col][im];
    }

    // Output Component 2
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] =  -a_tmp[0][col][re];
      bptr[ boffset++ ] =  -a_tmp[0][col][im];
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] =  -a_tmp[1][col][re];
      bptr[ boffset++ ] =  -a_tmp[1][col][im];
    }

  }  


}



// d += SpinReconstructDir0Plus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAddAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir0Plus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{


  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned long aoffset=0;
  unsigned long boffset=0;
  const int re=0;
  const int im=1;

  /* 1 + \gamma_0 =  1  0  0  i 
                     0  1  i  0
                     0 -i  1  0
                    -i  0  0  1 
 
    *  ( b2r + i b2i )  =  ( {a2r + a1i} + i{a2i - a1r} )  =  ( b1i - i b1r )
    *  ( b3r + i b3i )     ( {a3r + a0i} + i{a3i - a0r} )     ( b0i - i b0r ) 
   */
  for(int site=s.start(); site <= s.end(); ++site) {
    // Temporary to hold all of a
    SpinColHalf a_tmp; 
    
    // Stream in a 
    for(int spin=0; spin < 2; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }

    // Output Component 0
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] += a_tmp[0][col][re];
      bptr[ boffset++ ] += a_tmp[0][col][im];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] += a_tmp[1][col][re];
      bptr[ boffset++ ] += a_tmp[1][col][im];
    }

    // Output Component 2
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] +=  a_tmp[1][col][im];
      bptr[ boffset++ ] -=  a_tmp[1][col][re];
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] +=  a_tmp[0][col][im];
      bptr[ boffset++ ] -=  a_tmp[0][col][re];
    }


  }  

}

// d += SpinReconstructDir1Plus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAddAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir1Plus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{

  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned long aoffset=0;
  unsigned long boffset=0;
  const int re=0;
  const int im=1;

  /* 1 + \gamma_1 =  1  0  0 -1 
                     0  1  1  0
                     0  1  1  0
                    -1  0  0  1 
 

   *   ( b2r + i b2i )  =  ( {a2r + a1r} + i{a2i + a1i} )  =  (   b1r + i b1i )
   *   ( b3r + i b3i )     ( {a3r - a0r} + i{a3i - a0i} )     ( - b0r - i b0i ) 
  
  */
  for(int site=s.start(); site <= s.end(); ++site) {
    // Temporary to hold all of a
    SpinColHalf a_tmp; 
    
    // Stream in a 
    for(int spin=0; spin < 2; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }

    // Output Component 0
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] += a_tmp[0][col][re];
      bptr[ boffset++ ] += a_tmp[0][col][im];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] += a_tmp[1][col][re];
      bptr[ boffset++ ] += a_tmp[1][col][im];
    }

    // Output Component 2
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] +=  a_tmp[1][col][re];
      bptr[ boffset++ ] +=  a_tmp[1][col][im];
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] -= a_tmp[0][col][re];
      bptr[ boffset++ ] -= a_tmp[0][col][im];
    }

  }  

}

// d += SpinReconstructDir2Plus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAddAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir2Plus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{
  
  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned long aoffset=0;
  unsigned long boffset=0;
  const int re=0;
  const int im=1;
  
  /* 1 + \gamma_2 =  1  0  i  0 
                     0  1  0 -i
                    -i  0  1  0
                     0  i  0  1 
		     
   *  ( b2r + i b2i )  =  ( {a2r + a0i} + i{a2i - a0r} )  =  (   b0i - i b0r )
   *  ( b3r + i b3i )     ( {a3r - a1i} + i{a3i + a1r} )     ( - b1i + i b1r ) 
  */
  for(int site=s.start(); site <= s.end(); ++site) {
    // Temporary to hold all of a
    SpinColHalf a_tmp; 
    
    // Stream in a 
    for(int spin=0; spin < 2; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }

    // Output Component 0
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] += a_tmp[0][col][re];
      bptr[ boffset++ ] += a_tmp[0][col][im];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] += a_tmp[1][col][re];
      bptr[ boffset++ ] += a_tmp[1][col][im];
    }

    // Output Component 2
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] +=   a_tmp[0][col][im];
      bptr[ boffset++ ] -=   a_tmp[0][col][re];
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] -= a_tmp[1][col][im];
      bptr[ boffset++ ] +=  a_tmp[1][col][re];
    }

  }  

}

// d += SpinReconstructDir3Plus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAddAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir3Plus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{

  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned long aoffset=0;
  unsigned long boffset=0;
  const int re=0;
  const int im=1;

  /*                              ( 1  0  1  0)  ( a0 )    ( a0 + a2 )
   *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  0  1)  ( a1 )  = ( a1 + a3 )
   *                    3         ( 1  0  1  0)  ( a2 )    ( a2 + a0 )
   *                              ( 0  1  0  1)  ( a3 )    ( a3 + a1 )
   
   * The bottom components of be may be reconstructed using the formula
   
   *   ( b2r + i b2i )  =  ( {a2r + a0r} + i{a2i + a0i} )  =  ( b0r + i b0i )
   *   ( b3r + i b3i )     ( {a3r + a1r} + i{a3i + a1i} )     ( b1r + i b1i ) 
   */

  for(int site=s.start(); site <= s.end(); ++site) {
    // Temporary to hold all of a
    SpinColHalf a_tmp; 
    
    // Stream in a 
    for(int spin=0; spin < 2; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }

    // Output Component 0
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] += a_tmp[0][col][re];
      bptr[ boffset++ ] += a_tmp[0][col][im];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] += a_tmp[1][col][re];
      bptr[ boffset++ ] += a_tmp[1][col][im];
    }

    // Output Component 2
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] +=   a_tmp[0][col][re];
      bptr[ boffset++ ] +=   a_tmp[0][col][im];
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] +=  a_tmp[1][col][re];
      bptr[ boffset++ ] +=  a_tmp[1][col][im];
    }

  }  

}

// d += SpinReconstructDir0Minus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAddAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir0Minus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{


  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned long aoffset=0;
  unsigned long boffset=0;
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
    // Temporary to hold all of a
    SpinColHalf a_tmp; 
    
    // Stream in a 
    for(int spin=0; spin < 2; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }

    // Output Component 0
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] += a_tmp[0][col][re];
      bptr[ boffset++ ] += a_tmp[0][col][im];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] += a_tmp[1][col][re];
      bptr[ boffset++ ] += a_tmp[1][col][im];
    }

    // Output Component 2
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] -= a_tmp[1][col][im];
      bptr[ boffset++ ] +=  a_tmp[1][col][re];
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] -= a_tmp[0][col][im];
      bptr[ boffset++ ] +=  a_tmp[0][col][re];
    }

  }  
}

// d += SpinReconstructDir1Minus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAddAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir1Minus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{

  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned long aoffset=0;
  unsigned long boffset=0;
  const int re=0;
  const int im=1;

  /*                              ( 1  0  0  1)  ( a0 )    ( a0 + a3 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1 -1  0)  ( a1 )  = ( a1 - a2 )
   *                    1         ( 0 -1  1  0)  ( a2 )    ( a2 - a1 )
   *                              ( 1  0  0  1)  ( a3 )    ( a3 + a0 )
	 
   * The bottom components of be may be reconstructed using the formula

   *  ( b2r + i b2i )  =  ( {a2r - a1r} + i{a2i - a1i} )  =  ( - b1r - i b1i )
   *  ( b3r + i b3i )     ( {a3r + a0r} + i{a3i + a0i} )     (   b0r + i b0i ) 
   */

  for(int site=s.start(); site <= s.end(); ++site) {
    // Temporary to hold all of a
    SpinColHalf a_tmp; 
    
    // Stream in a 
    for(int spin=0; spin < 2; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }

    // Output Component 0
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] += a_tmp[0][col][re];
      bptr[ boffset++ ] += a_tmp[0][col][im];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] += a_tmp[1][col][re];
      bptr[ boffset++ ] += a_tmp[1][col][im];
    }

    // Output Component 2
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] -=  a_tmp[1][col][re];
      bptr[ boffset++ ] -=  a_tmp[1][col][im];
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] +=  a_tmp[0][col][re];
      bptr[ boffset++ ] +=  a_tmp[0][col][im];
    }

  }  

}

// d += SpinReconstructDir2Minus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAddAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir2Minus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{


  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned long aoffset=0;
  unsigned long boffset=0;
  const int re=0;
  const int im=1;

  /*                              ( 1  0 -i  0)  ( a0 )    ( a0 - i a2 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1  0  i)  ( a1 )  = ( a1 + i a3 )
   *                    2         ( i  0  1  0)  ( a2 )    ( a2 + i a0 )
   *                              ( 0 -i  0  1)  ( a3 )    ( a3 - i a1 )

   * The bottom components of be may be reconstructed using the formula
   *  ( b2r + i b2i )  =  ( {a2r - a0i} + i{a2i + a0r} )  =  ( - b0i + i b0r )
   *  ( b3r + i b3i )     ( {a3r + a1i} + i{a3i - a1r} )     (   b1i - i b1r )
   */

  for(int site=s.start(); site <= s.end(); ++site) {
    // Temporary to hold all of a
    SpinColHalf a_tmp; 
    
    // Stream in a 
    for(int spin=0; spin < 2; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }

    // Output Component 0
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] += a_tmp[0][col][re];
      bptr[ boffset++ ] += a_tmp[0][col][im];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] += a_tmp[1][col][re];
      bptr[ boffset++ ] += a_tmp[1][col][im];
    }

    // Output Component 2
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] -=  a_tmp[0][col][im];
      bptr[ boffset++ ] +=   a_tmp[0][col][re];
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] +=  a_tmp[1][col][im];
      bptr[ boffset++ ] -= a_tmp[1][col][re];
    }

  }  

}

// d += SpinReconstructDir3Minus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAddAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir3Minus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{


  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned long aoffset=0;
  unsigned long boffset=0;
  const int re=0;
  const int im=1;
  /*                              ( 1  0 -1  0)  ( a0 )    ( a0 - a2 )
   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1  0 -1)  ( a1 )  = ( a1 - a3 )
   *                    3         (-1  0  1  0)  ( a2 )    ( a2 - a0 )
   *                              ( 0 -1  0  1)  ( a3 )    ( a3 - a1 )
      
   * The bottom components of be may be reconstructed using the formula
   *  ( b2r + i b2i )  =  ( {a2r - a0r} + i{a2i - a0i} )  =  ( - b0r - i b0i )
   *  ( b3r + i b3i )     ( {a3r - a1r} + i{a3i - a1i} )     ( - b1r - i b1i ) 
   */

  for(int site=s.start(); site <= s.end(); ++site) {
    // Temporary to hold all of a
    SpinColHalf a_tmp; 
    
    // Stream in a 
    for(int spin=0; spin < 2; spin++) { 
      for(int col=0; col < 3; col++) { 
	a_tmp[spin][col][re] = aptr[(aoffset++)];
	a_tmp[spin][col][im] = aptr[(aoffset++)];
      }
    }

    // Output Component 0
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] += a_tmp[0][col][re];
      bptr[ boffset++ ] += a_tmp[0][col][im];
    }
    
    // Output Component 1
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] += a_tmp[1][col][re];
      bptr[ boffset++ ] += a_tmp[1][col][im];
    }

    // Output Component 2
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] -=  a_tmp[0][col][re];
      bptr[ boffset++ ] -=  a_tmp[0][col][im];
    }
    
    // Output Component 3
    for(int col=0; col < 3; col++) {
      bptr[ boffset++ ] -=  a_tmp[1][col][re];
      bptr[ boffset++ ] -=  a_tmp[1][col][im];
    }

  }  

}

QDP_END_NAMESPACE();

#endif
