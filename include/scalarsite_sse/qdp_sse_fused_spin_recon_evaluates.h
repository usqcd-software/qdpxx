#ifndef QDP_SSE_FUSED_SPIN_RECON_EVALYATES_H
#define QDP_SSE_FUSED_SPIN_RECON_EVALUATES_H

#include "sse_mult_su3_mat_hwvec.h"

namespace QDP {


// Vec = SpinReconstructDir0Plus( u * psi);
template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                FnSReconDir0PlusProd,
	                Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
                      >,
	              OLattice< FVec > 
                    >&rhs,
	      const Subset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().right());

  if( s.hasOrderedRep() ) { 
    for( int site = s.start(); site <= s.end(); site++) { 
      HVec tmp ;

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());

      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      
      inlineSpinReconDir0Plus( (REAL *)&(tmp.elem(0).elem(0).real()),
			       (REAL *)&(d.elem(site).elem(0).elem(0).real()),
			       1);
    }
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int site=tab[j];
      
      HVec tmp ;

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());

      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      
      inlineSpinReconDir0Plus( (REAL *)&(tmp.elem(0).elem(0).real()),
			       (REAL *)&(d.elem(site).elem(0).elem(0).real()),
			       1);
    }
  }
}

// Vec = SpinReconstructDir0Minus( u * psi);
template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                FnSReconDir0MinusProd,
	                Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
                      >,
	              OLattice< FVec > 
                    >&rhs,
	      const Subset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().right());

  if( s.hasOrderedRep() ) { 
    for(int site=s.start(); site <= s.end(); ++site) {
      HVec tmp ;
      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());

      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      inlineSpinReconDir0Minus( (REAL *)&(tmp.elem(0).elem(0).real()),
				(REAL *)&(d.elem(site).elem(0).elem(0).real()),
				1);
    }
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int site=tab[j];
      
      HVec tmp ;

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());

      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      inlineSpinReconDir0Minus( (REAL *)&(tmp.elem(0).elem(0).real()),
				(REAL *)&(d.elem(site).elem(0).elem(0).real()),
				1);

      
    }
  }
}




// Vec = SpinReconstructDir1Plus( u * psi);
template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                FnSReconDir1PlusProd,
	                Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
                      >,
	              OLattice< FVec > 
                    >&rhs,
	      const Subset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().right());

  if( s.hasOrderedRep() ) { 
    for(int site=s.start(); site <= s.end(); ++site) {
      HVec tmp ;

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());

      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      
      inlineSpinReconDir1Plus( (REAL *)&(tmp.elem(0).elem(0).real()),
			       (REAL *)&(d.elem(site).elem(0).elem(0).real()),
			       1);
    }
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int site=tab[j];
      
      HVec tmp ;

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());

      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      
      inlineSpinReconDir1Plus( (REAL *)&(tmp.elem(0).elem(0).real()),
			       (REAL *)&(d.elem(site).elem(0).elem(0).real()),
			       1);

      
    }
  }

}

// Vec = SpinReconstructDir1Minus( u * psi);
template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                FnSReconDir1MinusProd,
	                Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
                      >,
	              OLattice< FVec > 
                    >&rhs,
	      const Subset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().right());

  if( s.hasOrderedRep() ) { 
    for(int site=s.start(); site <= s.end(); ++site) {
      HVec tmp ;

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());

      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      
      inlineSpinReconDir1Minus( (REAL *)&(tmp.elem(0).elem(0).real()),
				(REAL *)&(d.elem(site).elem(0).elem(0).real()),
				1);

    }
  }
  else { 

    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int site=tab[j];
      
      HVec tmp ;
      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      
      inlineSpinReconDir1Minus( (REAL *)&(tmp.elem(0).elem(0).real()),
				(REAL *)&(d.elem(site).elem(0).elem(0).real()),
				1);
      
      
    }
  }
}



// Vec = SpinReconstructDir2Plus( u * psi);
template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                FnSReconDir2PlusProd,
	                Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
                      >,
	              OLattice< FVec > 
                    >&rhs,
	      const Subset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().right());

  if( s.hasOrderedRep() ) { 
    for(int site=s.start(); site <= s.end(); ++site) {
      HVec tmp ;

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);

      
      inlineSpinReconDir2Plus( (REAL *)&(tmp.elem(0).elem(0).real()),
			       (REAL *)&(d.elem(site).elem(0).elem(0).real()),
			       1);
      

    }
  }
  else { 

    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int site=tab[j];
      
      HVec tmp ;
      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);

      
      
      inlineSpinReconDir2Plus( (REAL *)&(tmp.elem(0).elem(0).real()),
			       (REAL *)&(d.elem(site).elem(0).elem(0).real()),
			       1);
      
      
    }
  }

}

// Vec = SpinReconstructDir2Minus( u * psi);
template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                FnSReconDir2MinusProd,
	                Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
                      >,
	              OLattice< FVec > 
                    >&rhs,
	      const Subset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().right());

  if( s.hasOrderedRep() ) { 
    for(int site=s.start(); site <= s.end(); ++site) {
      HVec tmp ;
      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      
      inlineSpinReconDir2Minus( (REAL *)&(tmp.elem(0).elem(0).real()),
				(REAL *)&(d.elem(site).elem(0).elem(0).real()),
				1);
      
    }
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int site=tab[j];
      
      HVec tmp ;

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      inlineSpinReconDir2Minus( (REAL *)&(tmp.elem(0).elem(0).real()),
				(REAL *)&(d.elem(site).elem(0).elem(0).real()),
			     1);
    }
  }
}



// Vec = SpinReconstructDir3Plus( u * psi);
template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                FnSReconDir3PlusProd,
	                Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
                      >,
	              OLattice< FVec > 
                    >&rhs,
	      const Subset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().right());

  if( s.hasOrderedRep() ) { 
    for(int site=s.start(); site <= s.end(); ++site) {
      HVec tmp ;

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      
      inlineSpinReconDir3Plus( (REAL *)&(tmp.elem(0).elem(0).real()),
			       (REAL *)&(d.elem(site).elem(0).elem(0).real()),
			       1);
      
    }
  }
  else { 
    
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int site=tab[j];
      
      HVec tmp ;

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      
      inlineSpinReconDir3Plus( (REAL *)&(tmp.elem(0).elem(0).real()),
			       (REAL *)&(d.elem(site).elem(0).elem(0).real()),
			       1);
    }
  }
}

// Vec = SpinReconstructDir3Minus( u * psi);
template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                FnSReconDir3MinusProd,
	                Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
                      >,
	              OLattice< FVec > 
                    >&rhs,
	      const Subset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().right());

  if( s.hasOrderedRep() ) { 
    for(int site=s.start(); site <= s.end(); ++site) {
      HVec tmp ;
      
      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      inlineSpinReconDir3Minus( (REAL *)&(tmp.elem(0).elem(0).real()),
				(REAL *)&(d.elem(site).elem(0).elem(0).real()),
				1);
    }
  }
  else { 

    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int site=tab[j];
      
      HVec tmp ;

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);

      
      
      inlineSpinReconDir3Minus( (REAL *)&(tmp.elem(0).elem(0).real()),
				(REAL *)&(d.elem(site).elem(0).elem(0).real()),
				1);
    }
  }
}



// Vec += SpinReconstructDir0Plus( u * psi);
template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAddAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                FnSReconDir0PlusProd,
	                Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
                      >,
	              OLattice< FVec > 
                    >&rhs,
	      const Subset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().right());

  if( s.hasOrderedRep() ) { 
    for(int site=s.start(); site <= s.end(); ++site) {
      HVec tmp ;

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      inlineAddSpinReconDir0Plus( (REAL *)&(tmp.elem(0).elem(0).real()),
				  (REAL *)&(d.elem(site).elem(0).elem(0).real()),
				  1);
    }
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int site=tab[j];
      
      HVec tmp ;

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      
      
      inlineAddSpinReconDir0Plus( (REAL *)&(tmp.elem(0).elem(0).real()),
				  (REAL *)&(d.elem(site).elem(0).elem(0).real()),
				  1);

 
    }
  }

}

// Vec += SpinReconstructDir0Minus( u * psi);
template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAddAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                FnSReconDir0MinusProd,
	                Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
                      >,
	              OLattice< FVec > 
                    >&rhs,
	      const Subset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().right());

  if( s.hasOrderedRep() ) { 
    for(int site=s.start(); site <= s.end(); ++site) {
      HVec tmp ;

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      inlineAddSpinReconDir0Minus( (REAL *)&(tmp.elem(0).elem(0).real()),
				   (REAL *)&(d.elem(site).elem(0).elem(0).real()),
				   1);
    }
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int site=tab[j];
      
      HVec tmp ;

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);

      
      
      inlineAddSpinReconDir0Minus( (REAL *)&(tmp.elem(0).elem(0).real()),
				   (REAL *)&(d.elem(site).elem(0).elem(0).real()),
				   1);
      
    }
  }
}



// Vec += SpinReconstructDir1Plus( u * psi);
template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAddAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                FnSReconDir1PlusProd,
	                Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
                      >,
	              OLattice< FVec > 
                    >&rhs,
	      const Subset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().right());


  if( s.hasOrderedRep() ) { 
    for(int site=s.start(); site <= s.end(); ++site) {
      
      HVec tmp ;

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      inlineAddSpinReconDir1Plus( (REAL *)&(tmp.elem(0).elem(0).real()),
				  (REAL *)&(d.elem(site).elem(0).elem(0).real()),
				  1);
    }
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int site=tab[j];
      
      HVec tmp ;

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      
      inlineAddSpinReconDir1Plus( (REAL *)&(tmp.elem(0).elem(0).real()),
				  (REAL *)&(d.elem(site).elem(0).elem(0).real()),
				  1);
    }
  }
}

// Vec += SpinReconstructDir1Minus( u * psi);
template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAddAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                FnSReconDir1MinusProd,
	                Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
                      >,
	              OLattice< FVec > 
                    >&rhs,
	      const Subset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().right());

  if( s.hasOrderedRep() ) { 
    for(int site=s.start(); site <= s.end(); ++site) {
      HVec tmp ;

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      
      inlineAddSpinReconDir1Minus( (REAL *)&(tmp.elem(0).elem(0).real()),
				   (REAL *)&(d.elem(site).elem(0).elem(0).real()),
				   1);
    }
  }
  else { 
    
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int site=tab[j];
      
       HVec tmp ;

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      
      inlineAddSpinReconDir1Minus( (REAL *)&(tmp.elem(0).elem(0).real()),
				   (REAL *)&(d.elem(site).elem(0).elem(0).real()),
				   1);
    }
  }
}



// Vec += SpinReconstructDir2Plus( u * psi);
template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAddAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                FnSReconDir2PlusProd,
	                Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
                      >,
	              OLattice< FVec > 
                    >&rhs,
	      const Subset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().right());

  if( s.hasOrderedRep() ) { 
    for(int site=s.start(); site <= s.end(); ++site) {
      HVec tmp ;

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      
      inlineAddSpinReconDir2Plus( (REAL *)&(tmp.elem(0).elem(0).real()),
				  (REAL *)&(d.elem(site).elem(0).elem(0).real()),
				  1);
   
    }
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int site=tab[j];
      
      HVec tmp ;

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      inlineAddSpinReconDir2Plus( (REAL *)&(tmp.elem(0).elem(0).real()),
				  (REAL *)&(d.elem(site).elem(0).elem(0).real()),
				  1);
      
      
    }
  }
}

// Vec += SpinReconstructDir2Minus( u * psi);
template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAddAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                FnSReconDir2MinusProd,
	                Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
                      >,
	              OLattice< FVec > 
                    >&rhs,
	      const Subset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().right());

  if( s.hasOrderedRep() ) { 
    for(int site=s.start(); site <= s.end(); ++site) {

      HVec tmp ;
      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      inlineAddSpinReconDir2Minus( (REAL *)&(tmp.elem(0).elem(0).real()),
				   (REAL *)&(d.elem(site).elem(0).elem(0).real()),
				   1);
    }
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int site=tab[j];
      
      HVec tmp ;
      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      
      inlineAddSpinReconDir2Minus( (REAL *)&(tmp.elem(0).elem(0).real()),
				   (REAL *)&(d.elem(site).elem(0).elem(0).real()),
				   1);
    }
  }
}



// Vec += SpinReconstructDir3Plus( u * psi);
template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAddAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                FnSReconDir3PlusProd,
	                Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
                      >,
	              OLattice< FVec > 
                    >&rhs,
	      const Subset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().right());

  if( s.hasOrderedRep() ) { 
    for(int site=s.start(); site <= s.end(); ++site) {
      HVec tmp ;

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      
      inlineAddSpinReconDir3Plus( (REAL *)&(tmp.elem(0).elem(0).real()),
				  (REAL *)&(d.elem(site).elem(0).elem(0).real()),
				  1);
    }
  }
  else { 

    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int site=tab[j];
      
      HVec tmp ;


      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      inlineAddSpinReconDir3Plus( (REAL *)&(tmp.elem(0).elem(0).real()),
				  (REAL *)&(d.elem(site).elem(0).elem(0).real()),
				  1);
      
    }
  }
}

// Vec += SpinReconstructDir3Minus( u * psi);
template<>
inline
void evaluate(OLattice< FVec >& d,
              const OpAddAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                FnSReconDir3MinusProd,
	                Reference< QDPType< SU3Mat, OLattice< SU3Mat > > >,
	                Reference< QDPType< HVec,   OLattice< HVec > > >
                      >,
	              OLattice< FVec > 
                    >&rhs,
	      const Subset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left());
  const OLattice< HVec >& a = static_cast< const OLattice< HVec >& >(rhs.expression().right());

  if( s.hasOrderedRep() ) { 
    for(int site=s.start(); site <= s.end(); ++site) {
      HVec tmp ;

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      
      
      inlineAddSpinReconDir3Minus( (REAL *)&(tmp.elem(0).elem(0).real()),
				   (REAL *)&(d.elem(site).elem(0).elem(0).real()),
				   1);

    }
  }
  else { 

    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int site=tab[j];
      
      HVec tmp ;

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *ah = (half_wilson_vectorf *)&( a.elem(site).elem(0).elem(0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());


      intrin_sse_mult_su3_mat_hwvec(um, ah, tmph);
      
      inlineAddSpinReconDir3Minus( (REAL *)&(tmp.elem(0).elem(0).real()),
				   (REAL *)&(d.elem(site).elem(0).elem(0).real()),
				   1);
    }
  }
}



} // namespace QDP;





#endif
