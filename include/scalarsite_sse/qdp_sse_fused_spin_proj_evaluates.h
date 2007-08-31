#ifndef QDP_SSE_FUSED_SPIN_PROJ_H
#define QDP_SSE_FUSED_SPIN_PROJ_H

#include "sse_mult_adj_su3_mat_hwvec.h"

/* Evaluates for things like adj(u)*spinProjectDir0Plus(y) */
using namespace QDP;
namespace QDP {

typedef PScalar< PColorMatrix< RComplex<REAL32>, 3> > SU3Mat;


// HalfVec = adj(u)*SpinProjectDir0Plus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& d,
              const OpAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                FnAdjMultSprojDir0Plus,
	               
	                UnaryNode< OpIdentity, 
	                  Reference< QDPType< SU3Mat, OLattice< SU3Mat > > > >,
	               
	                  UnaryNode< OpIdentity,
                          Reference< QDPType< FVec,   OLattice< FVec > > > >
                      >,
	              OLattice< HVec > 
                    >&rhs,
	      const Subset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left().child());
  const OLattice< FVec >& a = static_cast< const OLattice< FVec >& >(rhs.expression().right().child());

  if( s.hasOrderedRep() ) {
    for(int site = s.start() ; site <= s.end(); site++) { 
      HVec tmp ; 
      inlineSpinProjDir0Plus( (REAL32 *)&(a.elem(site).elem(0).elem(0).real()),
			      (REAL32 *)&(tmp.elem(0).elem(0).real()),
			      1);
      
      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());
      half_wilson_vectorf *dh = (half_wilson_vectorf *)&( d.elem(site).elem(0).elem(0).real());
      
      intrin_sse_mult_adj_su3_mat_hwvec(um, tmph, dh);
    }
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int site=tab[j];
      HVec tmp ; 
      inlineSpinProjDir0Plus( (REAL32 *)&(a.elem(site).elem(0).elem(0).real()),
			      (REAL32 *)&(tmp.elem(0).elem(0).real()),
			      1);
      
      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());
      half_wilson_vectorf *dh = (half_wilson_vectorf *)&( d.elem(site).elem(0).elem(0).real());
      
      intrin_sse_mult_adj_su3_mat_hwvec(um, tmph, dh);
    
    }
  }

}

// HalfVec = adj(u)*SpinProjectDir0Minus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& d,
              const OpAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                FnAdjMultSprojDir0Minus,
	               
	                UnaryNode< OpIdentity, 
	                  Reference< QDPType< SU3Mat, OLattice< SU3Mat > > > >,
	               
	                  UnaryNode< OpIdentity,
                          Reference< QDPType< FVec,   OLattice< FVec > > > >
                      >,
	              OLattice< HVec > 
                    >&rhs,
	      const Subset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left().child());
  const OLattice< FVec >& a = static_cast< const OLattice< FVec >& >(rhs.expression().right().child());

  if( s.hasOrderedRep() ) {
    for(int site = s.start() ; site <= s.end(); site++) { 
      HVec tmp ;
      inlineSpinProjDir0Minus( (REAL32 *)&(a.elem(site).elem(0).elem(0).real()),
			       (REAL32 *)&(tmp.elem(0).elem(0).real()),
			       1);
      
      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());
      half_wilson_vectorf *dh = (half_wilson_vectorf *)&( d.elem(site).elem(0).elem(0).real());
      intrin_sse_mult_adj_su3_mat_hwvec(um, tmph, dh);
   
    }
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int site=tab[j];
      
      HVec tmp ;
      inlineSpinProjDir0Minus( (REAL32 *)&(a.elem(site).elem(0).elem(0).real()),
			       (REAL32 *)&(tmp.elem(0).elem(0).real()),
			       1);
      
      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());
      half_wilson_vectorf *dh = (half_wilson_vectorf *)&( d.elem(site).elem(0).elem(0).real());
      intrin_sse_mult_adj_su3_mat_hwvec(um, tmph, dh);
      
    }
  }

}

// HalfVec = adj(u)*SpinProjectDir1Plus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& d,
              const OpAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                FnAdjMultSprojDir1Plus,
	               
	                UnaryNode< OpIdentity, 
	                  Reference< QDPType< SU3Mat, OLattice< SU3Mat > > > >,
	               
	                  UnaryNode< OpIdentity,
                          Reference< QDPType< FVec,   OLattice< FVec > > > >
                      >,
	              OLattice< HVec > 
                    >&rhs,
	      const Subset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left().child());
  const OLattice< FVec >& a = static_cast< const OLattice< FVec >& >(rhs.expression().right().child());

  
  if( s.hasOrderedRep() ) {
    for(int site = s.start() ; site <= s.end(); site++) { 
      HVec tmp ;
      inlineSpinProjDir1Plus( (REAL32 *)&(a.elem(site).elem(0).elem(0).real()),
			      (REAL32 *)&(tmp.elem(0).elem(0).real()),
			      1);
      
      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());
      half_wilson_vectorf *dh = (half_wilson_vectorf *)&( d.elem(site).elem(0).elem(0).real());
      intrin_sse_mult_adj_su3_mat_hwvec(um, tmph, dh);
      
      
    }
  }
  else { 
    
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int site=tab[j];
      
      HVec tmp ;
      inlineSpinProjDir1Plus( (REAL32 *)&(a.elem(site).elem(0).elem(0).real()),
			      (REAL32 *)&(tmp.elem(0).elem(0).real()),
			      1);
      
      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());
      half_wilson_vectorf *dh = (half_wilson_vectorf *)&( d.elem(site).elem(0).elem(0).real());
      intrin_sse_mult_adj_su3_mat_hwvec(um, tmph, dh);
      
    }
  }
}

// HalfVec = adj(u)*SpinProjectDir1Minus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& d,
              const OpAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                FnAdjMultSprojDir1Minus,
	               
	                UnaryNode< OpIdentity, 
	                  Reference< QDPType< SU3Mat, OLattice< SU3Mat > > > >,
	               
	                  UnaryNode< OpIdentity,
                          Reference< QDPType< FVec,   OLattice< FVec > > > >
                      >,
	              OLattice< HVec > 
                    >&rhs,
	      const Subset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left().child());
  const OLattice< FVec >& a = static_cast< const OLattice< FVec >& >(rhs.expression().right().child());

  if( s.hasOrderedRep() ) {
    for(int site = s.start() ; site <= s.end(); site++) { 
      HVec tmp ;
      inlineSpinProjDir1Minus( (REAL32 *)&(a.elem(site).elem(0).elem(0).real()),
			       (REAL32 *)&(tmp.elem(0).elem(0).real()),
			       1);
      
      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());
      half_wilson_vectorf *dh = (half_wilson_vectorf *)&( d.elem(site).elem(0).elem(0).real());
      intrin_sse_mult_adj_su3_mat_hwvec(um, tmph, dh);
   
    }
  }
  else { 

    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int site=tab[j];
      
      HVec tmp ;
      inlineSpinProjDir1Minus( (REAL32 *)&(a.elem(site).elem(0).elem(0).real()),
			       (REAL32 *)&(tmp.elem(0).elem(0).real()),
			       1);
      
      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());
      half_wilson_vectorf *dh = (half_wilson_vectorf *)&( d.elem(site).elem(0).elem(0).real());
      intrin_sse_mult_adj_su3_mat_hwvec(um, tmph, dh);
      
    }
  }
}


// HalfVec = adj(u)*SpinProjectDir2Plus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& d,
              const OpAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                FnAdjMultSprojDir2Plus,
	               
	                UnaryNode< OpIdentity, 
	                  Reference< QDPType< SU3Mat, OLattice< SU3Mat > > > >,
	               
	                  UnaryNode< OpIdentity,
                          Reference< QDPType< FVec,   OLattice< FVec > > > >
                      >,
	              OLattice< HVec > 
                    >&rhs,
	      const Subset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left().child());
  const OLattice< FVec >& a = static_cast< const OLattice< FVec >& >(rhs.expression().right().child());

  if( s.hasOrderedRep() ) {
    for(int site = s.start() ; site <= s.end(); site++) { 
      HVec tmp ;
      inlineSpinProjDir2Plus( (REAL32 *)&(a.elem(site).elem(0).elem(0).real()),
			      (REAL32 *)&(tmp.elem(0).elem(0).real()),
			      1);
      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());
      half_wilson_vectorf *dh = (half_wilson_vectorf *)&( d.elem(site).elem(0).elem(0).real());
      intrin_sse_mult_adj_su3_mat_hwvec(um, tmph, dh);
      
    }
  }
  else { 
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int site=tab[j];
      HVec tmp ;
      inlineSpinProjDir2Plus( (REAL32 *)&(a.elem(site).elem(0).elem(0).real()),
			      (REAL32 *)&(tmp.elem(0).elem(0).real()),
			      1);

      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());
      half_wilson_vectorf *dh = (half_wilson_vectorf *)&( d.elem(site).elem(0).elem(0).real());
      intrin_sse_mult_adj_su3_mat_hwvec(um, tmph, dh);
      
    }
  }
}

// HalfVec = adj(u)*SpinProjectDir2Minus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& d,
              const OpAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                FnAdjMultSprojDir2Minus,
	               
	                UnaryNode< OpIdentity, 
	                  Reference< QDPType< SU3Mat, OLattice< SU3Mat > > > >,
	               
	                  UnaryNode< OpIdentity,
                          Reference< QDPType< FVec,   OLattice< FVec > > > >
                      >,
	              OLattice< HVec > 
                    >&rhs,
	      const Subset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left().child());
  const OLattice< FVec >& a = static_cast< const OLattice< FVec >& >(rhs.expression().right().child());

  if( s.hasOrderedRep() ) {
    for(int site = s.start() ; site <= s.end(); site++) { 
      HVec tmp ;
      inlineSpinProjDir2Minus( (REAL32 *)&(a.elem(site).elem(0).elem(0).real()),
			       (REAL32 *)&(tmp.elem(0).elem(0).real()),
			       1);
      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());
      half_wilson_vectorf *dh = (half_wilson_vectorf *)&( d.elem(site).elem(0).elem(0).real());
      intrin_sse_mult_adj_su3_mat_hwvec(um, tmph, dh);
      
      
    }
  }
  else { 
    
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int site=tab[j];
      
      HVec tmp ;
      inlineSpinProjDir2Minus( (REAL32 *)&(a.elem(site).elem(0).elem(0).real()),
			       (REAL32 *)&(tmp.elem(0).elem(0).real()),
			       1);
      
      
      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());
      half_wilson_vectorf *dh = (half_wilson_vectorf *)&( d.elem(site).elem(0).elem(0).real());
      intrin_sse_mult_adj_su3_mat_hwvec(um, tmph, dh);
      
      
    }
  }
}

// HalfVec = adj(u)*SpinProjectDir3Plus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& d,
              const OpAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                FnAdjMultSprojDir3Plus,
	               
	                UnaryNode< OpIdentity, 
	                  Reference< QDPType< SU3Mat, OLattice< SU3Mat > > > >,
	               
	                  UnaryNode< OpIdentity,
                          Reference< QDPType< FVec,   OLattice< FVec > > > >
                      >,
	              OLattice< HVec > 
                    >&rhs,
	      const Subset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left().child());
  const OLattice< FVec >& a = static_cast< const OLattice< FVec >& >(rhs.expression().right().child());

  if( s.hasOrderedRep() ) {
    for(int site = s.start() ; site <= s.end(); site++) { 
      HVec tmp ;
      inlineSpinProjDir3Plus( (REAL32 *)&(a.elem(site).elem(0).elem(0).real()),
			      (REAL32 *)&(tmp.elem(0).elem(0).real()),
			      1);
      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());
      half_wilson_vectorf *dh = (half_wilson_vectorf *)&( d.elem(site).elem(0).elem(0).real());
      intrin_sse_mult_adj_su3_mat_hwvec(um, tmph, dh);
      
      
    }
  }
  else { 

    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int site=tab[j];
      HVec tmp ;
      inlineSpinProjDir3Plus( (REAL32 *)&(a.elem(site).elem(0).elem(0).real()),
			      (REAL32 *)&(tmp.elem(0).elem(0).real()),
			      1);
      
      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());
      half_wilson_vectorf *dh = (half_wilson_vectorf *)&( d.elem(site).elem(0).elem(0).real());
      intrin_sse_mult_adj_su3_mat_hwvec(um, tmph, dh);
      
      
    }
  }
}

// HalfVec = adj(u)*SpinProjectDir3Minus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& d,
              const OpAssign& op,
              const QDPExpr<
	              BinaryNode< 
	                FnAdjMultSprojDir3Minus,
	               
	                UnaryNode< OpIdentity, 
	                  Reference< QDPType< SU3Mat, OLattice< SU3Mat > > > >,
	               
	                  UnaryNode< OpIdentity,
                          Reference< QDPType< FVec,   OLattice< FVec > > > >
                      >,
	              OLattice< HVec > 
                    >&rhs,
	      const Subset& s)
{
  const OLattice< SU3Mat >& u = static_cast< const OLattice< SU3Mat >& >(rhs.expression().left().child());
  const OLattice< FVec >& a = static_cast< const OLattice< FVec >& >(rhs.expression().right().child());


  if( s.hasOrderedRep() ) {
    for(int site = s.start() ; site <= s.end(); site++) { 
      HVec tmp ;
      inlineSpinProjDir3Minus( (REAL32 *)&(a.elem(site).elem(0).elem(0).real()),
			       (REAL32 *)&(tmp.elem(0).elem(0).real()),
			       1);
      
      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());
      half_wilson_vectorf *dh = (half_wilson_vectorf *)&( d.elem(site).elem(0).elem(0).real());
      intrin_sse_mult_adj_su3_mat_hwvec(um, tmph, dh);
      

    }
  }
  else { 
    
    const int* tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int site=tab[j];
      
      HVec tmp ;
      inlineSpinProjDir3Minus( (REAL32 *)&(a.elem(site).elem(0).elem(0).real()),
			       (REAL32 *)&(tmp.elem(0).elem(0).real()),
			       1);
      
      su3_matrixf* um = (su3_matrixf *)&(u.elem(site).elem().elem(0,0).real());
      half_wilson_vectorf *tmph = (half_wilson_vectorf *)&( tmp.elem(0).elem(0).real());
      half_wilson_vectorf *dh = (half_wilson_vectorf *)&( d.elem(site).elem(0).elem(0).real());
      intrin_sse_mult_adj_su3_mat_hwvec(um, tmph, dh);
      
    }
  }

}

} // namespace QDP;

#endif
