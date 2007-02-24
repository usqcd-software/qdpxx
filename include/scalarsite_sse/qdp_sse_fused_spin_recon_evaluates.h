#ifndef QDP_SSE_FUSED_SPIN_RECON_EVALYATES_H
#define QDP_SSE_FUSED_SPIN_RECON_EVALUATES_H

QDP_BEGIN_NAMESPACE(QDP);


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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());

  if( s.hasOrderedRep() ) { 
    for(int site=s.start(); site <= s.end(); ++site) {
      HVec tmp ;
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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

      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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

      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
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
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(0), tmp.elem(0));
      _inline_mult_su3_mat_vec(u.elem(site).elem(), a.elem(site).elem(1), tmp.elem(1));
      
      
      inlineAddSpinReconDir3Minus( (REAL *)&(tmp.elem(0).elem(0).real()),
				   (REAL *)&(d.elem(site).elem(0).elem(0).real()),
				   1);
    }
  }
}



QDP_END_NAMESPACE();





#endif
