// $Id: qdp_scalarsite_linalg_double.cc,v 1.1 2008-06-26 23:20:56 bjoo Exp $

/*! @file
 * @brief Intel SSE optimizations
 * 
 * SSE optimizations of basic operations
 */


#include "qdp.h"


// These SSE asm instructions are only supported under GCC/G++
#include "scalarsite_sse/qdp_scalarsite_sse_linalg_double.h"

namespace QDP {



//-------------------------------------------------------------------
// Specialization to optimize the case   
//    LatticeColorMatrix[ Subset] = LatticeColorMatrix * LatticeColorMatrix
template<>
void evaluate(OLattice< DCol >& d, 
	      const OpAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	                    Reference<QDPType< DCol, OLattice< DCol > > >, 
	                    Reference<QDPType< DCol, OLattice< DCol > > > >,
	                    OLattice< DCol > >& rhs,
	      const Subset& s)
{
//  cout << "call single site QDP_M_eq_M_times_M" << endl;

  typedef OLattice< DCol >    C;

  const C& l = static_cast<const C&>(rhs.expression().left());
  const C& r = static_cast<const C&>(rhs.expression().right());

  if( s.hasOrderedRep() ) { 
    int n_3mat=s.end() - s.start() + 1;
    REAL64 *lm = (REAL64 *)&(l.elem(s.start()).elem().elem(0,0).real());
    REAL64 *rm = (REAL64 *)&(r.elem(s.start()).elem().elem(0,0).real());
    REAL64 *dm = (REAL64 *)&(d.elem(s.start()).elem().elem(0,0).real());
    ssed_m_eq_mm(dm,lm,rm, n_3mat);
  }
  else { 
    const int *tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); ++j) {
      int i = tab[j];
      REAL64 *lm = (REAL64 *)&(l.elem(i).elem().elem(0,0).real());
      REAL64 *rm = (REAL64 *)&(r.elem(i).elem().elem(0,0).real());
      REAL64 *dm = (REAL64 *)&(d.elem(i).elem().elem(0,0).real());

      ssed_m_eq_mm(dm,lm, rm, 1);

    }
  }
}


// Specialization to optimize the case   
//    LatticeColorMatrix[Subset] = adj(LatticeColorMatrix) * LatticeColorMatrix
template<>
void evaluate(OLattice< DCol >& d, 
	      const OpAssign& op, 
	      const QDPExpr<BinaryNode<OpAdjMultiply, 
	                    UnaryNode<OpIdentity, Reference<QDPType< DCol, OLattice< DCol > > > >, 
	                    Reference<QDPType< DCol, OLattice< DCol > > > >,
	                    OLattice< DCol > >& rhs,
	      const Subset& s)
{
//  cout << "call single site QDP_M_eq_aM_times_M" << endl;

  typedef OLattice< DCol >    C;

  const C& l = static_cast<const C&>(rhs.expression().left().child());
  const C& r = static_cast<const C&>(rhs.expression().right());

  if( s.hasOrderedRep() ) { 
    int n_3mat = s.end()-s.start()+1;
    REAL64 *lm = (REAL64 *)&(l.elem(s.start()).elem().elem(0,0).real());
    REAL64 *rm = (REAL64 *)&(r.elem(s.start()).elem().elem(0,0).real());
    REAL64 *dm = (REAL64 *)&(d.elem(s.start()).elem().elem(0,0).real());
    
    ssed_m_eq_hm(dm,lm,rm,n_3mat);
  }
  else { 
    const int *tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); ++j) {

      int i = tab[j];
      REAL64 *lm = (REAL64 *)&(l.elem(i).elem().elem(0,0).real());
      REAL64 *rm = (REAL64 *)&(r.elem(i).elem().elem(0,0).real());
      REAL64 *dm = (REAL64 *)&(d.elem(i).elem().elem(0,0).real());

      ssed_m_eq_hm(dm, lm, rm, 1);

    }
  }

}


// Specialization to optimize the case   
//    LatticeColorMatrix[Subset] = LatticeColorMatrix * adj(LatticeColorMatrix)
template<>
void evaluate(OLattice< DCol >& d, 
	      const OpAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiplyAdj, 
	                    Reference<QDPType< DCol, OLattice< DCol > > >, 
	                    UnaryNode<OpIdentity, Reference<QDPType< DCol, OLattice< DCol > > > > >,
	                    OLattice< DCol > >& rhs,
	      const Subset& s)
{
//  cout << "call single site QDP_M_eq_M_times_aM" << endl;

  typedef OLattice< DCol >    C;

  const C& l = static_cast<const C&>(rhs.expression().left());
  const C& r = static_cast<const C&>(rhs.expression().right().child());

  if( s.hasOrderedRep() ) { 
      REAL64 *lm = (REAL64 *)&(l.elem(s.start()).elem().elem(0,0).real());
      REAL64 *rm = (REAL64 *)&(r.elem(s.start()).elem().elem(0,0).real());
      REAL64 *dm = (REAL64 *)&(d.elem(s.start()).elem().elem(0,0).real());
      int n_3mat = s.end()-s.start()+1;
      ssed_m_eq_mh(dm,lm,rm,n_3mat);
  }
  else { 

    const int *tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); ++j) {
      int i = tab[j];
      REAL64 *lm = (REAL64 *)&(l.elem(i).elem().elem(0,0).real());
      REAL64 *rm = (REAL64 *)&(r.elem(i).elem().elem(0,0).real());
      REAL64 *dm = (REAL64 *)&(d.elem(i).elem().elem(0,0).real());
      ssed_m_eq_mh(dm, lm, rm, 1);

    }
  }
}


// Specialization to optimize the case   
//    LatticeColorMatrix[Subset] = adj(LatticeColorMatrix) * adj(LatticeColorMatrix)
template<>
void evaluate(OLattice< DCol >& d, 
	      const OpAssign& op, 
	      const QDPExpr<BinaryNode<OpAdjMultiplyAdj, 
	                    UnaryNode<OpIdentity, Reference<QDPType< DCol, OLattice< DCol > > > >,
	                    UnaryNode<OpIdentity, Reference<QDPType< DCol, OLattice< DCol > > > > >,
	                    OLattice< DCol > >& rhs,
	      const Subset& s)
{
//  cout << "call single site QDP_M_eq_Ma_times_Ma" << endl;

  typedef OLattice< DCol >    C;

  const C& l = static_cast<const C&>(rhs.expression().left().child());
  const C& r = static_cast<const C&>(rhs.expression().right().child());

  if( s.hasOrderedRep() ) { 
    int n_3mat = s.end() - s.start() + 1;
    REAL64 *lm = (REAL64 *)&(l.elem(s.start()).elem().elem(0,0).real());
    REAL64 *rm = (REAL64 *)&(r.elem(s.start()).elem().elem(0,0).real());
    REAL64 *dm = (REAL64 *)&(d.elem(s.start()).elem().elem(0,0).real());
    
    ssed_m_eq_hh(dm,lm,rm,n_3mat);
  }
  else { 
    const int *tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); ++j) {
      int i = tab[j];
      REAL64 *lm = (REAL64 *)&(l.elem(i).elem().elem(0,0).real());
      REAL64 *rm = (REAL64 *)&(r.elem(i).elem().elem(0,0).real());
      REAL64 *dm = (REAL64 *)&(d.elem(i).elem().elem(0,0).real());

      ssed_m_eq_hh(dm,lm,rm,1);
    }
  }
}

//-------------------------------------------------------------------

// Specialization to optimize the case   
//    LatticeColorMatrix[Subset] += LatticeColorMatrix * LatticeColorMatrix
template<>
void evaluate(OLattice< DCol >& d, 
	      const OpAddAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	                    Reference<QDPType< DCol, OLattice< DCol > > >, 
	                    Reference<QDPType< DCol, OLattice< DCol > > > >,
	                    OLattice< DCol > >& rhs,
	      const Subset& s)
{
//  cout << "call single site QDP_M_peq_M_times_M" << endl;

  typedef OLattice< DCol >    C;

  const C& l = static_cast<const C&>(rhs.expression().left());
  const C& r = static_cast<const C&>(rhs.expression().right());

  REAL64 one=(REAL64)1;
  
  if( s.hasOrderedRep() ) { 
    REAL64 *lm = (REAL64 *)&(l.elem(s.start()).elem().elem(0,0).real());
    REAL64 *rm = (REAL64 *)&(r.elem(s.start()).elem().elem(0,0).real());
    REAL64 *dm = (REAL64 *)&(d.elem(s.start()).elem().elem(0,0).real());
    int n_3mat=s.end()-s.start()+1;

    ssed_m_peq_amm(dm,&one,lm,rm,n_3mat);
  }
  else { 
  
    const int *tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); ++j) {
      int i = tab[j];
      REAL64 *lm = (REAL64 *)&(l.elem(i).elem().elem(0,0).real());
      REAL64 *rm = (REAL64 *)&(r.elem(i).elem().elem(0,0).real());
      REAL64 *dm = (REAL64 *)&(d.elem(i).elem().elem(0,0).real());
      ssed_m_peq_amm(dm,&one,lm,rm,1);
    }
  }
}


// Specialization to optimize the case   
//    LatticeColorMatrix[Subset] += adj(LatticeColorMatrix) * LatticeColorMatrix
template<>
void evaluate(OLattice< DCol >& d, 
	      const OpAddAssign& op, 
	      const QDPExpr<BinaryNode<OpAdjMultiply, 
	                    UnaryNode<OpIdentity, Reference<QDPType< DCol, OLattice< DCol > > > >, 
	                    Reference<QDPType< DCol, OLattice< DCol > > > >,
	                    OLattice< DCol > >& rhs,
	      const Subset& s)
{
//  cout << "call single site QDP_M_peq_aM_times_M" << endl;

  typedef OLattice< DCol >    C;

  const C& l = static_cast<const C&>(rhs.expression().left().child());
  const C& r = static_cast<const C&>(rhs.expression().right());


  REAL64 one = (REAL64)1;
  if( s.hasOrderedRep() ) { 
      REAL64 *lm = (REAL64 *)&(l.elem(s.start()).elem().elem(0,0).real());
      REAL64 *rm = (REAL64 *)&(r.elem(s.start()).elem().elem(0,0).real());
      REAL64 *dm = (REAL64 *)&(d.elem(s.start()).elem().elem(0,0).real());

      int n_3mat=s.end()-s.start()+1;
      ssed_m_peq_ahm(dm,&one,lm,rm,n_3mat);
  }
  else { 
    const int *tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); ++j) {
      int i = tab[j];
      REAL64 *lm = (REAL64 *)&(l.elem(i).elem().elem(0,0).real());
      REAL64 *rm = (REAL64 *)&(r.elem(i).elem().elem(0,0).real());
      REAL64 *dm = (REAL64 *)&(d.elem(i).elem().elem(0,0).real());
      ssed_m_peq_ahm(dm,&one,lm,rm,1);
    }
  }
}


// Specialization to optimize the case   
//    LatticeColorMatrix[Subset] += LatticeColorMatrix * adj(LatticeColorMatrix)
template<>
void evaluate(OLattice< DCol >& d, 
	      const OpAddAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiplyAdj, 
	                    Reference<QDPType< DCol, OLattice< DCol > > >, 
	                    UnaryNode<OpIdentity, Reference<QDPType< DCol, OLattice< DCol > > > > >,
	                    OLattice< DCol > >& rhs,
	      const Subset& s)
{
//  cout << "call single site QDP_M_peq_M_times_aM" << endl;

  typedef OLattice< DCol >    C;

  const C& l = static_cast<const C&>(rhs.expression().left());
  const C& r = static_cast<const C&>(rhs.expression().right().child());

  REAL64 one=(REAL64)1;

  if( s.hasOrderedRep() ) { 
    REAL64 *lm = (REAL64 *)&(l.elem(s.start()).elem().elem(0,0).real());
    REAL64 *rm = (REAL64 *)&(r.elem(s.start()).elem().elem(0,0).real());
    REAL64 *dm = (REAL64 *)&(d.elem(s.start()).elem().elem(0,0).real());
    int n_3mat=s.end()-s.start()+1;
    ssed_m_peq_amh(dm,&one,lm,rm,n_3mat);
  }
  else { 

    const int *tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); ++j) {
      int i = tab[j];
      REAL64 *lm = (REAL64 *)&(l.elem(i).elem().elem(0,0).real());
      REAL64 *rm = (REAL64 *)&(r.elem(i).elem().elem(0,0).real());
      REAL64 *dm = (REAL64 *)&(d.elem(i).elem().elem(0,0).real());
      ssed_m_peq_amh(dm,&one,lm,rm,1);
    }
  }
}


// Specialization to optimize the case   
//    LatticeColorMatrix[Subset] += adj(LatticeColorMatrix) * adj(LatticeColorMatrix)
template<>
void evaluate(OLattice< DCol >& d, 
	      const OpAddAssign& op, 
	      const QDPExpr<BinaryNode<OpAdjMultiplyAdj, 
	                    UnaryNode<OpIdentity, Reference<QDPType< DCol, OLattice< DCol > > > >,
	                    UnaryNode<OpIdentity, Reference<QDPType< DCol, OLattice< DCol > > > > >,
	                    OLattice< DCol > >& rhs,
	      const Subset& s)
{
//  cout << "call single site QDP_M_peq_Ma_times_Ma" << endl;

  typedef OLattice< DCol >    C;

  const C& l = static_cast<const C&>(rhs.expression().left().child());
  const C& r = static_cast<const C&>(rhs.expression().right().child());
  
  REAL64 one=(REAL64)1;
  

  if( s.hasOrderedRep() ) { 
      
    REAL64 *lm = (REAL64 *)&(l.elem(s.start()).elem().elem(0,0).real());
    REAL64 *rm = (REAL64 *)&(r.elem(s.start()).elem().elem(0,0).real());
    REAL64 *dm = (REAL64 *)&(d.elem(s.start()).elem().elem(0,0).real());
    int n_3mat=s.end()-s.start()+1;
    ssed_m_peq_ahh(dm,&one,lm,rm,n_3mat);
      
  }
  else { 

    const int *tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); ++j) {
      int i = tab[j];
      REAL64 *lm = (REAL64 *)&(l.elem(i).elem().elem(0,0).real());
      REAL64 *rm = (REAL64 *)&(r.elem(i).elem().elem(0,0).real());
      REAL64 *dm = (REAL64 *)&(d.elem(i).elem().elem(0,0).real());
      ssed_m_peq_ahh(dm,&one,lm,rm,1);
    }
  }
}

//-------------------------------------------------------------------
// Specialization to optimize the case
//   LatticeColorMatrix = LatticeColorMatrix
//   Implement as m1 = a*m2  (a=1)
template<>
void evaluate(OLattice< DCol >& d, 
	      const OpAssign& op, 
	      const QDPExpr<
	         UnaryNode<OpIdentity, Reference< QDPType< DCol, OLattice< DCol > > > >,
                 OLattice< DCol > >& rhs, 
	      const Subset& s) 
{
  typedef OLattice<DCol> C;
  const C& l = static_cast<const C&>(rhs.expression().child());
  REAL64 one=(REAL64)1;

  if( s.hasOrderedRep() ) {
    int n_3mat=s.end()-s.start()+1;
    REAL64* d_ptr =&(d.elem(s.start()).elem().elem(0,0).real());
    REAL64* l_ptr =(REAL64*)&(l.elem(s.start()).elem().elem(0,0).real());
  
    ssed_m_eq_scal_m(d_ptr,&one,l_ptr,n_3mat);
  }
  else {
    // Unordered case 
    const int* tab = s.siteTable().slice();
    
    // Loop through the sites
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];
      REAL64* d_ptr =&(d.elem(i).elem().elem(0,0).real());
      REAL64* l_ptr =(REAL64*)&(l.elem(i).elem().elem(0,0).real());
  
      ssed_m_eq_scal_m(d_ptr,&one,l_ptr,1);
    }
  }
}

//-------------------------------------------------------------------
// Specialization to optimize the case
//   LatticeColorMatrix += LatticeColorMatrix
template<>
void evaluate(OLattice< DCol >& d, 
	      const OpAddAssign& op, 
	      const QDPExpr<
	         UnaryNode<OpIdentity, Reference< QDPType< DCol, OLattice< DCol > > > >,
                 OLattice< DCol > >& rhs, 
	      const Subset& s) 
{
  typedef OLattice<DCol> C;
  const C& l = static_cast<const C&>(rhs.expression().child());

  if( s.hasOrderedRep() ) { 
    int n_3mat=s.end()-s.start()+1;
    REAL64* d_ptr =&(d.elem(s.start()).elem().elem(0,0).real());
    REAL64* l_ptr =(REAL64*)&(l.elem(s.start()).elem().elem(0,0).real());
    ssed_m_peq_m(d_ptr,l_ptr,n_3mat);
  }
  else {
    // Unordered case 
    const int* tab = s.siteTable().slice();
    
    // Loop through the sites
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];
      REAL64* d_ptr =&(d.elem(i).elem().elem(0,0).real());
      REAL64* l_ptr =(REAL64*)&(l.elem(i).elem().elem(0,0).real());
      ssed_m_peq_m(d_ptr,l_ptr,1);

    }

  }
}

//-------------------------------------------------------------------
// Specialization to optimize the case
//   LatticeColorMatrix -= LatticeColorMatrix
template<>
void evaluate(OLattice< DCol >& d, 
	      const OpSubtractAssign& op, 
	      const QDPExpr<
	         UnaryNode<OpIdentity, Reference< QDPType< DCol, OLattice< DCol > > > >,
                 OLattice< DCol > >& rhs, 
	      const Subset& s) 
{
  typedef OLattice<DCol> C;
  const C& l = static_cast<const C&>(rhs.expression().child());
  if (s.hasOrderedRep()) { 
    int n_3mat=s.end()-s.start()+1;
    REAL64* d_ptr =&(d.elem(s.start()).elem().elem(0,0).real());
    REAL64* l_ptr =(REAL64*)&(l.elem(s.start()).elem().elem(0,0).real());
    ssed_m_meq_m(d_ptr,l_ptr,n_3mat);
  }
  else {   
    // Unordered case 
    const int* tab = s.siteTable().slice();
    
    // Loop through the sites
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i = tab[j];
      REAL64* d_ptr =&(d.elem(i).elem().elem(0,0).real());
      REAL64* l_ptr =(REAL64*)&(l.elem(i).elem().elem(0,0).real());
      ssed_m_meq_m(d_ptr,l_ptr,1);
    }
  }
}

// Specialization to optimize the case   
//    LatticeColorMatrix[Subset] -= LatticeColorMatrix * LatticeColorMatrix
template<>
void evaluate(OLattice< DCol >& d, 
	      const OpSubtractAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	                    Reference<QDPType< DCol, OLattice< DCol > > >, 
	                    Reference<QDPType< DCol, OLattice< DCol > > > >,
	                    OLattice< DCol > >& rhs,
	      const Subset& s)
{
//  cout << "call single site QDP_M_meq_M_times_M" << endl;

  typedef OLattice< DCol >    C;

  const C& l = static_cast<const C&>(rhs.expression().left());
  const C& r = static_cast<const C&>(rhs.expression().right());

  REAL64 mone=(REAL64)-1;


  if( s.hasOrderedRep() ) { 

    REAL64 *lm = (REAL64 *)&(l.elem(s.start()).elem().elem(0,0).real());
    REAL64 *rm = (REAL64 *)&(r.elem(s.start()).elem().elem(0,0).real());
    REAL64 *dm = (REAL64 *)&(d.elem(s.start()).elem().elem(0,0).real());
    int n_3mat=s.end()-s.start()+1;

    ssed_m_peq_amm(dm,&mone,lm, rm, n_3mat);
  }
  else { 

    const int *tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i=tab[j];
      REAL64 *lm = (REAL64 *)&(l.elem(i).elem().elem(0,0).real());
      REAL64 *rm = (REAL64 *)&(r.elem(i).elem().elem(0,0).real());
      REAL64 *dm = (REAL64 *)&(d.elem(i).elem().elem(0,0).real());

      ssed_m_peq_amm(dm,&mone,lm,rm,1);
    }
  }
}


// Specialization to optimize the case   
//    LatticeColorMatrix[Subset] -= adj(LatticeColorMatrix) * LatticeColorMatrix
template<>
void evaluate(OLattice< DCol >& d, 
	      const OpSubtractAssign& op, 
	      const QDPExpr<BinaryNode<OpAdjMultiply, 
	                    UnaryNode<OpIdentity, Reference<QDPType< DCol, OLattice< DCol > > > >, 
	                    Reference<QDPType< DCol, OLattice< DCol > > > >,
	                    OLattice< DCol > >& rhs,
	      const Subset& s)
{
//  cout << "call single site QDP_M_meq_aM_times_M" << endl;

  typedef OLattice< DCol >    C;

  const C& l = static_cast<const C&>(rhs.expression().left().child());
  const C& r = static_cast<const C&>(rhs.expression().right());

  REAL64 mone=(REAL64)-1;

  if( s.hasOrderedRep() ) { 
    REAL64 *lm = (REAL64 *)&(l.elem(s.start()).elem().elem(0,0).real());
    REAL64 *rm = (REAL64 *)&(r.elem(s.start()).elem().elem(0,0).real());
    REAL64 *dm = (REAL64 *)&(d.elem(s.start()).elem().elem(0,0).real());
    int n_3mat=s.end()-s.start()+1;
    ssed_m_peq_ahm(dm,&mone,lm,rm,n_3mat);
  }
  else { 

    const int *tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i=tab[j];
      REAL64 *lm = (REAL64 *)&(l.elem(i).elem().elem(0,0).real());
      REAL64 *rm = (REAL64 *)&(r.elem(i).elem().elem(0,0).real());
      REAL64 *dm = (REAL64 *)&(d.elem(i).elem().elem(0,0).real());
      ssed_m_peq_ahm(dm,&mone,lm, rm, 1);
    }
  }
}


// Specialization to optimize the case   
//    LatticeColorMatrix[Subset] -= LatticeColorMatrix * adj(LatticeColorMatrix)
template<>
void evaluate(OLattice< DCol >& d, 
	      const OpSubtractAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiplyAdj, 
	                    Reference<QDPType< DCol, OLattice< DCol > > >, 
	                    UnaryNode<OpIdentity, Reference<QDPType< DCol, OLattice< DCol > > > > >,
	                    OLattice< DCol > >& rhs,
	      const Subset& s)
{
//  cout << "call single site QDP_M_meq_M_times_aM" << endl;

  typedef OLattice< DCol >    C;

  const C& l = static_cast<const C&>(rhs.expression().left());
  const C& r = static_cast<const C&>(rhs.expression().right().child());

  REAL64 mone=(REAL64)-1;

  if( s.hasOrderedRep() ) { 
    REAL64 *lm = (REAL64 *)&(l.elem(s.start()).elem().elem(0,0).real());
    REAL64 *rm = (REAL64 *)&(r.elem(s.start()).elem().elem(0,0).real());
    REAL64 *dm = (REAL64 *)&(d.elem(s.start()).elem().elem(0,0).real());
    int n_3mat=s.end()-s.start()+1;
    ssed_m_peq_amh(dm,&mone,lm, rm,n_3mat);
  }
  else { 
    const int *tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i=tab[j];
      REAL64 *lm = (REAL64 *)&(l.elem(i).elem().elem(0,0).real());
      REAL64 *rm = (REAL64 *)&(r.elem(i).elem().elem(0,0).real());
      REAL64 *dm = (REAL64 *)&(d.elem(i).elem().elem(0,0).real());

      ssed_m_peq_amh(dm,&mone,lm, rm,1);
    }
  }
}


// Specialization to optimize the case   
//    LatticeColorMatrix[Subset] -= adj(LatticeColorMatrix) * adj(LatticeColorMatrix)
template<>
void evaluate(OLattice< DCol >& d, 
	      const OpSubtractAssign& op, 
	      const QDPExpr<BinaryNode<OpAdjMultiplyAdj, 
	                    UnaryNode<OpIdentity, Reference<QDPType< DCol, OLattice< DCol > > > >,
	                    UnaryNode<OpIdentity, Reference<QDPType< DCol, OLattice< DCol > > > > >,
	                    OLattice< DCol > >& rhs,
	      const Subset& s)
{
//  cout << "call single site QDP_M_meq_Ma_times_Ma" << endl;

  typedef OLattice< DCol >    C;

  const C& l = static_cast<const C&>(rhs.expression().left().child());
  const C& r = static_cast<const C&>(rhs.expression().right().child());
  REAL64 mone=(REAL64)-1;

  if( s.hasOrderedRep() ) { 

    REAL64 *lm = (REAL64 *)&(l.elem(s.start()).elem().elem(0,0).real());
    REAL64 *rm = (REAL64 *)&(r.elem(s.start()).elem().elem(0,0).real());
    REAL64 *dm = (REAL64 *)&(d.elem(s.start()).elem().elem(0,0).real());
    int n_3mat=s.end()-s.start()+1;
    ssed_m_peq_ahh(dm,&mone,lm,rm,n_3mat);
  }
  else { 
    const int *tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i=tab[j];
      REAL64 *lm = (REAL64 *)&(l.elem(i).elem().elem(0,0).real());
      REAL64 *rm = (REAL64 *)&(r.elem(i).elem().elem(0,0).real());
      REAL64 *dm = (REAL64 *)&(d.elem(i).elem().elem(0,0).real());
      ssed_m_peq_ahh(dm,&mone,lm,rm,1);
    }
  }
}


#if 0
// DPREC MAT VEC is not yet implemented
//-------------------------------------------------------------------

// Specialization to optimize the case   
//    LatticeHalfFermion = LatticeColorMatrix * LatticeHalfFermion
// NOTE: let this be a subroutine to save space
template<>
void evaluate(OLattice< TVec2 >& d, 
	      const OpAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	                    Reference<QDPType< DCol, OLattice< DCol > > >, 
	                    Reference<QDPType< TVec2, OLattice< TVec2 > > > >,
	                    OLattice< TVec2 > >& rhs,
	      const Subset& s)
{
#if defined(QDP_SCALARSITE_DEBUG)
  cout << "specialized QDP_H_M_times_H" << endl;
#endif

  typedef OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >       C;
  typedef OLattice<PSpinVector<PColorVector<RComplexFloat, 3>, 2> > H;

  const C& l = static_cast<const C&>(rhs.expression().left());
  const H& r = static_cast<const H&>(rhs.expression().right());



  if( s.hasOrderedRep() ) { 
    for(int i=s.start(); i <= s.end(); i++) { 

      REAL64* lm = (REAL64 *)&(l.elem(i).elem().elem(0,0).real());
      half_wilson_vectorf* rh = (half_wilson_vectorf*)&(r.elem(i).elem(0).elem(0).real());
      half_wilson_vectorf* dh = (half_wilson_vectorf*)&(d.elem(i).elem(0).elem(0).real());

      intrin_sse_mult_su3_mat_hwvec(lm,rh,dh);

    }
  }
  else { 

    const int *tab = s.siteTable().slice();
    for(int j=0; j < s.numSiteTable(); j++) { 
      int i=tab[j];
      REAL64* lm = (REAL64 *)&(l.elem(i).elem().elem(0,0).real());
      half_wilson_vectorf* rh = (half_wilson_vectorf*)&(r.elem(i).elem(0).elem(0).real());
      half_wilson_vectorf* dh = (half_wilson_vectorf*)&(d.elem(i).elem(0).elem(0).real());

      intrin_sse_mult_su3_mat_hwvec(lm,rh,dh);
    }
  }
}
#endif



} // namespace QDP;
