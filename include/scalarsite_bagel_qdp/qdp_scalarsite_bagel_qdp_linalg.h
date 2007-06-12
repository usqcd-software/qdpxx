// -*- C++ -*-
// $Id: qdp_scalarsite_bagel_qdp_linalg.h,v 1.1 2007-06-12 21:45:42 bjoo Exp $

/*! @file
 * @brief Qcdoc optimizations
 *
 * Qcdoc version of optimized basic operations
 */

#ifndef QDP_SCALARSITE_BAGEL_QDP_LINALG_H
#define QDP_SCALARSITE_BAGEL_QDP_LINALG_H

namespace QDP {

/*! @defgroup optimizations  Optimizations
 *
 * Optimizations for basic QDP operations
 *
 * @{
 */

// Use this def just to safe some typing later on in the file



#include "bagel_qdp.h"

#if 1
typedef RComplex<Float>  RComplexFloat;


template<>
inline
void evaluate(OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >& d, 
	      const OpAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	      Reference<QDPType<PScalar<PColorMatrix<RComplexFloat, 3> >, 
	      OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > > >, 
	      Reference<QDPType<PScalar<PColorMatrix<RComplexFloat, 3> >, 
	      OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > > > >,
	      OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > >& rhs,
	      const Subset& s) {

  typedef OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >       C;
  const C& l = static_cast<const C&>(rhs.expression().left());
  const C& r = static_cast<const C&>(rhs.expression().right());

#ifdef DEBUG_BAGELQDP_LINALG
  QDPIO::cout << "evaluate(M*M) subset = s " << endl;
#endif
  
   if( s.hasOrderedRep() ) { 
     // Do whole subset
     unsigned int start = s.start();
     unsigned int end   = s.end();

     unsigned long num_sites = end - start + 1;

     Float *resptr = &(d.elem(start).elem().elem(0,0).real());
     Float *lptr   = const_cast<Float*>(&(l.elem(start).elem().elem(0,0).real()));
     Float *rptr   = const_cast<Float*>(&(r.elem(start).elem().elem(0,0).real()));


     qdp_su3_mm(resptr, lptr, rptr, num_sites, (unsigned long)0);

   }
   else { 
     // Do site by site
     const int* tab = s.siteTable().slice();
     unsigned int num_sites = s.numSiteTable();
     for(int j=0; j < num_sites; j++) {
       int i = tab[j];
       unsigned long one_site = 1;
       Float *resptr = &(d.elem(i).elem().elem(0,0).real());

       Float *lptr   = const_cast<Float*>(&(l.elem(i).elem().elem(0,0).real()));
       Float *rptr   = const_cast<Float*>(&(r.elem(i).elem().elem(0,0).real()));

       qdp_su3_mm(resptr, lptr, rptr, one_site, (unsigned long)0);

     }
   }

}

template<>
inline
void evaluate(OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >& d, 
	      const OpAssign& op, 
	      const QDPExpr<BinaryNode<OpAdjMultiply, 
	                              UnaryNode<OpIdentity, Reference<QDPType<PScalar<PColorMatrix<RComplexFloat, 3> >, 
	      OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > > > >, 
	      Reference<QDPType<PScalar<PColorMatrix<RComplexFloat, 3> >, 
	      OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > > > >,
	      OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > >& rhs,
	      const Subset& s) {

  typedef OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >       C;
  const C& l = static_cast<const C&>(rhs.expression().left().child());
  const C& r = static_cast<const C&>(rhs.expression().right());

#ifdef DEBUG_BAGELQDP_LINALG
  QDPIO::cout << "evaluate(A*M) subset = s " << endl;
#endif
  
   if( s.hasOrderedRep() ) { 
     // Do whole subset
     unsigned int start = s.start();
     unsigned int end   = s.end();

     unsigned long num_sites = end - start + 1;

     Float *resptr = &(d.elem(start).elem().elem(0,0).real());
     Float *lptr   = const_cast<Float*>(&(l.elem(start).elem().elem(0,0).real()));
     Float *rptr   = const_cast<Float*>(&(r.elem(start).elem().elem(0,0).real()));


     qdp_su3_am(resptr, lptr, rptr, num_sites, (unsigned long)0);

   }
   else { 
     // Do site by site
     const int* tab = s.siteTable().slice();
     unsigned int num_sites = s.numSiteTable();
     for(int j=0; j < num_sites; j++) {
       int i = tab[j];
       unsigned long one_site = 1;
       Float *resptr = &(d.elem(i).elem().elem(0,0).real());

       Float *lptr   = const_cast<Float*>(&(l.elem(i).elem().elem(0,0).real()));
       Float *rptr   = const_cast<Float*>(&(r.elem(i).elem().elem(0,0).real()));

       qdp_su3_am(resptr, lptr, rptr, one_site, (unsigned long)0);

     }
   }

}

template<>
inline
void evaluate(OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >& d, 
	      const OpAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiplyAdj, 
	                               Reference<
	                                 QDPType<
                                               PScalar<PColorMatrix<RComplexFloat, 3> >, 
	                                       OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > 
                                         > 
                                       >,
 
	                                UnaryNode< OpIdentity,
                                        Reference<
                                           QDPType<
                                               PScalar<PColorMatrix<RComplexFloat, 3> >, 
	                                       OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > 
                                           > 
                                         > 
                                        > 
                                      
	                   >,

	                   OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > 
                     >& rhs,
	      const Subset& s) {

  typedef OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >       C;
  const C& l = static_cast<const C&>(rhs.expression().left());
  const C& r = static_cast<const C&>(rhs.expression().right().child());

#ifdef DEBUG_BAGELQDP_LINALG
  QDPIO::cout << "evaluate(M*A) subset = s " << endl;
#endif
  
   if( s.hasOrderedRep() ) { 
     // Do whole subset
     unsigned int start = s.start();
     unsigned int end   = s.end();

     unsigned long num_sites = end - start + 1;

     Float *resptr = &(d.elem(start).elem().elem(0,0).real());
     Float *lptr   = const_cast<Float*>(&(l.elem(start).elem().elem(0,0).real()));
     Float *rptr   = const_cast<Float*>(&(r.elem(start).elem().elem(0,0).real()));


     qdp_su3_ma(resptr, lptr, rptr, num_sites, (unsigned long)0);

   }
   else { 
     // Do site by site
     const int* tab = s.siteTable().slice();
     unsigned int num_sites = s.numSiteTable();
     for(int j=0; j < num_sites; j++) {
       int i = tab[j];
       unsigned long one_site = 1;
       Float *resptr = &(d.elem(i).elem().elem(0,0).real());

       Float *lptr   = const_cast<Float*>(&(l.elem(i).elem().elem(0,0).real()));
       Float *rptr   = const_cast<Float*>(&(r.elem(i).elem().elem(0,0).real()));

       qdp_su3_ma(resptr, lptr, rptr, one_site, (unsigned long)0);

     }
   }

}


template<>
inline
void evaluate(OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >& d, 
	      const OpAssign& op, 
	      const QDPExpr<BinaryNode<OpAdjMultiplyAdj, 
	      UnaryNode<OpIdentity, Reference<QDPType<PScalar<PColorMatrix<RComplexFloat, 3> >, 
	      OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > > > >, 
	      UnaryNode<OpIdentity, Reference<QDPType<PScalar<PColorMatrix<RComplexFloat, 3> >, 
	      OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > > > > >,
	      OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > >& rhs,
	      const Subset& s) {

  typedef OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >       C;
  const C& l = static_cast<const C&>(rhs.expression().left().child());
  const C& r = static_cast<const C&>(rhs.expression().right().child());

#ifdef DEBUG_BAGELQDP_LINALG
  QDPIO::cout << "evaluate(A*A) subset = s " << endl;
#endif
  Float one_minus_i[2] QDP_ALIGN16;
  one_minus_i[0] = (Float)1;
  one_minus_i[1] = (Float)(-1);

   if( s.hasOrderedRep() ) { 
     // Do whole subset
     unsigned int start = s.start();
     unsigned int end   = s.end();

     unsigned long num_sites = end - start + 1;

     Float *resptr = &(d.elem(start).elem().elem(0,0).real());
     Float *lptr   = const_cast<Float*>(&(l.elem(start).elem().elem(0,0).real()));
     Float *rptr   = const_cast<Float*>(&(r.elem(start).elem().elem(0,0).real()));


     qdp_su3_aa(resptr, lptr, rptr, num_sites, (unsigned long)one_minus_i);

   }
   else { 
     // Do site by site
     const int* tab = s.siteTable().slice();
     unsigned int num_sites = s.numSiteTable();
     for(int j=0; j < num_sites; j++) {
       int i = tab[j];
       unsigned long one_site = 1;
       Float *resptr = &(d.elem(i).elem().elem(0,0).real());

       Float *lptr   = const_cast<Float*>(&(l.elem(i).elem().elem(0,0).real()));
       Float *rptr   = const_cast<Float*>(&(r.elem(i).elem().elem(0,0).real()));

       qdp_su3_aa(resptr, lptr, rptr, one_site, (unsigned long)one_minus_i);

     }
   }

}

#endif


#if defined(DEBUG_BAGELQDP_LINALG)
#undef DEBUG_BAGELQDP_LINALG
#endif

} // namespace QDP;

#endif
