// -*- C++ -*-
// $Id: qdp_scalarsite_bagel_qdp_linalg.h,v 1.3 2007-06-13 20:47:58 bjoo Exp $

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


  // += 
template<>
inline
void evaluate(OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >& d, 
	      const OpAddAssign& op, 
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

  Float plus_one[2] QDP_ALIGN16;
  plus_one[0] = (Float)1;
  plus_one[1] = (Float)0;
  
  if( s.hasOrderedRep() ) { 
     // Do whole subset
     unsigned int start = s.start();
     unsigned int end   = s.end();

     unsigned long num_sites = end - start + 1;

     Float *resptr = &(d.elem(start).elem().elem(0,0).real());
     Float *lptr   = const_cast<Float*>(&(l.elem(start).elem().elem(0,0).real()));
     Float *rptr   = const_cast<Float*>(&(r.elem(start).elem().elem(0,0).real()));

     qdp_su3_mm_peq(resptr, plus_one, lptr, rptr, num_sites, (unsigned long)0);

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

       qdp_su3_mm_peq(resptr, plus_one, lptr, rptr, one_site, (unsigned long)0);

     }
   }

}

template<>
inline
void evaluate(OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >& d, 
	      const OpAddAssign& op, 
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

  Float plus_one[2] QDP_ALIGN16;
  plus_one[0] = (Float)1;
  plus_one[1] = (Float)0;
  
  if( s.hasOrderedRep() ) { 
     // Do whole subset
     unsigned int start = s.start();
     unsigned int end   = s.end();

     unsigned long num_sites = end - start + 1;

     Float *resptr = &(d.elem(start).elem().elem(0,0).real());
     Float *lptr   = const_cast<Float*>(&(l.elem(start).elem().elem(0,0).real()));
     Float *rptr   = const_cast<Float*>(&(r.elem(start).elem().elem(0,0).real()));


     qdp_su3_am_peq(resptr, plus_one, lptr, rptr, num_sites, (unsigned long)0);

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

       qdp_su3_am_peq(resptr, plus_one, lptr, rptr, one_site, (unsigned long)0);

     }
   }

}

template<>
inline
void evaluate(OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >& d, 
	      const OpAddAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiplyAdj, 
	      Reference< QDPType<
                  PScalar<PColorMatrix<RComplexFloat, 3> >, 
	          OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > 
	        > 
	      >,
 
	      UnaryNode< OpIdentity,
	      Reference< QDPType<
	          PScalar<PColorMatrix<RComplexFloat, 3> >, 
	          OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > 
	        > 
	      > > >,
	      OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > >& rhs,
	      const Subset& s) {

  typedef OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >       C;
  const C& l = static_cast<const C&>(rhs.expression().left());
  const C& r = static_cast<const C&>(rhs.expression().right().child());

#ifdef DEBUG_BAGELQDP_LINALG
  QDPIO::cout << "evaluate(M*A) subset = s " << endl;
#endif
  Float plus_one[2] QDP_ALIGN16;
  plus_one[0] = (Float)1;
  plus_one[1] = (Float)0;
  
  if( s.hasOrderedRep() ) { 
     // Do whole subset
     unsigned int start = s.start();
     unsigned int end   = s.end();

     unsigned long num_sites = end - start + 1;

     Float *resptr = &(d.elem(start).elem().elem(0,0).real());
     Float *lptr   = const_cast<Float*>(&(l.elem(start).elem().elem(0,0).real()));
     Float *rptr   = const_cast<Float*>(&(r.elem(start).elem().elem(0,0).real()));


     qdp_su3_ma_peq(resptr, plus_one, lptr, rptr, num_sites, (unsigned long)0);

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

       qdp_su3_ma_peq(resptr, plus_one, lptr, rptr, one_site, (unsigned long)0);

     }
   }

}


template<>
inline
void evaluate(OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >& d, 
	      const OpAddAssign& op, 
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

  Float plus_one[2] QDP_ALIGN16;
  plus_one[0] = (Float)1;
  plus_one[1] = (Float)0;

  if( s.hasOrderedRep() ) { 
     // Do whole subset
     unsigned int start = s.start();
     unsigned int end   = s.end();

     unsigned long num_sites = end - start + 1;

     Float *resptr = &(d.elem(start).elem().elem(0,0).real());
     Float *lptr   = const_cast<Float*>(&(l.elem(start).elem().elem(0,0).real()));
     Float *rptr   = const_cast<Float*>(&(r.elem(start).elem().elem(0,0).real()));


     qdp_su3_aa_peq(resptr,  plus_one, lptr, rptr, num_sites, (unsigned long)one_minus_i);

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

       qdp_su3_aa_peq(resptr, plus_one, lptr, rptr, one_site, (unsigned long)one_minus_i);

     }
   }

}



  //  -= 
  
template<>
inline
void evaluate(OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >& d, 
	      const OpSubtractAssign& op, 
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

  Float minus_one[2] QDP_ALIGN16;
  minus_one[0] = (Float)-1;
  minus_one[1] = (Float)0;
  
  if( s.hasOrderedRep() ) { 
     // Do whole subset
     unsigned int start = s.start();
     unsigned int end   = s.end();

     unsigned long num_sites = end - start + 1;

     Float *resptr = &(d.elem(start).elem().elem(0,0).real());
     Float *lptr   = const_cast<Float*>(&(l.elem(start).elem().elem(0,0).real()));
     Float *rptr   = const_cast<Float*>(&(r.elem(start).elem().elem(0,0).real()));

     qdp_su3_mm_peq(resptr, minus_one, lptr, rptr, num_sites, (unsigned long)0);

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

       qdp_su3_mm_peq(resptr, minus_one, lptr, rptr, one_site, (unsigned long)0);

     }
   }

}

template<>
inline
void evaluate(OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >& d, 
	      const OpSubtractAssign& op, 
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

  Float minus_one[2] QDP_ALIGN16;
  minus_one[0] = (Float)-1;
  minus_one[1] = (Float)0;
  
  if( s.hasOrderedRep() ) { 
     // Do whole subset
     unsigned int start = s.start();
     unsigned int end   = s.end();

     unsigned long num_sites = end - start + 1;

     Float *resptr = &(d.elem(start).elem().elem(0,0).real());
     Float *lptr   = const_cast<Float*>(&(l.elem(start).elem().elem(0,0).real()));
     Float *rptr   = const_cast<Float*>(&(r.elem(start).elem().elem(0,0).real()));


     qdp_su3_am_peq(resptr, minus_one, lptr, rptr, num_sites, (unsigned long)0);

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

       qdp_su3_am_peq(resptr, minus_one, lptr, rptr, one_site, (unsigned long)0);

     }
   }

}

template<>
inline
void evaluate(OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >& d, 
	      const OpSubtractAssign& op, 
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
  Float minus_one[2] QDP_ALIGN16;
  minus_one[0] = (Float)-1;
  minus_one[1] = (Float)0;
  
  if( s.hasOrderedRep() ) { 
     // Do whole subset
     unsigned int start = s.start();
     unsigned int end   = s.end();

     unsigned long num_sites = end - start + 1;

     Float *resptr = &(d.elem(start).elem().elem(0,0).real());
     Float *lptr   = const_cast<Float*>(&(l.elem(start).elem().elem(0,0).real()));
     Float *rptr   = const_cast<Float*>(&(r.elem(start).elem().elem(0,0).real()));


     qdp_su3_ma_peq(resptr, minus_one, lptr, rptr, num_sites, (unsigned long)0);

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

       qdp_su3_ma_peq(resptr, minus_one, lptr, rptr, one_site, (unsigned long)0);

     }
   }

}


template<>
inline
void evaluate(OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >& d, 
	      const OpSubtractAssign& op, 
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

  Float minus_one[2] QDP_ALIGN16;
  minus_one[0] = (Float)-1;
  minus_one[1] = (Float)0;

  if( s.hasOrderedRep() ) { 
     // Do whole subset
     unsigned int start = s.start();
     unsigned int end   = s.end();

     unsigned long num_sites = end - start + 1;

     Float *resptr = &(d.elem(start).elem().elem(0,0).real());
     Float *lptr   = const_cast<Float*>(&(l.elem(start).elem().elem(0,0).real()));
     Float *rptr   = const_cast<Float*>(&(r.elem(start).elem().elem(0,0).real()));


     qdp_su3_aa_peq(resptr, minus_one, lptr, rptr, num_sites, (unsigned long)one_minus_i);

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

       qdp_su3_aa_peq(resptr, minus_one, lptr, rptr, one_site, (unsigned long)one_minus_i);

     }
   }

}




  // += *
template<>
inline
void evaluate(OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >& d, 
	      const OpAddAssign& op, 

       const QDPExpr<
                    BinaryNode< OpMultiply, 
	              BinaryNode<OpMultiply, 
                        Reference<
                          QDPType<
                            PScalar<PScalar<RScalar<Float> > >,
	                    OScalar<PScalar<PScalar<RScalar<Float> > > >
                          > 
                        >,
                        Reference<
                          QDPType<
                            PScalar<PColorMatrix<RComplexFloat, 3> >, 
	                    OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > 
                          > 
                        >
	             >, 
	             Reference<
	               QDPType<
                         PScalar<PColorMatrix<RComplexFloat, 3> >, 
	                 OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > 
                       > 
	             >
                    >,
	            OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > 
              >& rhs,

	      const Subset& s) {

  typedef OScalar<PScalar<PScalar<RScalar<Float> > > > F;

  typedef OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >       C;

  typedef BinaryNode<OpMultiply, 
                        Reference<
                          QDPType<
                            PScalar<PScalar<RScalar<Float> > >,
	                    OScalar<PScalar<PScalar<RScalar<Float> > > >
                          > 
                        >,
                        Reference<
                          QDPType<
                            PScalar<PColorMatrix<RComplexFloat, 3> >, 
	                    OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > 
                          > 
                        >
    > BN;


  const BN& node = static_cast<const BN&>(rhs.expression().left());
  const F& scal = static_cast<const F&>(node.left());
  const C& l = static_cast<const C&>(node.right());
  const C& r = static_cast<const C&>(rhs.expression().right());

#if DEBUG_BAGELQDP_LINALG
  QDPIO::cout << "evaluate(M += alpha*M*M 2 ) subset = s " << endl;
#endif

  Float scalar[2] QDP_ALIGN16;
  scalar[0] = (Float)(scal.elem().elem().elem().elem());
  scalar[1] = (Float)0;
  
  if( s.hasOrderedRep() ) { 
     // Do whole subset
     unsigned int start = s.start();
     unsigned int end   = s.end();

     unsigned long num_sites = end - start + 1;

     Float *resptr = &(d.elem(start).elem().elem(0,0).real());
     Float *lptr   = const_cast<Float*>(&(l.elem(start).elem().elem(0,0).real()));
     Float *rptr   = const_cast<Float*>(&(r.elem(start).elem().elem(0,0).real()));

     qdp_su3_mm_peq(resptr, scalar, lptr, rptr, num_sites, (unsigned long)0);

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

       qdp_su3_mm_peq(resptr, scalar, lptr, rptr, one_site, (unsigned long)0);

     }
   }

}

  // +-= *
template<>
inline
void evaluate(OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >& d, 
	      const OpSubtractAssign& op, 

       const QDPExpr<
                    BinaryNode< OpMultiply, 
	              BinaryNode<OpMultiply, 
                        Reference<
                          QDPType<
                            PScalar<PScalar<RScalar<Float> > >,
	                    OScalar<PScalar<PScalar<RScalar<Float> > > >
                          > 
                        >,
                        Reference<
                          QDPType<
                            PScalar<PColorMatrix<RComplexFloat, 3> >, 
	                    OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > 
                          > 
                        >
	             >, 
	             Reference<
	               QDPType<
                         PScalar<PColorMatrix<RComplexFloat, 3> >, 
	                 OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > 
                       > 
	             >
                    >,
	            OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > 
              >& rhs,

	      const Subset& s) {

  typedef OScalar<PScalar<PScalar<RScalar<Float> > > > F;

  typedef OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >       C;

  typedef BinaryNode<OpMultiply, 
                        Reference<
                          QDPType<
                            PScalar<PScalar<RScalar<Float> > >,
	                    OScalar<PScalar<PScalar<RScalar<Float> > > >
                          > 
                        >,
                        Reference<
                          QDPType<
                            PScalar<PColorMatrix<RComplexFloat, 3> >, 
	                    OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > 
                          > 
                        >
    > BN;


  const BN& node = static_cast<const BN&>(rhs.expression().left());
  const F& scal = static_cast<const F&>(node.left());
  const C& l = static_cast<const C&>(node.right());
  const C& r = static_cast<const C&>(rhs.expression().right());

#if DEBUG_BAGELQDP_LINALG
  QDPIO::cout << "evaluate(M -= alpha*M*M 2 ) subset = s " << endl;
#endif

  Float scalar[2] QDP_ALIGN16;
  scalar[0] = -(Float)(scal.elem().elem().elem().elem());
  scalar[1] = (Float)0;
  
  if( s.hasOrderedRep() ) { 
     // Do whole subset
     unsigned int start = s.start();
     unsigned int end   = s.end();

     unsigned long num_sites = end - start + 1;

     Float *resptr = &(d.elem(start).elem().elem(0,0).real());
     Float *lptr   = const_cast<Float*>(&(l.elem(start).elem().elem(0,0).real()));
     Float *rptr   = const_cast<Float*>(&(r.elem(start).elem().elem(0,0).real()));

     qdp_su3_mm_peq(resptr, scalar, lptr, rptr, num_sites, (unsigned long)0);

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

       qdp_su3_mm_peq(resptr, scalar, lptr, rptr, one_site, (unsigned long)0);

     }
   }

}


  // +=a * M
template<>
inline
void evaluate(OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >& d, 
	      const OpAddAssign& op, 

	      const QDPExpr<
	              BinaryNode< OpMultiply, 
	                Reference<
                          QDPType<
                            PScalar<PScalar<RScalar<Float> > >,
	                    OScalar<PScalar<PScalar<RScalar<Float> > > >
                          > 
                        >,
                        Reference<
                          QDPType<
                            PScalar<PColorMatrix<RComplexFloat, 3> >, 
	                    OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > 
                          > 
                        >
	              >,
	            OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > 
              >& rhs,

	      const Subset& s) {

  typedef OScalar<PScalar<PScalar<RScalar<Float> > > > F;

  typedef OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >       C;



  const F& scal = static_cast<const F&>(rhs.expression().left());
  const C& r = static_cast<const C&>(rhs.expression().right());

#if DEBUG_BAGELQDP_LINALG
  QDPIO::cout << "evaluate(M += alpha*M ) subset = s " << endl;
#endif

  Float *scalar   = const_cast<Float*>(&(scal.elem().elem().elem().elem()));  
  if( s.hasOrderedRep() ) { 
     // Do whole subset
     unsigned int start = s.start();
     unsigned int end   = s.end();

     unsigned long num_sites = end - start + 1;
     
     Float *y = &(d.elem(start).elem().elem(0,0).real());

     Float *x   = const_cast<Float*>(&(r.elem(start).elem().elem(0,0).real()));

     qdp_vaxpy3(y, scalar, x, y, 3*num_sites);

   }
   else { 
     // Do site by site
     const int* tab = s.siteTable().slice();
     unsigned int num_sites = s.numSiteTable();
     for(int j=0; j < num_sites; j++) {
       int i = tab[j];
       Float *y = &(d.elem(i).elem().elem(0,0).real());
       Float *x   = const_cast<Float*>(&(r.elem(i).elem().elem(0,0).real()));

       qdp_vaxpy3(y, scalar, x,y, 3);

     }
   }

}

  // -=a * M
template<>
inline
void evaluate(OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >& d, 
	      const OpSubtractAssign& op, 

	      const QDPExpr<
	              BinaryNode< OpMultiply, 
	                Reference<
                          QDPType<
                            PScalar<PScalar<RScalar<Float> > >,
	                    OScalar<PScalar<PScalar<RScalar<Float> > > >
                          > 
                        >,
                        Reference<
                          QDPType<
                            PScalar<PColorMatrix<RComplexFloat, 3> >, 
	                    OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > 
                          > 
                        >
	              >,
	            OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > 
              >& rhs,

	      const Subset& s) {

  typedef OScalar<PScalar<PScalar<RScalar<Float> > > > F;

  typedef OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >       C;



  const F& scal = static_cast<const F&>(rhs.expression().left());
  const C& r = static_cast<const C&>(rhs.expression().right());

#if DEBUG_BAGELQDP_LINALG
  QDPIO::cout << "evaluate(M -= alpha*M ) subset = s " << endl;
#endif

  Float scalar = -scal.elem().elem().elem().elem();
  if( s.hasOrderedRep() ) { 
     // Do whole subset
     unsigned int start = s.start();
     unsigned int end   = s.end();

     unsigned long num_sites = end - start + 1;
     
     Float *y = &(d.elem(start).elem().elem(0,0).real());

     Float *x   = const_cast<Float*>(&(r.elem(start).elem().elem(0,0).real()));

     qdp_vaxpy3(y, &scalar, x, y, 3*num_sites);

   }
   else { 
     // Do site by site
     const int* tab = s.siteTable().slice();
     unsigned int num_sites = s.numSiteTable();
     for(int j=0; j < num_sites; j++) {
       int i = tab[j];
       Float *y = &(d.elem(i).elem().elem(0,0).real());
       Float *x   = const_cast<Float*>(&(r.elem(i).elem().elem(0,0).real()));

       qdp_vaxpy3(y, &scalar, x,y, 3);

     }
   }

}

  // += M
template<>
inline
void evaluate(OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >& d, 
	      const OpAddAssign& op, 

	      const QDPExpr< 
	              UnaryNode<OpIdentity,
	                Reference<
                          QDPType<
	      PScalar<PColorMatrix<RComplexFloat, 3> >,
                            OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >
                          > 
                        >
	              >,
	              OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > 
                    >& rhs,

	      const Subset& s) {

  typedef OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >       C;
  const C& r = static_cast<const C&>(rhs.expression().child());

#if DEBUG_BAGELQDP_LINALG
  QDPIO::cout << "evaluate(M += M ) subset = s " << endl;
#endif

  if( s.hasOrderedRep() ) { 
     // Do whole subset
     unsigned int start = s.start();
     unsigned int end   = s.end();

     unsigned long num_sites = end - start + 1;
     
     Float *y = &(d.elem(start).elem().elem(0,0).real());
     Float *x   = const_cast<Float*>(&(r.elem(start).elem().elem(0,0).real()));

     qdp_vadd3(y, x, y, 3*num_sites);

   }
   else { 
     // Do site by site
     const int* tab = s.siteTable().slice();
     unsigned int num_sites = s.numSiteTable();
     for(int j=0; j < num_sites; j++) {
       int i = tab[j];
       Float *y = &(d.elem(i).elem().elem(0,0).real());
       Float *x   = const_cast<Float*>(&(r.elem(i).elem().elem(0,0).real()));

       qdp_vadd3(y,x,y, 3);

     }
   }

}

  // -=a * M
template<>
inline
void evaluate(OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >& d, 
	      const OpSubtractAssign& op, 

	      const QDPExpr< 
	              UnaryNode< OpIdentity, 
	                Reference<
                          QDPType<
                            PScalar<PColorMatrix<RComplexFloat, 3> >, 
	                    OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > 
                          > 
                        >
                      >,
 	              OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > 
                   >& rhs,

	      const Subset& s) {

  typedef OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >       C;

  const C& r = static_cast<const C&>(rhs.expression().child());

#if DEBUG_BAGELQDP_LINALG
  QDPIO::cout << "evaluate(M -= M ) subset = s " << endl;
#endif

  if( s.hasOrderedRep() ) { 
     // Do whole subset
     unsigned int start = s.start();
     unsigned int end   = s.end();

     unsigned long num_sites = end - start + 1;
     
     Float *y = &(d.elem(start).elem().elem(0,0).real());

     Float *x   = const_cast<Float*>(&(r.elem(start).elem().elem(0,0).real()));

     qdp_vsub3(y, y,x, 3*num_sites);

   }
   else { 
     // Do site by site
     const int* tab = s.siteTable().slice();
     unsigned int num_sites = s.numSiteTable();
     for(int j=0; j < num_sites; j++) {
       int i = tab[j];
       Float *y = &(d.elem(i).elem().elem(0,0).real());
       Float *x   = const_cast<Float*>(&(r.elem(i).elem().elem(0,0).real()));

       qdp_vsub3(y,y,x, 3);

     }
   }

}


#endif


#if defined(DEBUG_BAGELQDP_LINALG)
#undef DEBUG_BAGELQDP_LINALG
#endif

} // namespace QDP;

#endif
