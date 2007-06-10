// -*- C++ -*-
// $Id: qdp_scalarsite_qcdoc_linalg.h,v 1.10 2007-06-10 14:32:10 edwards Exp $

/*! @file
 * @brief Qcdoc optimizations
 *
 * Qcdoc version of optimized basic operations
 */

#ifndef QDP_SCALARSITE_QCDOC_LINALG_H
#define QDP_SCALARSITE_QCDOC_LINALG_H

namespace QDP {

/*! @defgroup optimizations  Optimizations
 *
 * Optimizations for basic QDP operations
 *
 * @{
 */

// Use this def just to safe some typing later on in the file
typedef RComplex<float>  RComplexFloat;


#include "scalarsite_qcdoc/save_fp_regs.h"
#include "scalarsite_qcdoc/qcdoc_mult_nn.h"
#include "scalarsite_qcdoc/qcdoc_mult_an.h"
#include "scalarsite_generic/generic_mult_na.h"
#include "scalarsite_generic/generic_mult_aa.h"
#include "scalarsite_generic/generic_mat_vec.h"
#include "scalarsite_generic/generic_adj_mat_vec.h"
#include "scalarsite_generic/generic_addvec.h"

extern "C" {
 /* Peter's assembler  W = U * V  -- W, U, V are pointers to matrices */
 void qcdoc_mult_su3_nn_subset(int *length, REAL *Uptr, REAL *Vptr, REAL *Wptr);
};
// #define QDP_SCALARSITE_DEBUG

// Optimized version of  
//    PColorMatrix<RComplexFloat,3> <- PColorMatrix<RComplexFloat,3> * PColorMatrix<RComplexFloat,3>
template<>
inline BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
  PMatrix<RComplexFloat,3,PColorMatrix>, OpMultiply>::Type_t
operator*(const PMatrix<RComplexFloat,3,PColorMatrix>& l, 
	  const PMatrix<RComplexFloat,3,PColorMatrix>& r)
{
  BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
    PMatrix<RComplexFloat,3,PColorMatrix>, OpMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "M*M" << endl;
#endif
  double fp_regs[32];
  _save_fp_regs(fp_regs);
  _inline_qcdoc_mult_su3_nn(l,r,d);
  _restore_fp_regs(fp_regs);


  return d;
}


// Optimized version of  
//    PScalar<PColorMatrix<RComplexFloat,3>> <- PScalar<PColorMatrix<RComplexFloat,3>> * 
//                         PScalar<PColorMatrix<RComplexFloat,3>>
template<>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >,
  PScalar<PColorMatrix<RComplexFloat,3> >, OpMultiply>::Type_t
operator*(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	  const PScalar<PColorMatrix<RComplexFloat,3> >& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PScalar<PColorMatrix<RComplexFloat,3> >, OpMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "PSc<M>*PSc<M>" << endl;
#endif

  double fp_regs[32];
  _save_fp_regs(fp_regs);
  _inline_qcdoc_mult_su3_nn(l.elem(),r.elem(),d.elem());
  _restore_fp_regs(fp_regs);

  return d;
}


// Optimized version of  
//   PColorMatrix<RComplexFloat,3> <- adj(PColorMatrix<RComplexFloat,3>) * PColorMatrix<RComplexFloat,3>
template<>
inline BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
  PMatrix<RComplexFloat,3,PColorMatrix>, OpAdjMultiply>::Type_t
adjMultiply(const PMatrix<RComplexFloat,3,PColorMatrix>& l, 
	    const PMatrix<RComplexFloat,3,PColorMatrix>& r)
{
  BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
    PMatrix<RComplexFloat,3,PColorMatrix>, OpAdjMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "adj(M)*M" << endl;
#endif

  _inline_qcdoc_mult_su3_an(l,r,d);

  return d;
}


// Optimized version of  
//   PScalar<PColorMatrix<RComplexFloat,3>> <- adj(PScalar<PColorMatrix<RComplexFloat,3>>) * PScalar<PColorMatrix<RComplexFloat,3>>
template<>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PScalar<PColorMatrix<RComplexFloat,3> >, OpAdjMultiply>::Type_t
adjMultiply(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	    const PScalar<PColorMatrix<RComplexFloat,3> >& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PScalar<PColorMatrix<RComplexFloat,3> >, OpAdjMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "adj(PSc<M>)*PSc<M>" << endl;
#endif

  _inline_qcdoc_mult_su3_an(l.elem(),r.elem(),d.elem());

  return d;
}


// Optimized version of  
//   PColorMatrix<RComplexFloat,3> <- PColorMatrix<RComplexFloat,3> * adj(PColorMatrix<RComplexFloat,3>)
template<>
inline BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
  PMatrix<RComplexFloat,3,PColorMatrix>, OpMultiplyAdj>::Type_t
multiplyAdj(const PMatrix<RComplexFloat,3,PColorMatrix>& l, 
	    const PMatrix<RComplexFloat,3,PColorMatrix>& r)
{
  BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
    PMatrix<RComplexFloat,3,PColorMatrix>, OpMultiplyAdj>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "M*adj(M)" << endl;
#endif

  _inline_generic_mult_su3_na(l,r,d);

  return d;
}


// Optimized version of  
//   PScalar<PColorMatrix<RComplexFloat,3>> <- PScalar<PColorMatrix<RComplexFloat,3>> * adj(PScalar<PColorMatrix<RComplexFloat,3>>)
template<>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PScalar<PColorMatrix<RComplexFloat,3> >, OpMultiplyAdj>::Type_t
multiplyAdj(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	    const PScalar<PColorMatrix<RComplexFloat,3> >& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PScalar<PColorMatrix<RComplexFloat,3> >, OpMultiplyAdj>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "PSc<M>*adj(PSc<M>)" << endl;
#endif

  _inline_generic_mult_su3_na(l.elem(),r.elem(),d.elem());

  return d;
}


#if 0
// Do not use this routine - for some unknown reason on P4 it is slower than
// the default code!!

// Optimized version of  
//   PColorMatrix<RComplexFloat,3> <- adj(PColorMatrix<RComplexFloat,3>) * adj(PColorMatrix<RComplexFloat,3>)
template<>
inline BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
  PMatrix<RComplexFloat,3,PColorMatrix>, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(const PMatrix<RComplexFloat,3,PColorMatrix>& l, 
	       const PMatrix<RComplexFloat,3,PColorMatrix>& r)
{
  BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
    PMatrix<RComplexFloat,3,PColorMatrix>, OpAdjMultiplyAdj>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "adj(PSc<M>)*adj(PSc<M>)" << endl;
#endif

  _inline_generic_mult_su3_aa(l,r,d);

  return d;
}
#endif


// Optimized version of  
//    PColorVector<RComplexFloat,3> <- PColorMatrix<RComplexFloat,3> * PColorVector<RComplexFloat,3>
template<>
inline BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
  PVector<RComplexFloat,3,PColorVector>, OpMultiply>::Type_t
operator*(const PMatrix<RComplexFloat,3,PColorMatrix>& l, 
	  const PVector<RComplexFloat,3,PColorVector>& r)
{
  BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
    PVector<RComplexFloat,3,PColorVector>, OpMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "M*V" << endl;
#endif

  _inline_generic_mult_su3_mat_vec(l,r,d);

  return d;
}


// Optimized version of  
//    PScalar<PColorVector<RComplexFloat,3>> <- PScalar<PColorMatrix<RComplexFloat,3>> * PScalar<PColorVector<RComplexFloat,3>>
template<>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PScalar<PColorVector<RComplexFloat,3> >, OpMultiply>::Type_t
operator*(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	  const PScalar<PColorVector<RComplexFloat,3> >& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PScalar<PColorVector<RComplexFloat,3> >, OpMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "PSc<M>*PSc<V>" << endl;
#endif

  _inline_generic_mult_su3_mat_vec(l.elem(),r.elem(),d.elem());

  return d;
}


// Optimized version of  
//    PColorVector<RComplexFloat,3> <- adj(PColorMatrix<RComplexFloat,3>) * PColorVector<RComplexFloat,3>
template<>
inline BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
  PVector<RComplexFloat,3,PColorVector>, OpAdjMultiply>::Type_t
adjMultiply(const PMatrix<RComplexFloat,3,PColorMatrix>& l, 
	    const PVector<RComplexFloat,3,PColorVector>& r)
{
  BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
    PVector<RComplexFloat,3,PColorVector>, OpAdjMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "adj(M)*V" << endl;
#endif

  _inline_generic_mult_adj_su3_mat_vec(l,r,d);
  
  return d;
}


// Optimized version of   StaggeredFermion <- ColorMatrix*StaggeredFermion
//    PSpinVector<PColorVector<RComplexFloat,3>,1> <- PScalar<PColorMatrix<RComplexFloat,3>> * PSpinVector<PColorVector<RComplexFloat,3>,1>
template<>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PVector<PColorVector<RComplexFloat,3>,1,PSpinVector>, OpMultiply>::Type_t
operator*(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	  const PVector<PColorVector<RComplexFloat,3>,1,PSpinVector>& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PVector<PColorVector<RComplexFloat,3>,1,PSpinVector>, OpMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "PSc<M>*S" << endl;
#endif

  _inline_generic_mult_su3_mat_vec(l.elem(),r.elem(0),d.elem(0));

  return d;
}


// Optimized version of  
//    PScalar<PColorVector<RComplexFloat,3>> <- adj(PScalar<PColorMatrix<RComplexFloat,3>>) * PScalar<PColorVector<RComplexFloat,3>>
template<>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PScalar<PColorVector<RComplexFloat,3> >, OpAdjMultiply>::Type_t
adjMultiply(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	    const PScalar<PColorVector<RComplexFloat,3> >& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PScalar<PColorVector<RComplexFloat,3> >, OpAdjMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "adj(PSc<M>)*PSc<V>" << endl;
#endif

  _inline_generic_mult_adj_su3_mat_vec(l.elem(),r.elem(),d.elem());

  return d;
}


// Optimized version of   StaggeredFermion <- adj(ColorMatrix)*StaggeredFermion
//    PSpinVector<PColorVector<RComplexFloat,3>,1> <- adj(PScalar<PColorMatrix<RComplexFloat,3>>) * PSpinVector<PColorVector<RComplexFloat,3>,1>
template<>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PVector<PColorVector<RComplexFloat,3>,1,PSpinVector>, OpAdjMultiply>::Type_t
adjMultiply(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	    const PVector<PColorVector<RComplexFloat,3>,1,PSpinVector>& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PVector<PColorVector<RComplexFloat,3>,1,PSpinVector>, OpAdjMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "adj(PSc<M>)*S" << endl;
#endif

  _inline_generic_mult_adj_su3_mat_vec(l.elem(),r.elem(0),d.elem(0));

  return d;
}


// Optimized version of    HalfFermion <- ColorMatrix*HalfFermion
//    PSpinVector<PColorVector<RComplexFloat,3>,2> <- PScalar<PColorMatrix<RComplexFloat,3>> * 
//                     PSpinVector<ColorVector<RComplexFloat,3>,2>
template<>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PVector<PColorVector<RComplexFloat,3>,2,PSpinVector>, OpMultiply>::Type_t
operator*(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
          const PVector<PColorVector<RComplexFloat,3>,2,PSpinVector>& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PVector<PColorVector<RComplexFloat,3>,2,PSpinVector>, OpMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "PSc<M>*H" << endl;
#endif

  _inline_generic_mult_su3_mat_vec(l.elem(),r.elem(0),d.elem(0));
  _inline_generic_mult_su3_mat_vec(l.elem(),r.elem(1),d.elem(1));

  return d;
}


// Optimized version of    HalfFermion <- ColorMatrix*HalfFermion
//    PSpinVector<PColorVector<RComplexFloat,3>,2> <- adj(PScalar<PColorMatrix<RComplexFloat,3>>) * 
//                     PSpinVector<ColorVector<RComplexFloat,3>,2>
template<>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PVector<PColorVector<RComplexFloat,3>,2,PSpinVector>, OpAdjMultiply>::Type_t
adjMultiply(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
            const PVector<PColorVector<RComplexFloat,3>,2,PSpinVector>& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PVector<PColorVector<RComplexFloat,3>,2,PSpinVector>, OpAdjMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "adj(PSc<M>)*H" << endl;
#endif

  _inline_generic_mult_adj_su3_mat_vec(l.elem(),r.elem(0),d.elem(0));
  _inline_generic_mult_adj_su3_mat_vec(l.elem(),r.elem(1),d.elem(1));

  return d;
}


// Optimized version of  
//    PColorVector<RComplexFloat,3> <- PColorVector<RComplexFloat,3> + PColorVector<RComplexFloat,3>
template<>
inline BinaryReturn<PVector<RComplexFloat,3,PColorVector>, 
  PVector<RComplexFloat,3,PColorVector>, OpAdd>::Type_t
operator+(const PVector<RComplexFloat,3,PColorVector>& l, 
	  const PVector<RComplexFloat,3,PColorVector>& r)
{
  BinaryReturn<PVector<RComplexFloat,3,PColorVector>, 
    PVector<RComplexFloat,3,PColorVector>, OpAdd>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "V+V" << endl;
#endif

  _inline_generic_add_su3_vector(l,r,d);

  return d;
}


// Optimized version of   DiracFermion <- ColorMatrix*DiracFermion
//    PSpinVector<PColorVector<RComplexFloat,3>,4> <- PScalar<PColorMatrix<RComplexFloat,3>> 
//                           * PSpinVector<PColorVector<RComplexFloat,3>,4>
template<>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PVector<PColorVector<RComplexFloat,3>,4,PSpinVector>, OpMultiply>::Type_t
operator*(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	  const PVector<PColorVector<RComplexFloat,3>,4,PSpinVector>& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PVector<PColorVector<RComplexFloat,3>,4,PSpinVector>, OpMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "PSc<M>*D" << endl;
#endif

  _inline_generic_mult_su3_mat_vec(l.elem(),r.elem(0),d.elem(0));
  _inline_generic_mult_su3_mat_vec(l.elem(),r.elem(1),d.elem(1));
  _inline_generic_mult_su3_mat_vec(l.elem(),r.elem(2),d.elem(2));
  _inline_generic_mult_su3_mat_vec(l.elem(),r.elem(3),d.elem(3));

  return d;
}


// Optimized version of   DiracFermion <- adj(ColorMatrix)*DiracFermion
//    PSpinVector<PColorVector<RComplexFloat,3>,4> <- adj(PScalar<PColorMatrix<RComplexFloat,3>>)
//                           * PSpinVector<PColorVector<RComplexFloat,3>,4>
template<>
inline BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
  PVector<PColorVector<RComplexFloat,3>,4,PSpinVector>, OpAdjMultiply>::Type_t
adjMultiply(const PScalar<PColorMatrix<RComplexFloat,3> >& l, 
	    const PVector<PColorVector<RComplexFloat,3>,4,PSpinVector>& r)
{
  BinaryReturn<PScalar<PColorMatrix<RComplexFloat,3> >, 
    PVector<PColorVector<RComplexFloat,3>,4,PSpinVector>, OpAdjMultiply>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "adj(PSc<M>)*D" << endl;
#endif

  _inline_generic_mult_adj_su3_mat_vec(l.elem(),r.elem(0),d.elem(0));
  _inline_generic_mult_adj_su3_mat_vec(l.elem(),r.elem(1),d.elem(1));
  _inline_generic_mult_adj_su3_mat_vec(l.elem(),r.elem(2),d.elem(2));
  _inline_generic_mult_adj_su3_mat_vec(l.elem(),r.elem(3),d.elem(3));

  return d;
}



// Optimized version of  
//    PScalar<PColorVector<RComplexFloat,3>> <- PScalar<PColorVector<RComplexFloat,3>> 
//                                            + PScalar<PColorVector<RComplexFloat,3>>
template<>
inline BinaryReturn<PScalar<PColorVector<RComplexFloat,3> >, 
  PScalar<PColorVector<RComplexFloat,3> >, OpAdd>::Type_t
operator+(const PScalar<PColorVector<RComplexFloat,3> >& l, 
	  const PScalar<PColorVector<RComplexFloat,3> >& r)
{
  BinaryReturn<PScalar<PColorVector<RComplexFloat,3> >, 
    PScalar<PColorVector<RComplexFloat,3> >, OpAdd>::Type_t  d;

#if defined(QDP_SCALARSITE_DEBUG)
  cout << "PSc<V>+PSc<V>" << endl;
#endif

  _inline_generic_add_su3_vector(l.elem(),r.elem(),d.elem());

  return d;
}


#if 1
// Specialization to optimize the case   
//    LatticeColorMatrix = LatticeColorMatrix * LatticeColorMatrix
// NOTE: let this be a subroutine to save space
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

  
   double fp_regs[32];
   _save_fp_regs(fp_regs);

   if( s.hasOrderedRep() ) { 
     for(int i = s.start(); i <= s.end(); i++) { 
       _inline_qcdoc_mult_su3_nn(l.elem(i).elem(),
				 r.elem(i).elem(),
				 d.elem(i).elem());
     }
   }
   else { 

     const int* tab = s.siteTable().slice();
     for(int j=0; j < s.numSiteTable(); j++) {
       int i = tab[j];
       _inline_qcdoc_mult_su3_nn(l.elem(i).elem(),
				 r.elem(i).elem(),
				 d.elem(i).elem());
     }
   }
   _restore_fp_regs(fp_regs);
}

#endif

#if 1
// Specialization to optimize the case   
//    LatticeHalfFermion = LatticeColorMatrix * LatticeHalfFermion
// NOTE: let this be a subroutine to save space
template<>
inline 
void evaluate(OLattice<PSpinVector<PColorVector<RComplexFloat, 3>, 2> >& d, 
	      const OpAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	      Reference<QDPType<PScalar<PColorMatrix<RComplexFloat, 3> >, 
	      OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > > >, 
	      Reference<QDPType<PSpinVector<PColorVector<RComplexFloat, 3>, 2>, 
	      OLattice<PSpinVector<PColorVector<RComplexFloat, 3>, 2> > > > >,
	      OLattice<PSpinVector<PColorVector<RComplexFloat, 3>, 2> > >& rhs,
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
      _inline_generic_mult_su3_mat_vec(l.elem(i).elem(),
				       r.elem(i).elem(0),
				       d.elem(i).elem(0));
      _inline_generic_mult_su3_mat_vec(l.elem(i).elem(),
				       r.elem(i).elem(1),
				       d.elem(i).elem(1));
    }
  }
  else { 
    const int* tab=s.siteTable().slice();
    for(int j=0; j <= s.numSiteTable(); ++j) {
      
      int i=tab[j];
      _inline_generic_mult_su3_mat_vec(l.elem(i).elem(),
				       r.elem(i).elem(0),
				       d.elem(i).elem(0));
      _inline_generic_mult_su3_mat_vec(l.elem(i).elem(),
				       r.elem(i).elem(1),
				       d.elem(i).elem(1));
    }
  }
}

#endif

/*! @} */   // end of group optimizations

#if defined(QDP_SCALARSITE_DEBUG)
#undef QDP_SCALARSITE_DEBUG
#endif

} // namespace QDP;

#endif
