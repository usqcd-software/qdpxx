// -*- C++ -*-
// $Id: qdp_scalarvecsite_sse_linalg.h,v 1.3 2004-08-10 03:55:50 edwards Exp $

/*! @file
 * @brief Intel SSE optimizations
 *
 * SSE optimizations of basic operations
 */

#ifndef QDP_SCALARVECSITE_SSE_LINALG_H
#define QDP_SCALARVECSITE_SSE_LINALG_H

// These SSE asm instructions are only supported under GCC/G++ 3.2 or greater
#if defined(__GNUC__) && __GNUC_MINOR__ >= 2

QDP_BEGIN_NAMESPACE(QDP);

/*! @defgroup optimizations  Optimizations
 *
 * Optimizations for basic QDP operations
 *
 * @{
 */

// Use this def just to safe some typing later on in the file
typedef ILattice<REAL32,4>             ILatticeFloat;
typedef RComplex<ILattice<REAL32,4> >  RComplexFloat; 

#include "scalarvecsite_sse/ssevec_mult_nn.h"

//--------------------------------------------------------------------------------------
// Optimized version of  
//    ILatticeFloat <- ILatticeFloat + ILatticeFloat
template<>
inline BinaryReturn<ILatticeFloat, ILatticeFloat, OpAdd>::Type_t
operator+(const ILatticeFloat& l, const ILatticeFloat& r)
{
  typedef BinaryReturn<ILatticeFloat, ILatticeFloat, OpAdd>::Type_t  Ret_t;

//  cout << "I+I" << endl; 
  return Ret_t(__builtin_ia32_addps(l.elem_v(), r.elem_v()));
}


// Optimized version of  
//    ILatticeFloat <- ILatticeFloat - ILatticeFloat
template<>
inline BinaryReturn<ILatticeFloat, ILatticeFloat, OpSubtract>::Type_t
operator-(const ILatticeFloat& l, const ILatticeFloat& r)
{
  typedef BinaryReturn<ILatticeFloat, ILatticeFloat, OpSubtract>::Type_t  Ret_t;

//  cout << "I-I" << endl;
  return Ret_t(__builtin_ia32_subps(l.elem_v(), r.elem_v()));
}


// Optimized version of  
//    ILatticeFloat <- ILatticeFloat * ILatticeFloat
template<>
inline BinaryReturn<ILatticeFloat, ILatticeFloat, OpMultiply>::Type_t
operator*(const ILatticeFloat& l, const ILatticeFloat& r)
{
  typedef BinaryReturn<ILatticeFloat, ILatticeFloat, OpMultiply>::Type_t  Ret_t;

//  cout << "I*I" << endl;
  return Ret_t(__builtin_ia32_mulps(l.elem_v(), r.elem_v()));
}


// Optimized version of  
//    ILatticeFloat <- ILatticeFloat / ILatticeFloat
template<>
inline BinaryReturn<ILatticeFloat, ILatticeFloat, OpDivide>::Type_t
operator/(const ILatticeFloat& l, const ILatticeFloat& r)
{
  typedef BinaryReturn<ILatticeFloat, ILatticeFloat, OpDivide>::Type_t  Ret_t;

//  cout << "I/I" << endl;
  return Ret_t(__builtin_ia32_divps(l.elem_v(), r.elem_v()));
}



//--------------------------------------------------------------------------------------
// Optimized version of  
//    RComplexFloat <- RComplexFloat + RComplexFloat
template<>
inline BinaryReturn<RComplexFloat, RComplexFloat, OpAdd>::Type_t
operator+(const RComplexFloat& l, const RComplexFloat& r)
{
  typedef BinaryReturn<RComplexFloat, RComplexFloat, OpAdd>::Type_t  Ret_t;

//  cout << "C+C" << endl;
  return Ret_t(__builtin_ia32_addps(l.real().elem_v(), r.real().elem_v()),
	       __builtin_ia32_addps(l.imag().elem_v(), r.imag().elem_v()));
}


// Optimized version of  
//    RComplexFloat <- RComplexFloat - RComplexFloat
template<>
inline BinaryReturn<RComplexFloat, RComplexFloat, OpSubtract>::Type_t
operator-(const RComplexFloat& l, const RComplexFloat& r)
{
  typedef BinaryReturn<RComplexFloat, RComplexFloat, OpSubtract>::Type_t  Ret_t;

//  cout << "C-C" << endl;
  return Ret_t(__builtin_ia32_subps(l.real().elem_v(), r.real().elem_v()),
	       __builtin_ia32_subps(l.imag().elem_v(), r.imag().elem_v()));
}


// Optimized version of  
//    RComplexFloat <- RComplexFloat * RComplexFloat
template<>
inline BinaryReturn<RComplexFloat, RComplexFloat, OpMultiply>::Type_t
operator*(const RComplexFloat& l, const RComplexFloat& r)
{
  typedef BinaryReturn<RComplexFloat, RComplexFloat, OpMultiply>::Type_t  Ret_t;

//  cout << "C*C" << endl;
  return Ret_t(__builtin_ia32_subps(__builtin_ia32_mulps(l.real().elem_v(), r.real().elem_v()),
				    __builtin_ia32_mulps(l.imag().elem_v(), r.imag().elem_v())),
	       __builtin_ia32_addps(__builtin_ia32_mulps(l.real().elem_v(), r.imag().elem_v()),
				    __builtin_ia32_mulps(l.imag().elem_v(), r.real().elem_v())));
}

// Optimized version of  
//    RComplexFloat <- adj(RComplexFloat) * RComplexFloat
template<>
inline BinaryReturn<RComplexFloat, RComplexFloat, OpAdjMultiply>::Type_t
adjMultiply(const RComplexFloat& l, const RComplexFloat& r)
{
  typedef BinaryReturn<RComplexFloat, RComplexFloat, OpAdjMultiply>::Type_t  Ret_t;

//  cout << "adj(C)*C" << endl;
  return Ret_t(__builtin_ia32_addps(__builtin_ia32_mulps(l.real().elem_v(), r.real().elem_v()),
				    __builtin_ia32_mulps(l.imag().elem_v(), r.imag().elem_v())),
	       __builtin_ia32_subps(__builtin_ia32_mulps(l.real().elem_v(), r.imag().elem_v()),
				    __builtin_ia32_mulps(l.imag().elem_v(), r.real().elem_v())));
}

// Optimized  RComplex*adj(RComplex)
template<>
inline BinaryReturn<RComplexFloat, RComplexFloat, OpMultiplyAdj>::Type_t
multiplyAdj(const RComplexFloat& l, const RComplexFloat& r)
{
  typedef BinaryReturn<RComplexFloat, RComplexFloat, OpMultiplyAdj>::Type_t  Ret_t;

//  cout << "C*adj(C)" << endl;
  return Ret_t(__builtin_ia32_addps(__builtin_ia32_mulps(l.real().elem_v(), r.real().elem_v()),
				    __builtin_ia32_mulps(l.imag().elem_v(), r.imag().elem_v())),
	       __builtin_ia32_subps(__builtin_ia32_mulps(l.imag().elem_v(), r.real().elem_v()),
				    __builtin_ia32_mulps(l.real().elem_v(), r.imag().elem_v())));
}

// Optimized  adj(RComplex)*adj(RComplex)
template<>
inline BinaryReturn<RComplexFloat, RComplexFloat, OpAdjMultiplyAdj>::Type_t
adjMultiplyAdj(const RComplexFloat& l, const RComplexFloat& r)
{
  typedef BinaryReturn<RComplexFloat, RComplexFloat, OpAdjMultiplyAdj>::Type_t  Ret_t;
  REAL32 zero = 0.0;

//  cout << "adj(C)*adj(C)" << endl;
  return Ret_t(__builtin_ia32_subps(__builtin_ia32_mulps(l.real().elem_v(), r.real().elem_v()),
				    __builtin_ia32_mulps(l.imag().elem_v(), r.imag().elem_v())),
	       __builtin_ia32_subps(vmk1(zero),
				    __builtin_ia32_addps(__builtin_ia32_mulps(l.real().elem_v(), r.imag().elem_v()),
							 __builtin_ia32_mulps(l.imag().elem_v(), r.real().elem_v()))));
}


//--------------------------------------------------------------------------------------


// Optimized version of  
//    PColorMatrix<RComplexFloat,3> <- PColorMatrix<RComplexFloat,3> * PColorMatrix<RComplexFloat,3>
template<>
inline BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
  PMatrix<RComplexFloat,3,PColorMatrix>, OpMultiply>::Type_t
operator*(const PMatrix<RComplexFloat,3,PColorMatrix>& l, 
	  const PMatrix<RComplexFloat,3,PColorMatrix>& r)
{
//  cout << "M*M" << endl;

  BinaryReturn<PMatrix<RComplexFloat,3,PColorMatrix>, 
    PMatrix<RComplexFloat,3,PColorMatrix>, OpMultiply>::Type_t  d;

  REAL32 *dd = (REAL32*)&d;
  REAL32 *ll = (REAL32*)&l;
  REAL32 *rr = (REAL32*)&r;

  _inline_ssevec_mult_su3_nn(dd,ll,rr,0);
  _inline_ssevec_mult_su3_nn(dd,ll,rr,1);
  _inline_ssevec_mult_su3_nn(dd,ll,rr,2);

  return d;
}



// Specialization to optimize the case   
//    LatticeColorMatrix = LatticeColorMatrix * LatticeColorMatrix
// NOTE: let this be a subroutine to save space
template<>
void evaluate(OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >& d, 
	      const OpAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	      Reference<QDPType<PScalar<PColorMatrix<RComplexFloat, 3> >, 
	      OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > > >, 
	      Reference<QDPType<PScalar<PColorMatrix<RComplexFloat, 3> >, 
	      OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > > > >,
	      OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > >& rhs,
	      const OrderedSubset& s);


/*! @} */   // end of group optimizations

QDP_END_NAMESPACE();

#endif  // defined(__GNUC__)

#endif
