// $Id: qdp_scalarsite_sse.cc,v 1.3 2003-08-08 19:13:06 edwards Exp $

/*! @file
 * @brief Intel SSE optimizations
 * 
 * SSE optimizations of basic operations
 */


#include "qdp.h"

// These SSE asm instructions are only supported under GCC/G++
#if defined(__GNUC__)

QDP_BEGIN_NAMESPACE(QDP);

// Specialization to optimize the case   
//    LatticeColorMatrix[OrderedSubset] = LatticeColorMatrix * LatticeColorMatrix
template<>
void evaluate(OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >& d, 
	      const OpAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	      Reference<QDPType<PScalar<PColorMatrix<RComplexFloat, 3> >, 
	      OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > > >, 
	      Reference<QDPType<PScalar<PColorMatrix<RComplexFloat, 3> >, 
	      OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > > > >,
	      OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > > >& rhs,
	      const OrderedSubset& s)
{
// cout << "call single site QDP_M_eq_M_times_M" << endl;

  const LatticeColorMatrix& l = static_cast<const LatticeColorMatrix&>(rhs.expression().left());
  const LatticeColorMatrix& r = static_cast<const LatticeColorMatrix&>(rhs.expression().right());

  for(int i=s.start(); i <= s.end(); ++i) 
  {
    _inline_sse_mult_su3_nn(l.elem(i).elem(),r.elem(i).elem(),d.elem(i).elem());
  }
}

QDP_END_NAMESPACE();

#endif  // defined(__GNUC__)
