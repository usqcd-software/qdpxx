// $Id: qdp_scalarsite_sse.cc,v 1.22 2006-09-26 15:20:39 edwards Exp $

/*! @file
 * @brief Intel SSE optimizations
 * 
 * SSE optimizations of basic operations
 */


#include "qdp.h"


// These SSE asm instructions are only supported under GCC/G++
#if defined(__GNUC__)
#include "qdp_sse_intrin.h"
QDP_BEGIN_NAMESPACE(QDP);

#if BASE_PRECISION==32


#if 1
//-------------------------------------------------------------------
// Specialization to optimize the case   
//    LatticeColorMatrix[OrderedSubset] = LatticeColorMatrix * LatticeColorMatrix
template<>
void evaluate(OLattice< TCol >& d, 
	      const OpAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	                    Reference<QDPType< TCol, OLattice< TCol > > >, 
	                    Reference<QDPType< TCol, OLattice< TCol > > > >,
	                    OLattice< TCol > >& rhs,
	      const OrderedSubset& s)
{
//  cout << "call single site QDP_M_eq_M_times_M" << endl;

  typedef OLattice< TCol >    C;

  const C& l = static_cast<const C&>(rhs.expression().left());
  const C& r = static_cast<const C&>(rhs.expression().right());

  for(int i=s.start(); i <= s.end(); ++i) 
  {
    _inline_sse_mult_su3_nn(l.elem(i).elem(),r.elem(i).elem(),d.elem(i).elem());
  }
}


// Specialization to optimize the case   
//    LatticeColorMatrix[OrderedSubset] = adj(LatticeColorMatrix) * LatticeColorMatrix
template<>
void evaluate(OLattice< TCol >& d, 
	      const OpAssign& op, 
	      const QDPExpr<BinaryNode<OpAdjMultiply, 
	                    UnaryNode<OpIdentity, Reference<QDPType< TCol, OLattice< TCol > > > >, 
	                    Reference<QDPType< TCol, OLattice< TCol > > > >,
	                    OLattice< TCol > >& rhs,
	      const OrderedSubset& s)
{
//  cout << "call single site QDP_M_eq_aM_times_M" << endl;

  typedef OLattice< TCol >    C;

  const C& l = static_cast<const C&>(rhs.expression().left().child());
  const C& r = static_cast<const C&>(rhs.expression().right());

  for(int i=s.start(); i <= s.end(); ++i) 
  {
    _inline_sse_mult_su3_an(l.elem(i).elem(),r.elem(i).elem(),d.elem(i).elem());
  }
}


// Specialization to optimize the case   
//    LatticeColorMatrix[OrderedSubset] = LatticeColorMatrix * adj(LatticeColorMatrix)
template<>
void evaluate(OLattice< TCol >& d, 
	      const OpAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiplyAdj, 
	                    Reference<QDPType< TCol, OLattice< TCol > > >, 
	                    UnaryNode<OpIdentity, Reference<QDPType< TCol, OLattice< TCol > > > > >,
	                    OLattice< TCol > >& rhs,
	      const OrderedSubset& s)
{
//  cout << "call single site QDP_M_eq_M_times_aM" << endl;

  typedef OLattice< TCol >    C;

  const C& l = static_cast<const C&>(rhs.expression().left());
  const C& r = static_cast<const C&>(rhs.expression().right().child());

  for(int i=s.start(); i <= s.end(); ++i) 
  {
    _inline_sse_mult_su3_na(l.elem(i).elem(),r.elem(i).elem(),d.elem(i).elem());
  }
}


// Specialization to optimize the case   
//    LatticeColorMatrix[OrderedSubset] = adj(LatticeColorMatrix) * adj(LatticeColorMatrix)
template<>
void evaluate(OLattice< TCol >& d, 
	      const OpAssign& op, 
	      const QDPExpr<BinaryNode<OpAdjMultiplyAdj, 
	                    UnaryNode<OpIdentity, Reference<QDPType< TCol, OLattice< TCol > > > >,
	                    UnaryNode<OpIdentity, Reference<QDPType< TCol, OLattice< TCol > > > > >,
	                    OLattice< TCol > >& rhs,
	      const OrderedSubset& s)
{
//  cout << "call single site QDP_M_eq_Ma_times_Ma" << endl;

  typedef OLattice< TCol >    C;

  const C& l = static_cast<const C&>(rhs.expression().left().child());
  const C& r = static_cast<const C&>(rhs.expression().right().child());

  PColorMatrix<RComplexFloat,3> tmp;

  for(int i=s.start(); i <= s.end(); ++i) 
  {
    _inline_sse_mult_su3_nn(r.elem(i).elem(),l.elem(i).elem(),tmp);

    // Take the adj(r*l) = adj(l)*adj(r)
    d.elem(i).elem().elem(0,0).real() =  tmp.elem(0,0).real();
    d.elem(i).elem().elem(0,0).imag() = -tmp.elem(0,0).imag();
    d.elem(i).elem().elem(0,1).real() =  tmp.elem(1,0).real();
    d.elem(i).elem().elem(0,1).imag() = -tmp.elem(1,0).imag();
    d.elem(i).elem().elem(0,2).real() =  tmp.elem(2,0).real();
    d.elem(i).elem().elem(0,2).imag() = -tmp.elem(2,0).imag();

    d.elem(i).elem().elem(1,0).real() =  tmp.elem(0,1).real();
    d.elem(i).elem().elem(1,0).imag() = -tmp.elem(0,1).imag();
    d.elem(i).elem().elem(1,1).real() =  tmp.elem(1,1).real();
    d.elem(i).elem().elem(1,1).imag() = -tmp.elem(1,1).imag();
    d.elem(i).elem().elem(1,2).real() =  tmp.elem(2,1).real();
    d.elem(i).elem().elem(1,2).imag() = -tmp.elem(2,1).imag();

    d.elem(i).elem().elem(2,0).real() =  tmp.elem(0,2).real();
    d.elem(i).elem().elem(2,0).imag() = -tmp.elem(0,2).imag();
    d.elem(i).elem().elem(2,1).real() =  tmp.elem(1,2).real();
    d.elem(i).elem().elem(2,1).imag() = -tmp.elem(1,2).imag();
    d.elem(i).elem().elem(2,2).real() =  tmp.elem(2,2).real();
    d.elem(i).elem().elem(2,2).imag() = -tmp.elem(2,2).imag();
  }
}


//-------------------------------------------------------------------

// Specialization to optimize the case   
//    LatticeColorMatrix[OrderedSubset] += LatticeColorMatrix * LatticeColorMatrix
template<>
void evaluate(OLattice< TCol >& d, 
	      const OpAddAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	                    Reference<QDPType< TCol, OLattice< TCol > > >, 
	                    Reference<QDPType< TCol, OLattice< TCol > > > >,
	                    OLattice< TCol > >& rhs,
	      const OrderedSubset& s)
{
//  cout << "call single site QDP_M_peq_M_times_M" << endl;

  typedef OLattice< TCol >    C;

  const C& l = static_cast<const C&>(rhs.expression().left());
  const C& r = static_cast<const C&>(rhs.expression().right());

  PColorMatrix<RComplexFloat,3> tmp;

  for(int i=s.start(); i <= s.end(); ++i) 
  {
    _inline_sse_mult_su3_nn(l.elem(i).elem(),r.elem(i).elem(),tmp);

    d.elem(i).elem().elem(0,0).real() += tmp.elem(0,0).real();
    d.elem(i).elem().elem(0,0).imag() += tmp.elem(0,0).imag();
    d.elem(i).elem().elem(0,1).real() += tmp.elem(0,1).real();
    d.elem(i).elem().elem(0,1).imag() += tmp.elem(0,1).imag();
    d.elem(i).elem().elem(0,2).real() += tmp.elem(0,2).real();
    d.elem(i).elem().elem(0,2).imag() += tmp.elem(0,2).imag();

    d.elem(i).elem().elem(1,0).real() += tmp.elem(1,0).real();
    d.elem(i).elem().elem(1,0).imag() += tmp.elem(1,0).imag();
    d.elem(i).elem().elem(1,1).real() += tmp.elem(1,1).real();
    d.elem(i).elem().elem(1,1).imag() += tmp.elem(1,1).imag();
    d.elem(i).elem().elem(1,2).real() += tmp.elem(1,2).real();
    d.elem(i).elem().elem(1,2).imag() += tmp.elem(1,2).imag();

    d.elem(i).elem().elem(2,0).real() += tmp.elem(2,0).real();
    d.elem(i).elem().elem(2,0).imag() += tmp.elem(2,0).imag();
    d.elem(i).elem().elem(2,1).real() += tmp.elem(2,1).real();
    d.elem(i).elem().elem(2,1).imag() += tmp.elem(2,1).imag();
    d.elem(i).elem().elem(2,2).real() += tmp.elem(2,2).real();
    d.elem(i).elem().elem(2,2).imag() += tmp.elem(2,2).imag();
  }
}


// Specialization to optimize the case   
//    LatticeColorMatrix[OrderedSubset] += adj(LatticeColorMatrix) * LatticeColorMatrix
template<>
void evaluate(OLattice< TCol >& d, 
	      const OpAddAssign& op, 
	      const QDPExpr<BinaryNode<OpAdjMultiply, 
	                    UnaryNode<OpIdentity, Reference<QDPType< TCol, OLattice< TCol > > > >, 
	                    Reference<QDPType< TCol, OLattice< TCol > > > >,
	                    OLattice< TCol > >& rhs,
	      const OrderedSubset& s)
{
//  cout << "call single site QDP_M_peq_aM_times_M" << endl;

  typedef OLattice< TCol >    C;

  const C& l = static_cast<const C&>(rhs.expression().left().child());
  const C& r = static_cast<const C&>(rhs.expression().right());

  PColorMatrix<RComplexFloat,3> tmp;

  for(int i=s.start(); i <= s.end(); ++i) 
  {
    _inline_sse_mult_su3_an(l.elem(i).elem(),r.elem(i).elem(),tmp);

    d.elem(i).elem().elem(0,0).real() += tmp.elem(0,0).real();
    d.elem(i).elem().elem(0,0).imag() += tmp.elem(0,0).imag();
    d.elem(i).elem().elem(0,1).real() += tmp.elem(0,1).real();
    d.elem(i).elem().elem(0,1).imag() += tmp.elem(0,1).imag();
    d.elem(i).elem().elem(0,2).real() += tmp.elem(0,2).real();
    d.elem(i).elem().elem(0,2).imag() += tmp.elem(0,2).imag();

    d.elem(i).elem().elem(1,0).real() += tmp.elem(1,0).real();
    d.elem(i).elem().elem(1,0).imag() += tmp.elem(1,0).imag();
    d.elem(i).elem().elem(1,1).real() += tmp.elem(1,1).real();
    d.elem(i).elem().elem(1,1).imag() += tmp.elem(1,1).imag();
    d.elem(i).elem().elem(1,2).real() += tmp.elem(1,2).real();
    d.elem(i).elem().elem(1,2).imag() += tmp.elem(1,2).imag();

    d.elem(i).elem().elem(2,0).real() += tmp.elem(2,0).real();
    d.elem(i).elem().elem(2,0).imag() += tmp.elem(2,0).imag();
    d.elem(i).elem().elem(2,1).real() += tmp.elem(2,1).real();
    d.elem(i).elem().elem(2,1).imag() += tmp.elem(2,1).imag();
    d.elem(i).elem().elem(2,2).real() += tmp.elem(2,2).real();
    d.elem(i).elem().elem(2,2).imag() += tmp.elem(2,2).imag();
  }
}


// Specialization to optimize the case   
//    LatticeColorMatrix[OrderedSubset] += LatticeColorMatrix * adj(LatticeColorMatrix)
template<>
void evaluate(OLattice< TCol >& d, 
	      const OpAddAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiplyAdj, 
	                    Reference<QDPType< TCol, OLattice< TCol > > >, 
	                    UnaryNode<OpIdentity, Reference<QDPType< TCol, OLattice< TCol > > > > >,
	                    OLattice< TCol > >& rhs,
	      const OrderedSubset& s)
{
//  cout << "call single site QDP_M_peq_M_times_aM" << endl;

  typedef OLattice< TCol >    C;

  const C& l = static_cast<const C&>(rhs.expression().left());
  const C& r = static_cast<const C&>(rhs.expression().right().child());

  PColorMatrix<RComplexFloat,3> tmp;

  for(int i=s.start(); i <= s.end(); ++i) 
  {
    _inline_sse_mult_su3_na(l.elem(i).elem(),r.elem(i).elem(),tmp);

    d.elem(i).elem().elem(0,0).real() += tmp.elem(0,0).real();
    d.elem(i).elem().elem(0,0).imag() += tmp.elem(0,0).imag();
    d.elem(i).elem().elem(0,1).real() += tmp.elem(0,1).real();
    d.elem(i).elem().elem(0,1).imag() += tmp.elem(0,1).imag();
    d.elem(i).elem().elem(0,2).real() += tmp.elem(0,2).real();
    d.elem(i).elem().elem(0,2).imag() += tmp.elem(0,2).imag();

    d.elem(i).elem().elem(1,0).real() += tmp.elem(1,0).real();
    d.elem(i).elem().elem(1,0).imag() += tmp.elem(1,0).imag();
    d.elem(i).elem().elem(1,1).real() += tmp.elem(1,1).real();
    d.elem(i).elem().elem(1,1).imag() += tmp.elem(1,1).imag();
    d.elem(i).elem().elem(1,2).real() += tmp.elem(1,2).real();
    d.elem(i).elem().elem(1,2).imag() += tmp.elem(1,2).imag();

    d.elem(i).elem().elem(2,0).real() += tmp.elem(2,0).real();
    d.elem(i).elem().elem(2,0).imag() += tmp.elem(2,0).imag();
    d.elem(i).elem().elem(2,1).real() += tmp.elem(2,1).real();
    d.elem(i).elem().elem(2,1).imag() += tmp.elem(2,1).imag();
    d.elem(i).elem().elem(2,2).real() += tmp.elem(2,2).real();
    d.elem(i).elem().elem(2,2).imag() += tmp.elem(2,2).imag();
  }
}


// Specialization to optimize the case   
//    LatticeColorMatrix[OrderedSubset] += adj(LatticeColorMatrix) * adj(LatticeColorMatrix)
template<>
void evaluate(OLattice< TCol >& d, 
	      const OpAddAssign& op, 
	      const QDPExpr<BinaryNode<OpAdjMultiplyAdj, 
	                    UnaryNode<OpIdentity, Reference<QDPType< TCol, OLattice< TCol > > > >,
	                    UnaryNode<OpIdentity, Reference<QDPType< TCol, OLattice< TCol > > > > >,
	                    OLattice< TCol > >& rhs,
	      const OrderedSubset& s)
{
//  cout << "call single site QDP_M_peq_Ma_times_Ma" << endl;

  typedef OLattice< TCol >    C;

  const C& l = static_cast<const C&>(rhs.expression().left().child());
  const C& r = static_cast<const C&>(rhs.expression().right().child());

  PColorMatrix<RComplexFloat,3> tmp;

  for(int i=s.start(); i <= s.end(); ++i) 
  {
    _inline_sse_mult_su3_nn(r.elem(i).elem(),l.elem(i).elem(),tmp);

    // Take the adj(r*l) = adj(l)*adj(r)
    d.elem(i).elem().elem(0,0).real() += tmp.elem(0,0).real();
    d.elem(i).elem().elem(0,0).imag() -= tmp.elem(0,0).imag();
    d.elem(i).elem().elem(0,1).real() += tmp.elem(1,0).real();
    d.elem(i).elem().elem(0,1).imag() -= tmp.elem(1,0).imag();
    d.elem(i).elem().elem(0,2).real() += tmp.elem(2,0).real();
    d.elem(i).elem().elem(0,2).imag() -= tmp.elem(2,0).imag();

    d.elem(i).elem().elem(1,0).real() += tmp.elem(0,1).real();
    d.elem(i).elem().elem(1,0).imag() -= tmp.elem(0,1).imag();
    d.elem(i).elem().elem(1,1).real() += tmp.elem(1,1).real();
    d.elem(i).elem().elem(1,1).imag() -= tmp.elem(1,1).imag();
    d.elem(i).elem().elem(1,2).real() += tmp.elem(2,1).real();
    d.elem(i).elem().elem(1,2).imag() -= tmp.elem(2,1).imag();

    d.elem(i).elem().elem(2,0).real() += tmp.elem(0,2).real();
    d.elem(i).elem().elem(2,0).imag() -= tmp.elem(0,2).imag();
    d.elem(i).elem().elem(2,1).real() += tmp.elem(1,2).real();
    d.elem(i).elem().elem(2,1).imag() -= tmp.elem(1,2).imag();
    d.elem(i).elem().elem(2,2).real() += tmp.elem(2,2).real();
    d.elem(i).elem().elem(2,2).imag() -= tmp.elem(2,2).imag();
  }
}

//-------------------------------------------------------------------

// Specialization to optimize the case   
//    LatticeColorMatrix[OrderedSubset] -= LatticeColorMatrix * LatticeColorMatrix
template<>
void evaluate(OLattice< TCol >& d, 
	      const OpSubtractAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	                    Reference<QDPType< TCol, OLattice< TCol > > >, 
	                    Reference<QDPType< TCol, OLattice< TCol > > > >,
	                    OLattice< TCol > >& rhs,
	      const OrderedSubset& s)
{
//  cout << "call single site QDP_M_meq_M_times_M" << endl;

  typedef OLattice< TCol >    C;

  const C& l = static_cast<const C&>(rhs.expression().left());
  const C& r = static_cast<const C&>(rhs.expression().right());

  PColorMatrix<RComplexFloat,3> tmp;

  for(int i=s.start(); i <= s.end(); ++i) 
  {
    _inline_sse_mult_su3_nn(l.elem(i).elem(),r.elem(i).elem(),tmp);

    d.elem(i).elem().elem(0,0).real() -= tmp.elem(0,0).real();
    d.elem(i).elem().elem(0,0).imag() -= tmp.elem(0,0).imag();
    d.elem(i).elem().elem(0,1).real() -= tmp.elem(0,1).real();
    d.elem(i).elem().elem(0,1).imag() -= tmp.elem(0,1).imag();
    d.elem(i).elem().elem(0,2).real() -= tmp.elem(0,2).real();
    d.elem(i).elem().elem(0,2).imag() -= tmp.elem(0,2).imag();

    d.elem(i).elem().elem(1,0).real() -= tmp.elem(1,0).real();
    d.elem(i).elem().elem(1,0).imag() -= tmp.elem(1,0).imag();
    d.elem(i).elem().elem(1,1).real() -= tmp.elem(1,1).real();
    d.elem(i).elem().elem(1,1).imag() -= tmp.elem(1,1).imag();
    d.elem(i).elem().elem(1,2).real() -= tmp.elem(1,2).real();
    d.elem(i).elem().elem(1,2).imag() -= tmp.elem(1,2).imag();

    d.elem(i).elem().elem(2,0).real() -= tmp.elem(2,0).real();
    d.elem(i).elem().elem(2,0).imag() -= tmp.elem(2,0).imag();
    d.elem(i).elem().elem(2,1).real() -= tmp.elem(2,1).real();
    d.elem(i).elem().elem(2,1).imag() -= tmp.elem(2,1).imag();
    d.elem(i).elem().elem(2,2).real() -= tmp.elem(2,2).real();
    d.elem(i).elem().elem(2,2).imag() -= tmp.elem(2,2).imag();
  }
}


// Specialization to optimize the case   
//    LatticeColorMatrix[OrderedSubset] -= adj(LatticeColorMatrix) * LatticeColorMatrix
template<>
void evaluate(OLattice< TCol >& d, 
	      const OpSubtractAssign& op, 
	      const QDPExpr<BinaryNode<OpAdjMultiply, 
	                    UnaryNode<OpIdentity, Reference<QDPType< TCol, OLattice< TCol > > > >, 
	                    Reference<QDPType< TCol, OLattice< TCol > > > >,
	                    OLattice< TCol > >& rhs,
	      const OrderedSubset& s)
{
//  cout << "call single site QDP_M_meq_aM_times_M" << endl;

  typedef OLattice< TCol >    C;

  const C& l = static_cast<const C&>(rhs.expression().left().child());
  const C& r = static_cast<const C&>(rhs.expression().right());

  PColorMatrix<RComplexFloat,3> tmp;

  for(int i=s.start(); i <= s.end(); ++i) 
  {
    _inline_sse_mult_su3_an(l.elem(i).elem(),r.elem(i).elem(),tmp);

    d.elem(i).elem().elem(0,0).real() -= tmp.elem(0,0).real();
    d.elem(i).elem().elem(0,0).imag() -= tmp.elem(0,0).imag();
    d.elem(i).elem().elem(0,1).real() -= tmp.elem(0,1).real();
    d.elem(i).elem().elem(0,1).imag() -= tmp.elem(0,1).imag();
    d.elem(i).elem().elem(0,2).real() -= tmp.elem(0,2).real();
    d.elem(i).elem().elem(0,2).imag() -= tmp.elem(0,2).imag();

    d.elem(i).elem().elem(1,0).real() -= tmp.elem(1,0).real();
    d.elem(i).elem().elem(1,0).imag() -= tmp.elem(1,0).imag();
    d.elem(i).elem().elem(1,1).real() -= tmp.elem(1,1).real();
    d.elem(i).elem().elem(1,1).imag() -= tmp.elem(1,1).imag();
    d.elem(i).elem().elem(1,2).real() -= tmp.elem(1,2).real();
    d.elem(i).elem().elem(1,2).imag() -= tmp.elem(1,2).imag();

    d.elem(i).elem().elem(2,0).real() -= tmp.elem(2,0).real();
    d.elem(i).elem().elem(2,0).imag() -= tmp.elem(2,0).imag();
    d.elem(i).elem().elem(2,1).real() -= tmp.elem(2,1).real();
    d.elem(i).elem().elem(2,1).imag() -= tmp.elem(2,1).imag();
    d.elem(i).elem().elem(2,2).real() -= tmp.elem(2,2).real();
    d.elem(i).elem().elem(2,2).imag() -= tmp.elem(2,2).imag();
  }
}


// Specialization to optimize the case   
//    LatticeColorMatrix[OrderedSubset] -= LatticeColorMatrix * adj(LatticeColorMatrix)
template<>
void evaluate(OLattice< TCol >& d, 
	      const OpSubtractAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiplyAdj, 
	                    Reference<QDPType< TCol, OLattice< TCol > > >, 
	                    UnaryNode<OpIdentity, Reference<QDPType< TCol, OLattice< TCol > > > > >,
	                    OLattice< TCol > >& rhs,
	      const OrderedSubset& s)
{
//  cout << "call single site QDP_M_meq_M_times_aM" << endl;

  typedef OLattice< TCol >    C;

  const C& l = static_cast<const C&>(rhs.expression().left());
  const C& r = static_cast<const C&>(rhs.expression().right().child());

  PColorMatrix<RComplexFloat,3> tmp;

  for(int i=s.start(); i <= s.end(); ++i) 
  {
    _inline_sse_mult_su3_na(l.elem(i).elem(),r.elem(i).elem(),tmp);

    d.elem(i).elem().elem(0,0).real() -= tmp.elem(0,0).real();
    d.elem(i).elem().elem(0,0).imag() -= tmp.elem(0,0).imag();
    d.elem(i).elem().elem(0,1).real() -= tmp.elem(0,1).real();
    d.elem(i).elem().elem(0,1).imag() -= tmp.elem(0,1).imag();
    d.elem(i).elem().elem(0,2).real() -= tmp.elem(0,2).real();
    d.elem(i).elem().elem(0,2).imag() -= tmp.elem(0,2).imag();

    d.elem(i).elem().elem(1,0).real() -= tmp.elem(1,0).real();
    d.elem(i).elem().elem(1,0).imag() -= tmp.elem(1,0).imag();
    d.elem(i).elem().elem(1,1).real() -= tmp.elem(1,1).real();
    d.elem(i).elem().elem(1,1).imag() -= tmp.elem(1,1).imag();
    d.elem(i).elem().elem(1,2).real() -= tmp.elem(1,2).real();
    d.elem(i).elem().elem(1,2).imag() -= tmp.elem(1,2).imag();

    d.elem(i).elem().elem(2,0).real() -= tmp.elem(2,0).real();
    d.elem(i).elem().elem(2,0).imag() -= tmp.elem(2,0).imag();
    d.elem(i).elem().elem(2,1).real() -= tmp.elem(2,1).real();
    d.elem(i).elem().elem(2,1).imag() -= tmp.elem(2,1).imag();
    d.elem(i).elem().elem(2,2).real() -= tmp.elem(2,2).real();
    d.elem(i).elem().elem(2,2).imag() -= tmp.elem(2,2).imag();
  }
}


// Specialization to optimize the case   
//    LatticeColorMatrix[OrderedSubset] -= adj(LatticeColorMatrix) * adj(LatticeColorMatrix)
template<>
void evaluate(OLattice< TCol >& d, 
	      const OpSubtractAssign& op, 
	      const QDPExpr<BinaryNode<OpAdjMultiplyAdj, 
	                    UnaryNode<OpIdentity, Reference<QDPType< TCol, OLattice< TCol > > > >,
	                    UnaryNode<OpIdentity, Reference<QDPType< TCol, OLattice< TCol > > > > >,
	                    OLattice< TCol > >& rhs,
	      const OrderedSubset& s)
{
//  cout << "call single site QDP_M_meq_Ma_times_Ma" << endl;

  typedef OLattice< TCol >    C;

  const C& l = static_cast<const C&>(rhs.expression().left().child());
  const C& r = static_cast<const C&>(rhs.expression().right().child());

  PColorMatrix<RComplexFloat,3> tmp;

  for(int i=s.start(); i <= s.end(); ++i) 
  {
    _inline_sse_mult_su3_nn(r.elem(i).elem(),l.elem(i).elem(),tmp);

    // Take the adj(r*l) = adj(l)*adj(r)
    d.elem(i).elem().elem(0,0).real() -= tmp.elem(0,0).real();
    d.elem(i).elem().elem(0,0).imag() += tmp.elem(0,0).imag();
    d.elem(i).elem().elem(0,1).real() -= tmp.elem(1,0).real();
    d.elem(i).elem().elem(0,1).imag() += tmp.elem(1,0).imag();
    d.elem(i).elem().elem(0,2).real() -= tmp.elem(2,0).real();
    d.elem(i).elem().elem(0,2).imag() += tmp.elem(2,0).imag();

    d.elem(i).elem().elem(1,0).real() -= tmp.elem(0,1).real();
    d.elem(i).elem().elem(1,0).imag() += tmp.elem(0,1).imag();
    d.elem(i).elem().elem(1,1).real() -= tmp.elem(1,1).real();
    d.elem(i).elem().elem(1,1).imag() += tmp.elem(1,1).imag();
    d.elem(i).elem().elem(1,2).real() -= tmp.elem(2,1).real();
    d.elem(i).elem().elem(1,2).imag() += tmp.elem(2,1).imag();

    d.elem(i).elem().elem(2,0).real() -= tmp.elem(0,2).real();
    d.elem(i).elem().elem(2,0).imag() += tmp.elem(0,2).imag();
    d.elem(i).elem().elem(2,1).real() -= tmp.elem(1,2).real();
    d.elem(i).elem().elem(2,1).imag() += tmp.elem(1,2).imag();
    d.elem(i).elem().elem(2,2).real() -= tmp.elem(2,2).real();
    d.elem(i).elem().elem(2,2).imag() += tmp.elem(2,2).imag();
  }
}


//-------------------------------------------------------------------

// Specialization to optimize the case   
//    LatticeHalfFermion = LatticeColorMatrix * LatticeHalfFermion
// NOTE: let this be a subroutine to save space
template<>
void evaluate(OLattice< TVec2 >& d, 
	      const OpAssign& op, 
	      const QDPExpr<BinaryNode<OpMultiply, 
	                    Reference<QDPType< TCol, OLattice< TCol > > >, 
	                    Reference<QDPType< TVec2, OLattice< TVec2 > > > >,
	                    OLattice< TVec2 > >& rhs,
	      const OrderedSubset& s)
{
#if defined(QDP_SCALARSITE_DEBUG)
  cout << "specialized QDP_H_M_times_H" << endl;
#endif

  typedef OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >       C;
  typedef OLattice<PSpinVector<PColorVector<RComplexFloat, 3>, 2> > H;

  const C& l = static_cast<const C&>(rhs.expression().left());
  const H& r = static_cast<const H&>(rhs.expression().right());

  for(int i=s.start(); i <= s.end(); ++i) 
  {
#if 0
    // This form appears significantly slower than below
    _inline_sse_mult_su3_mat_hwvec(l.elem(i),
				   r.elem(i),
				   d.elem(i));
#else
    _inline_sse_mult_su3_mat_vec(l.elem(i).elem(),
				 r.elem(i).elem(0),
				 d.elem(i).elem(0));
    _inline_sse_mult_su3_mat_vec(l.elem(i).elem(),
				 r.elem(i).elem(1),
				 d.elem(i).elem(1));
#endif
  }
}
#endif


//-------------------------------------------------------------------
// GNUC vector type

//#define DEBUG_BLAS

// AXPY and AXMY routines
void vaxpy3(REAL32 *Out,REAL32 *scalep,REAL32 *InScale, REAL32 *Add,int n_3vec)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "SSE_TEST: vaxpy3" << endl;
#endif

//  int n_loops = n_3vec >> 2;   // only works on multiple of length 4 vectors
  int n_loops = n_3vec / 24;   // only works on multiple of length 24 vectors

//  register v4sf va = load_v4sf((float *)&a);
  v4sf vscalep = _mm_load_ss(scalep);
  asm("shufps\t$0,%0,%0" : "+x" (vscalep));

  for (; n_loops-- > 0; )
  {
    _mm_store_ps(Out+ 0, _mm_add_ps(_mm_mul_ps(vscalep, _mm_load_ps(InScale+ 0)), _mm_load_ps(Add+ 0)));
    _mm_store_ps(Out+ 4, _mm_add_ps(_mm_mul_ps(vscalep, _mm_load_ps(InScale+ 4)), _mm_load_ps(Add+ 4)));
    _mm_store_ps(Out+ 8, _mm_add_ps(_mm_mul_ps(vscalep, _mm_load_ps(InScale+ 8)), _mm_load_ps(Add+ 8)));
    _mm_store_ps(Out+12, _mm_add_ps(_mm_mul_ps(vscalep, _mm_load_ps(InScale+12)), _mm_load_ps(Add+12)));
    _mm_store_ps(Out+16, _mm_add_ps(_mm_mul_ps(vscalep, _mm_load_ps(InScale+16)), _mm_load_ps(Add+16)));
    _mm_store_ps(Out+20, _mm_add_ps(_mm_mul_ps(vscalep, _mm_load_ps(InScale+20)), _mm_load_ps(Add+20)));

    Out += 24; InScale += 24; Add += 24;
  }
}


void vaxmy3(REAL32 *Out,REAL32 *scalep,REAL32 *InScale, REAL32 *Sub,int n_3vec)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "SSE_TEST: vaxmy3" << endl;
#endif

//  int n_loops = n_3vec >> 2;   // only works on multiple of length 4 vectors
  int n_loops = n_3vec / 24;   // only works on multiple of length 24 vectors

//  register v4sf va = load_v4sf((float *)&a);
  v4sf vscalep = _mm_load_ss(scalep);
  asm("shufps\t$0,%0,%0" : "+x" (vscalep));

  for (; n_loops-- > 0; )
  {
    _mm_store_ps(Out+ 0, _mm_sub_ps(_mm_mul_ps(vscalep, _mm_load_ps(InScale+ 0)), _mm_load_ps(Sub+ 0)));
    _mm_store_ps(Out+ 4, _mm_sub_ps(_mm_mul_ps(vscalep, _mm_load_ps(InScale+ 4)), _mm_load_ps(Sub+ 4)));
    _mm_store_ps(Out+ 8, _mm_sub_ps(_mm_mul_ps(vscalep, _mm_load_ps(InScale+ 8)), _mm_load_ps(Sub+ 8)));
    _mm_store_ps(Out+12, _mm_sub_ps(_mm_mul_ps(vscalep, _mm_load_ps(InScale+12)), _mm_load_ps(Sub+12)));
    _mm_store_ps(Out+16, _mm_sub_ps(_mm_mul_ps(vscalep, _mm_load_ps(InScale+16)), _mm_load_ps(Sub+16)));
    _mm_store_ps(Out+20, _mm_sub_ps(_mm_mul_ps(vscalep, _mm_load_ps(InScale+20)), _mm_load_ps(Sub+20)));

    Out += 24; InScale += 24; Sub += 24;
  }
}


void vadd(REAL32 *Out, REAL32 *In1, REAL32 *In2, int n_3vec)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "SSE_TEST: vadd" << endl;
#endif

//  int n_loops = n_3vec >> 2;   // only works on multiple of length 4 vectors
  int n_loops = n_3vec / 24;   // only works on multiple of length 24 vectors

  for (; n_loops-- > 0; )
  {
    _mm_store_ps(Out+ 0, _mm_add_ps(_mm_load_ps(In1+ 0), _mm_load_ps(In2+ 0)));
    _mm_store_ps(Out+ 4, _mm_add_ps(_mm_load_ps(In1+ 4), _mm_load_ps(In2+ 4)));
    _mm_store_ps(Out+ 8, _mm_add_ps(_mm_load_ps(In1+ 8), _mm_load_ps(In2+ 8)));
    _mm_store_ps(Out+12, _mm_add_ps(_mm_load_ps(In1+12), _mm_load_ps(In2+12)));
    _mm_store_ps(Out+16, _mm_add_ps(_mm_load_ps(In1+16), _mm_load_ps(In2+16)));
    _mm_store_ps(Out+20, _mm_add_ps(_mm_load_ps(In1+20), _mm_load_ps(In2+20)));

    Out += 24; In1 += 24; In2 += 24;
  }
}


void vsub(REAL32 *Out, REAL32 *In1, REAL32 *In2, int n_3vec)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "SSE_TEST: vsub" << endl;
#endif

//  int n_loops = n_3vec >> 2;   // only works on multiple of length 4 vectors
  int n_loops = n_3vec / 24;   // only works on multiple of length 24 vectors

  for (; n_loops-- > 0; )
  {
    _mm_store_ps(Out+ 0, _mm_sub_ps(_mm_load_ps(In1+ 0), _mm_load_ps(In2+ 0)));
    _mm_store_ps(Out+ 4, _mm_sub_ps(_mm_load_ps(In1+ 4), _mm_load_ps(In2+ 4)));
    _mm_store_ps(Out+ 8, _mm_sub_ps(_mm_load_ps(In1+ 8), _mm_load_ps(In2+ 8)));
    _mm_store_ps(Out+12, _mm_sub_ps(_mm_load_ps(In1+12), _mm_load_ps(In2+12)));
    _mm_store_ps(Out+16, _mm_sub_ps(_mm_load_ps(In1+16), _mm_load_ps(In2+16)));
    _mm_store_ps(Out+20, _mm_sub_ps(_mm_load_ps(In1+20), _mm_load_ps(In2+20)));

    Out += 24; In1 += 24; In2 += 24;
  }
}

void vscal(REAL32 *Out, REAL32 *scalep, REAL32 *In, int n_3vec)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "SSE_TEST: vadd" << endl;
#endif

//  int n_loops = n_3vec >> 2;   // only works on multiple of length 4 vectors
  int n_loops = n_3vec / 24;   // only works on multiple of length 24 vectors

//  register v4sf va = load_v4sf((float *)&a);
  v4sf vscalep = _mm_load_ss(scalep);
  asm("shufps\t$0,%0,%0" : "+x" (vscalep));

  for (; n_loops-- > 0; )
  {
    _mm_store_ps(Out+ 0, _mm_mul_ps(vscalep, _mm_load_ps(In+ 0)));
    _mm_store_ps(Out+ 4, _mm_mul_ps(vscalep, _mm_load_ps(In+ 4)));
    _mm_store_ps(Out+ 8, _mm_mul_ps(vscalep, _mm_load_ps(In+ 8)));
    _mm_store_ps(Out+12, _mm_mul_ps(vscalep, _mm_load_ps(In+12)));
    _mm_store_ps(Out+16, _mm_mul_ps(vscalep, _mm_load_ps(In+16)));
    _mm_store_ps(Out+20, _mm_mul_ps(vscalep, _mm_load_ps(In+20)));

    Out += 24; In += 24;
  }
}  


void vaxpby3(REAL32 *Out, REAL32 *a, REAL32 *x, REAL32 *b, REAL32 *y, int n_3vec)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "SSE_TEST: vaxpby3: a*x+b*y" << endl;
#endif

//  int n_loops = n_3vec >> 2;   // only works on multiple of length 4 vectors
  int n_loops = n_3vec / 24;   // only works on multiple of length 24 vectors

//  register v4sf va = load_v4sf((float *)&a);
  v4sf va = _mm_load_ss(a);
  v4sf vb = _mm_load_ss(b);
  asm("shufps\t$0,%0,%0" : "+x" (va));
  asm("shufps\t$0,%0,%0" : "+x" (vb));

  for (; n_loops-- > 0; )
  {
    _mm_store_ps(Out+ 0, _mm_add_ps(_mm_mul_ps(va, _mm_load_ps(x+ 0)), _mm_mul_ps(vb, _mm_load_ps(y+ 0))));
    _mm_store_ps(Out+ 4, _mm_add_ps(_mm_mul_ps(va, _mm_load_ps(x+ 4)), _mm_mul_ps(vb, _mm_load_ps(y+ 4))));
    _mm_store_ps(Out+ 8, _mm_add_ps(_mm_mul_ps(va, _mm_load_ps(x+ 8)), _mm_mul_ps(vb, _mm_load_ps(y+ 8))));
    _mm_store_ps(Out+12, _mm_add_ps(_mm_mul_ps(va, _mm_load_ps(x+12)), _mm_mul_ps(vb, _mm_load_ps(y+12))));
    _mm_store_ps(Out+16, _mm_add_ps(_mm_mul_ps(va, _mm_load_ps(x+16)), _mm_mul_ps(vb, _mm_load_ps(y+16))));
    _mm_store_ps(Out+20, _mm_add_ps(_mm_mul_ps(va, _mm_load_ps(x+20)), _mm_mul_ps(vb, _mm_load_ps(y+20))));

    Out += 24; x += 24; y += 24;
  }
}


void vaxmby3(REAL32 *Out, REAL32 *a, REAL32 *x, REAL32 *b, REAL32 *y, int n_3vec)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "SSE_TEST: vaxmby3: a*x-b*y" << endl;
#endif

//  int n_loops = n_3vec >> 2;   // only works on multiple of length 4 vectors
  int n_loops = n_3vec / 24;   // only works on multiple of length 24 vectors

//  register v4sf va = load_v4sf((float *)&a);
  v4sf va = _mm_load_ss(a);
  v4sf vb = _mm_load_ss(b);
  asm("shufps\t$0,%0,%0" : "+x" (va));
  asm("shufps\t$0,%0,%0" : "+x" (vb));

  for (; n_loops-- > 0; )
  {
    _mm_store_ps(Out+ 0, _mm_sub_ps(_mm_mul_ps(va, _mm_load_ps(x+ 0)), _mm_mul_ps(vb, _mm_load_ps(y+ 0))));
    _mm_store_ps(Out+ 4, _mm_sub_ps(_mm_mul_ps(va, _mm_load_ps(x+ 4)), _mm_mul_ps(vb, _mm_load_ps(y+ 4))));
    _mm_store_ps(Out+ 8, _mm_sub_ps(_mm_mul_ps(va, _mm_load_ps(x+ 8)), _mm_mul_ps(vb, _mm_load_ps(y+ 8))));
    _mm_store_ps(Out+12, _mm_sub_ps(_mm_mul_ps(va, _mm_load_ps(x+12)), _mm_mul_ps(vb, _mm_load_ps(y+12))));
    _mm_store_ps(Out+16, _mm_sub_ps(_mm_mul_ps(va, _mm_load_ps(x+16)), _mm_mul_ps(vb, _mm_load_ps(y+16))));
    _mm_store_ps(Out+20, _mm_sub_ps(_mm_mul_ps(va, _mm_load_ps(x+20)), _mm_mul_ps(vb, _mm_load_ps(y+20))));

    Out += 24; x += 24; y += 24;
  }
}



void local_sumsq(REAL64 *Out, REAL32 *In, int n_3vec)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "SSE_TEST: local_sumsq" << endl;
#endif

//  int n_loops = n_3vec >> 2;   // only works on multiple of length 4 vectors
  int n_loops = n_3vec / 24;   // only works on multiple of length 24 vectors


  (*Out) = (REAL64)0;

  register SSEVec in0;
  register SSEVec in1;
  register SSEVec in2;
  register SSEVec in3;
  register SSEVec sq;
  register SSEVec sum;

  n_loops--;  // We pull out the first 4

  sum.floats[0] = sum.floats[1] = sum.floats[2] = sum.floats[3] = 0;

  in0.vector = _mm_load_ps(In);
   sq.vector = _mm_mul_ps(in0.vector,in0.vector);
  sum.vector = _mm_add_ps(sum.vector, sq.vector);
  
  in1.vector = _mm_load_ps(In+4);
  sq.vector = _mm_mul_ps(in1.vector,in1.vector);
  sum.vector = _mm_add_ps(sum.vector, sq.vector);
  
  in2.vector = _mm_load_ps(In+8);
  sq.vector = _mm_mul_ps(in2.vector,in2.vector);
  sum.vector = _mm_add_ps(sum.vector, sq.vector);

  in3.vector = _mm_load_ps(In+12);
  sq.vector = _mm_mul_ps(in3.vector,in3.vector);
  sum.vector = _mm_add_ps(sum.vector, sq.vector);

  *Out += (double)sum.floats[0] 
    + (double)sum.floats[1] 
    + (double)sum.floats[2]
    + (double)sum.floats[3];
  
  In += 16;

  for (; n_loops-- > 0; ) {
    
    // Initialise the sum
    sum.floats[0] = sum.floats[1] = sum.floats[2] = sum.floats[3] = 0;

    // Do 24
    
    in0.vector = _mm_load_ps(In);
    sq.vector = _mm_mul_ps(in0.vector,in0.vector);
    sum.vector = _mm_add_ps(sum.vector, sq.vector);

    in1.vector = _mm_load_ps(In+4);
    sq.vector = _mm_mul_ps(in1.vector,in1.vector);
    sum.vector = _mm_add_ps(sum.vector, sq.vector);
      
    in2.vector = _mm_load_ps(In+8);
    sq.vector = _mm_mul_ps(in2.vector,in2.vector);
    sum.vector = _mm_add_ps(sum.vector, sq.vector);
    
    in3.vector = _mm_load_ps(In+12);
    sq.vector = _mm_mul_ps(in3.vector,in3.vector);
    sum.vector = _mm_add_ps(sum.vector, sq.vector);
    
    in0.vector = _mm_load_ps(In+16);
    sq.vector = _mm_mul_ps(in0.vector,in0.vector);
    sum.vector = _mm_add_ps(sum.vector, sq.vector);
    
    in1.vector = _mm_load_ps(In+20);
    sq.vector = _mm_mul_ps(in1.vector,in1.vector);
    sum.vector = _mm_add_ps(sum.vector, sq.vector);

  *Out += (double)sum.floats[0] 
    + (double)sum.floats[1] 
    + (double)sum.floats[2]
    + (double)sum.floats[3];

    In +=24;

  }

  sum.floats[0] = sum.floats[1] = sum.floats[2] = sum.floats[3] = 0;

  in2.vector = _mm_load_ps(In);
  sq.vector = _mm_mul_ps(in2.vector,in2.vector);
  sum.vector = _mm_add_ps(sum.vector, sq.vector);

  in3.vector = _mm_load_ps(In+4);
  sq.vector = _mm_mul_ps(in3.vector,in3.vector);
  sum.vector = _mm_add_ps(sum.vector, sq.vector);

  *Out += (double)sum.floats[0] 
    + (double)sum.floats[1] 
    + (double)sum.floats[2]
    + (double)sum.floats[3];

}

void local_sumsq2(REAL64 *Out, REAL32 *In, int n_3vec)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "SSE_TEST: local_sumsq" << endl;
#endif

//  int n_loops = n_3vec >> 2;   // only works on multiple of length 4 vectors
  int n_loops = n_3vec / 24;   // only works on multiple of length 24 vectors


  (*Out) = (REAL64)0;

  register SSEVec in;
  register SSEVec sq;
  register SSEVec sum;


  sum.floats[0] = sum.floats[1] = sum.floats[2] = sum.floats[3] = 0;
  for (; n_loops-- > 0; ) {
    

    // Do 24
    
    in.vector = _mm_load_ps(In);
    sq.vector = _mm_mul_ps(in.vector,in.vector);
    sum.vector = _mm_add_ps(sum.vector, sq.vector);

    in.vector = _mm_load_ps(In+4);
    sq.vector = _mm_mul_ps(in.vector,in.vector);
    sum.vector = _mm_add_ps(sum.vector, sq.vector);
      
    in.vector = _mm_load_ps(In+8);
    sq.vector = _mm_mul_ps(in.vector,in.vector);
    sum.vector = _mm_add_ps(sum.vector, sq.vector);
    
    in.vector = _mm_load_ps(In+12);
    sq.vector = _mm_mul_ps(in.vector,in.vector);
    sum.vector = _mm_add_ps(sum.vector, sq.vector);
    
    in.vector = _mm_load_ps(In+16);
    sq.vector = _mm_mul_ps(in.vector,in.vector);
    sum.vector = _mm_add_ps(sum.vector, sq.vector);
    
    in.vector = _mm_load_ps(In+20);
    sq.vector = _mm_mul_ps(in.vector,in.vector);
    sum.vector = _mm_add_ps(sum.vector, sq.vector);

    In +=24;

  }

  *Out += (double)sum.floats[0] 
    + (double)sum.floats[1] 
    + (double)sum.floats[2]
    + (double)sum.floats[3];

}


// (Vector) out = (Scalar) (*scalep) * (Vector) InScale + (*scalep2)*(Vector) P{+} Add
void axpbyz(REAL32 *Out,REAL32 *scalep,REAL32 *InScale, REAL32 *scalep2, REAL32 *Add,int n_4vec)
{
  // GNUC vector type
  

  // Load Vscalep
  v4sf vscalep = _mm_load_ss(scalep);
  v4sf vscalep2 = _mm_load_ss(scalep2);

  asm("shufps\t$0,%0,%0" : "+x" (vscalep));
  asm("shufps\t$0,%0,%0" : "+x" (vscalep2));

  for(int i=0; i < n_4vec; i++) {

    // Spin Component 0: z0r, z0i, z1r, z1i
    _mm_store_ps(Out+ 0, _mm_add_ps(_mm_mul_ps(vscalep, _mm_load_ps(InScale+ 0)), _mm_mul_ps(vscalep2,_mm_load_ps(Add+ 0))));
     
    // Spin Component 0: z2r, z2i, SpinComponent 1: z0r, z0i
    _mm_store_ps(Out+ 4, _mm_add_ps(_mm_mul_ps(vscalep, _mm_load_ps(InScale+ 4)), _mm_mul_ps(vscalep2,_mm_load_ps(Add+ 4))));

    // Spin Component 1: z1r, z1i, z2r, z2i
    _mm_store_ps(Out+ 8, _mm_add_ps(_mm_mul_ps(vscalep, _mm_load_ps(InScale+ 8)), _mm_mul_ps(vscalep2,_mm_load_ps(Add+ 8))));

    // Spin Component 2: z0r, z0i, z1r, z1i
    _mm_store_ps(Out+ 12, _mm_add_ps(_mm_mul_ps(vscalep, _mm_load_ps(InScale+ 12)), _mm_mul_ps(vscalep2,_mm_load_ps(Add+ 12))));

    // Spin Component 2: z2r, z2i, z0r, z0
    _mm_store_ps(Out+ 16, _mm_add_ps(_mm_mul_ps(vscalep, _mm_load_ps(InScale+ 16)), _mm_mul_ps(vscalep2,_mm_load_ps(Add+ 16))));


    // Spin Component 3: z1r, z1i, z2r, z2
    _mm_store_ps(Out+ 20, _mm_add_ps(_mm_mul_ps(vscalep, _mm_load_ps(InScale+ 20)), _mm_mul_ps(vscalep2,_mm_load_ps(Add+ 20))));

    // Update offsets
    Out += 24; InScale += 24; Add += 24;
  }

}


// (Vector) out = (Scalar) (*scalep) * (Vector) InScale - (*scalep2)*(Vector) P{+} Add
void axmbyz(REAL32 *Out,REAL32 *scalep,REAL32 *InScale, REAL32 *scalep2, REAL32 *Add,int n_4vec)
{
  // GNUC vector type
  

  // Load Vscalep
  v4sf vscalep = _mm_load_ss(scalep);
  v4sf vscalep2 = _mm_load_ss(scalep2);

  asm("shufps\t$0,%0,%0" : "+x" (vscalep));
  asm("shufps\t$0,%0,%0" : "+x" (vscalep2));

  for(int i=0; i < n_4vec; i++) {

    // Spin Component 0: z0r, z0i, z1r, z1i
    _mm_store_ps(Out+ 0, _mm_sub_ps(_mm_mul_ps(vscalep, _mm_load_ps(InScale+ 0)), _mm_mul_ps(vscalep2,_mm_load_ps(Add+ 0))));
     
    _mm_store_ps(Out+ 4, _mm_sub_ps(_mm_mul_ps(vscalep, _mm_load_ps(InScale+ 4)), _mm_mul_ps(vscalep2,_mm_load_ps(Add+ 4))));

    _mm_store_ps(Out+ 8, _mm_sub_ps(_mm_mul_ps(vscalep, _mm_load_ps(InScale+ 8)), _mm_mul_ps(vscalep2,_mm_load_ps(Add+ 8))));

    _mm_store_ps(Out+ 12, _mm_sub_ps(_mm_mul_ps(vscalep, _mm_load_ps(InScale+ 12)), _mm_mul_ps(vscalep2,_mm_load_ps(Add+ 12))));

    _mm_store_ps(Out+ 16, _mm_sub_ps(_mm_mul_ps(vscalep, _mm_load_ps(InScale+ 16)), _mm_mul_ps(vscalep2,_mm_load_ps(Add+ 16))));

    _mm_store_ps(Out+ 20, _mm_sub_ps(_mm_mul_ps(vscalep, _mm_load_ps(InScale+ 20)), _mm_mul_ps(vscalep2,_mm_load_ps(Add+ 20))));


    // Update offsets
    Out += 24; InScale += 24; Add += 24;
  }

}



#endif // BASE PRECISION==32

QDP_END_NAMESPACE();

#endif  // defined(__GNUC__)
