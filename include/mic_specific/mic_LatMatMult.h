//===================== MIC ==========================

#ifdef __MIC

#ifndef INCL_mic_LatMatMult
#define INCL_mic_LatMatMult

#include "mic_MatMult.h"

namespace QDP {

typedef RComplex<REAL64>  RComplexD;
typedef PColorMatrix<RComplexD, 3> MatSU3;
typedef PScalar<MatSU3> ScalMatSU3;
typedef LatticeColorMatrixD3  LatSU3;

// Specialization to optimize the case
//    LatticeColorMatrix = LatticeColorMatrix * LatticeColorMatrix
// d: destinationa
// rhs.left() and rhs.right()
inline void mic_LatMatMult(void mic_func(MatSU3& w, const MatSU3& u, const MatSU3& v),
	LatSU3& w, const LatSU3& u, const LatSU3& v, const Subset& s)
{
	const int* tab = s.siteTable().slice();

#pragma omp parallel for
	 for(int j=0; j < s.numSiteTable(); ++j)
	 {
		 int i = tab[j];
		 mic_func(w.elem(i).elem(), u.elem(i).elem(), v.elem(i).elem());
	 }
}

template<>
inline
void evaluate(OLattice<ScalMatSU3>& d, 		// destination
	      const OpAssign& op,
	      const QDPExpr<BinaryNode<OpMultiply,
	      Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > >,
	      Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > >,
	      OLattice<ScalMatSU3> >& rhs,	// right hand side
	      const Subset& s)
{
//QDPIO::cout << "specialized QDP_M=M*M" << "  subset=" << s.end()-s.start()+1 << endl;
	mic_LatMatMult(mic_MatMult, d,
		static_cast<const LatSU3&>(rhs.expression().left()),
		static_cast<const LatSU3&>(rhs.expression().right()), s);
}

template<>
inline
void evaluate(OLattice<ScalMatSU3>& d, 		// destination
	      const OpAssign& op,
	      const QDPExpr<BinaryNode<OpAdjMultiply,
	      UnaryNode<OpIdentity, Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > >,
	      Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > >,
	      OLattice<ScalMatSU3> >& rhs,	// right hand side
	      const Subset& s)
{
//QDPIO::cout << "specialized QDP_M=adjM*M" << "  subset=" << s.end()-s.start()+1 << endl;
	mic_LatMatMult(mic_MatAdjMult, d,
		static_cast<const LatSU3&>(rhs.expression().left().child()),
		static_cast<const LatSU3&>(rhs.expression().right()), s);
}

template<>
inline
void evaluate(OLattice<ScalMatSU3>& d, 		// destination
	      const OpAssign& op,
	      const QDPExpr<BinaryNode<OpMultiplyAdj,
	      Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > >,
	      UnaryNode<OpIdentity, Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > > >,
	      OLattice<ScalMatSU3> >& rhs,	// right hand side
	      const Subset& s)
{
//QDPIO::cout << "specialized QDP_M=M*adjM" << "  subset=" << s.end()-s.start()+1 << endl;
	mic_LatMatMult(mic_MatMultAdj, d,
		static_cast<const LatSU3&>(rhs.expression().left()),
		static_cast<const LatSU3&>(rhs.expression().right().child()), s);
}


// Specialization to optimize the case
//    LatticeColorMatrix = LatticeColorMatrix * LatticeColorMatrix
// d: destinationa
// rhs.left() and rhs.right()
template<>
inline
void evaluate(OLattice<ScalMatSU3>& d, 		// destination
	      const OpAssign& op,
	      const QDPExpr<BinaryNode<OpAdjMultiplyAdj,
	      UnaryNode<OpIdentity, Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > >,
	      UnaryNode<OpIdentity, Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > > >,
	      OLattice<ScalMatSU3> >& rhs,	// right hand side
	      const Subset& s)
{
//QDPIO::cout << "specialized QDP_M=adjM*adjM" << "  subset=" << s.end()-s.start()+1 << endl;
	mic_LatMatMult(mic_MatAdjMultAdj, d,
		static_cast<const LatSU3&>(rhs.expression().left().child()),
		static_cast<const LatSU3&>(rhs.expression().right().child()), s);
}

//===============================AddAssign ==================================


// ADD ASSIGN
template<>
inline
void evaluate(OLattice<ScalMatSU3>& d, 		// destination
	      const OpAddAssign& op,
	      const QDPExpr<BinaryNode<OpMultiply,
	      Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > >,
	      Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > >,
	      OLattice<ScalMatSU3> >& rhs,	// right hand side
	      const Subset& s)
{
//QDPIO::cout << "specialized QDP_M+=M*M" << "  subset=" << s.end()-s.start()+1 << endl;
	mic_LatMatMult(mic_AddMatMult, d,
		static_cast<const LatSU3&>(rhs.expression().left()),
		static_cast<const LatSU3&>(rhs.expression().right()), s);
}

template<>
inline
void evaluate(OLattice<ScalMatSU3>& d, 		// destination
	      const OpAddAssign& op,
	      const QDPExpr<BinaryNode<OpAdjMultiply,
	      UnaryNode<OpIdentity, Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > >,
	      Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > >,
	      OLattice<ScalMatSU3> >& rhs,	// right hand side
	      const Subset& s)
{
//QDPIO::cout << "specialized QDP_M+=adjM*M" << "  subset=" << s.end()-s.start()+1 << endl;
	mic_LatMatMult(mic_AddMatAdjMult, d,
		static_cast<const LatSU3&>(rhs.expression().left().child()),
		static_cast<const LatSU3&>(rhs.expression().right()), s);
}

template<>
inline
void evaluate(OLattice<ScalMatSU3>& d, 		// destination
	      const OpAddAssign& op,
	      const QDPExpr<BinaryNode<OpMultiplyAdj,
	      Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > >,
	      UnaryNode<OpIdentity, Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > > >,
	      OLattice<ScalMatSU3> >& rhs,	// right hand side
	      const Subset& s)
{
//QDPIO::cout << "specialized QDP_M+=M*adjM" << "  subset=" << s.end()-s.start()+1 << endl;
	mic_LatMatMult(mic_AddMatMultAdj, d,
		static_cast<const LatSU3&>(rhs.expression().left()),
		static_cast<const LatSU3&>(rhs.expression().right().child()), s);
}

template<>
inline
void evaluate(OLattice<ScalMatSU3>& d, 		// destination
	      const OpAddAssign& op,
	      const QDPExpr<BinaryNode<OpAdjMultiplyAdj,
	      UnaryNode<OpIdentity, Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > >,
	      UnaryNode<OpIdentity, Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > > >,
	      OLattice<ScalMatSU3> >& rhs,	// right hand side
	      const Subset& s)
{
//QDPIO::cout << "specialized QDP_M+=adjM*adjM" << "  subset=" << s.end()-s.start()+1 << endl;
	mic_LatMatMult(mic_AddMatAdjMultAdj, d,
		static_cast<const LatSU3&>(rhs.expression().left().child()),
		static_cast<const LatSU3&>(rhs.expression().right().child()), s);
}

#ifdef BLABLA
template<>
inline
void evaluate(OLattice<ScalMatSU3>& d, 		// destination
	      const OpAssign& op,
	      const QDPExpr<BinaryNode<OpMultiply,
	      Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > >,
	      Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > >,
	      OLattice<ScalMatSU3> >& rhs,	// right hand side
	      const Subset& s)
{
//QDPIO::cout << "specialized QDP_M=M*M" << "  subset=" << s.end()-s.start()+1 << endl;

	const int* tab = s.siteTable().slice();

  typedef OLattice<ScalMatSU3>  LatSU3;

  const LatSU3& u = static_cast<const LatSU3&>(rhs.expression().left());
  const LatSU3& v = static_cast<const LatSU3&>(rhs.expression().right());

#pragma omp parallel for
	 for(int j=0; j < s.numSiteTable(); ++j)
	 {
		 int i = tab[j];
		 mic_MatMult(d.elem(i).elem(), u.elem(i).elem(), v.elem(i).elem());
	 }
}

// Specialization to optimize the case
//    LatticeColorMatrix = LatticeColorMatrix * LatticeColorMatrix
// d: destinationa
// rhs.left() and rhs.right()
template<>
inline
void evaluate(OLattice<ScalMatSU3>& d, 		// destination
	      const OpAssign& op,
	      const QDPExpr<BinaryNode<OpAdjMultiply,
	      UnaryNode<OpIdentity, Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > >,
	      Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > >,
	      OLattice<ScalMatSU3> >& rhs,	// right hand side
	      const Subset& s)
{
//QDPIO::cout << "specialized QDP_M=adjM*M" << "  subset=" << s.end()-s.start()+1 << endl;

	const int* tab = s.siteTable().slice();

  typedef OLattice<ScalMatSU3>  LatSU3;

  const LatSU3& u = static_cast<const LatSU3&>(rhs.expression().left().child());
  const LatSU3& v = static_cast<const LatSU3&>(rhs.expression().right());

#pragma omp parallel for
	 for(int j=0; j < s.numSiteTable(); ++j)
	 {
		 int i = tab[j];
		 mic_MatAdjMult(d.elem(i).elem(), u.elem(i).elem(), v.elem(i).elem());
	 }
}

// Specialization to optimize the case
//    LatticeColorMatrix = LatticeColorMatrix * LatticeColorMatrix
// d: destinationa
// rhs.left() and rhs.right()
template<>
inline
void evaluate(OLattice<ScalMatSU3>& d, 		// destination
	      const OpAssign& op,
	      const QDPExpr<BinaryNode<OpMultiplyAdj,
	      Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > >,
	      UnaryNode<OpIdentity, Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > > >,
	      OLattice<ScalMatSU3> >& rhs,	// right hand side
	      const Subset& s)
{
//QDPIO::cout << "specialized QDP_M=M*adjM" << "  subset=" << s.end()-s.start()+1 << endl;

	const int* tab = s.siteTable().slice();

  typedef OLattice<ScalMatSU3>  LatSU3;

  const LatSU3& u = static_cast<const LatSU3&>(rhs.expression().left());
  const LatSU3& v = static_cast<const LatSU3&>(rhs.expression().right().child());

#pragma omp parallel for
	 for(int j=0; j < s.numSiteTable(); ++j)
	 {
		 int i = tab[j];
		 mic_MatMultAdj(d.elem(i).elem(), u.elem(i).elem(), v.elem(i).elem());
	 }
}

// Specialization to optimize the case
//    LatticeColorMatrix = LatticeColorMatrix * LatticeColorMatrix
// d: destinationa
// rhs.left() and rhs.right()
template<>
inline
void evaluate(OLattice<ScalMatSU3>& d, 		// destination
	      const OpAssign& op,
	      const QDPExpr<BinaryNode<OpAdjMultiplyAdj,
	      UnaryNode<OpIdentity, Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > >,
	      UnaryNode<OpIdentity, Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > > >,
	      OLattice<ScalMatSU3> >& rhs,	// right hand side
	      const Subset& s)
{
//QDPIO::cout << "specialized QDP_M=adjM*adjM" << "  subset=" << s.end()-s.start()+1 << endl;

	const int* tab = s.siteTable().slice();

  typedef OLattice<ScalMatSU3>  LatSU3;

  const LatSU3& u = static_cast<const LatSU3&>(rhs.expression().left().child());
  const LatSU3& v = static_cast<const LatSU3&>(rhs.expression().right().child());

#pragma omp parallel for
	 for(int j=0; j < s.numSiteTable(); ++j)
	 {
		 int i = tab[j];
		 mic_MatAdjMultAdj(d.elem(i).elem(), u.elem(i).elem(), v.elem(i).elem());
	 }
}

//===============================AddAssign ==================================


// ADD ASSIGN
template<>
inline
void evaluate(OLattice<ScalMatSU3>& d, 		// destination
	      const OpAddAssign& op,
	      const QDPExpr<BinaryNode<OpMultiply,
	      Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > >,
	      Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > >,
	      OLattice<ScalMatSU3> >& rhs,	// right hand side
	      const Subset& s)
{
//QDPIO::cout << "specialized QDP_M+=M*M" << "  subset=" << s.end()-s.start()+1 << endl;

	const int* tab = s.siteTable().slice();

  typedef OLattice<ScalMatSU3>  LatSU3;

  const LatSU3& u = static_cast<const LatSU3&>(rhs.expression().left());
  const LatSU3& v = static_cast<const LatSU3&>(rhs.expression().right());

#pragma omp parallel for
	 for(int j=0; j < s.numSiteTable(); ++j)
	 {
		 int i = tab[j];
		 mic_AddMatMult(d.elem(i).elem(), u.elem(i).elem(), v.elem(i).elem());
	 }
}

template<>
inline
void evaluate(OLattice<ScalMatSU3>& d, 		// destination
	      const OpAddAssign& op,
	      const QDPExpr<BinaryNode<OpAdjMultiply,
	      UnaryNode<OpIdentity, Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > >,
	      Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > >,
	      OLattice<ScalMatSU3> >& rhs,	// right hand side
	      const Subset& s)
{
//QDPIO::cout << "specialized QDP_M+=adjM*M" << "  subset=" << s.end()-s.start()+1 << endl;

	const int* tab = s.siteTable().slice();

  typedef OLattice<ScalMatSU3>  LatSU3;

  const LatSU3& u = static_cast<const LatSU3&>(rhs.expression().left().child());
  const LatSU3& v = static_cast<const LatSU3&>(rhs.expression().right());

#pragma omp parallel for
	 for(int j=0; j < s.numSiteTable(); ++j)
	 {
		 int i = tab[j];
		 mic_AddMatAdjMult(d.elem(i).elem(), u.elem(i).elem(), v.elem(i).elem());
	 }
}

template<>
inline
void evaluate(OLattice<ScalMatSU3>& d, 		// destination
	      const OpAddAssign& op,
	      const QDPExpr<BinaryNode<OpMultiplyAdj,
	      Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > >,
	      UnaryNode<OpIdentity, Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > > >,
	      OLattice<ScalMatSU3> >& rhs,	// right hand side
	      const Subset& s)
{
//QDPIO::cout << "specialized QDP_M+=M*adjM" << "  subset=" << s.end()-s.start()+1 << endl;

	const int* tab = s.siteTable().slice();

  typedef OLattice<ScalMatSU3>  LatSU3;

  const LatSU3& u = static_cast<const LatSU3&>(rhs.expression().left());
  const LatSU3& v = static_cast<const LatSU3&>(rhs.expression().right().child());

#pragma omp parallel for
	 for(int j=0; j < s.numSiteTable(); ++j)
	 {
		 int i = tab[j];
		 mic_AddMatMultAdj(d.elem(i).elem(), u.elem(i).elem(), v.elem(i).elem());
	 }
}

template<>
inline
void evaluate(OLattice<ScalMatSU3>& d, 		// destination
	      const OpAddAssign& op,
	      const QDPExpr<BinaryNode<OpAdjMultiplyAdj,
	      UnaryNode<OpIdentity, Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > >,
	      UnaryNode<OpIdentity, Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > > >,
	      OLattice<ScalMatSU3> >& rhs,	// right hand side
	      const Subset& s)
{
//QDPIO::cout << "specialized QDP_M+=adjM*adjM" << "  subset=" << s.end()-s.start()+1 << endl;

	const int* tab = s.siteTable().slice();

  typedef OLattice<ScalMatSU3>  LatSU3;

  const LatSU3& u = static_cast<const LatSU3&>(rhs.expression().left().child());
  const LatSU3& v = static_cast<const LatSU3&>(rhs.expression().right().child());

#pragma omp parallel for
	 for(int j=0; j < s.numSiteTable(); ++j)
	 {
		 int i = tab[j];
		 mic_AddMatAdjMultAdj(d.elem(i).elem(), u.elem(i).elem(), v.elem(i).elem());
	 }
}

//===
#ifdef MORE
// MULTIPLY ASSIGN
template<>
inline
void evaluate(OLattice<ScalMatSU3>& d, 		// destination
	      const OpMultiplyAssign& op,
	      const QDPExpr<UnaryNode<OpIdentity, Reference<QDPType<ScalMatSU3,
	      OLattice<ScalMatSU3> > > > >& rhs,	// right hand side
	      const Subset& s)
{
//QDPIO::cout << "specialized QDP_M+=M*M" << "  subset=" << s.end()-s.start()+1 << endl;

	const int* tab = s.siteTable().slice();

  typedef OLattice<ScalMatSU3>  LatSU3;

//  const LatSU3& u = static_cast<const LatSU3&>(rhs.expression().left());
  const LatSU3& v = static_cast<const LatSU3&>(rhs.expression().right());

#pragma omp parallel for
	 for(int j=0; j < s.numSiteTable(); ++j)
	 {
		 int i = tab[j];
		 MatSU3 u = d.elem(i).elem();
		 mic_MatMult(d.elem(i).elem(), u.elem(i).elem(), v.elem(i).elem());
	 }
}
#endif // MORE
//===

#endif // BLABLA

}		// namespace QDP

#endif // INCL_mic_LatMatMult
#endif	// __MIC
