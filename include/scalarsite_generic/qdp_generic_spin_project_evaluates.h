#ifndef QDP_GENERIC_SPIN_PROJECT_EVALUTATES_H
#define QDP_GENERIC_SPIN_PROJECT_EVALUTATES_H

using namespace QDP;
QDP_BEGIN_NAMESPACE(QDP);

// Typedefs
typedef PSpinVector< PColorVector< RComplex<REAL>, Nc>, Ns>>1 > HVec;
typedef PSpinVector< PColorVector< RComplex<REAL>, Nc>, 4> FVec;

// Four spinor (Ns * Nc * Ncomplex ) Ncomplex fastest
typedef REAL SpinColFull[4][3][2];

// Half spinor (Ns/2 * Nc * Ncomplex ) Ncomplex fastest
typedef REAL SpinColHalf[2][3][2];
// d = SpinProjectDir0Plus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinProjectDir0Plus, 
	      Reference< QDPType<FVec,OLattice< FVec > > > >,
	      OLattice< HVec > > &rhs,
	      const OrderedSubset& s) 
{

  //  Get at pointer for 4 vec
  const OLattice< FVec >& a = static_cast<const OLattice< FVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());
  unsigned int n_vec=s.end() - s.start()+1;

  inlineSpinProjDir0Plus(aptr, bptr, n_vec);


}

// d = SpinProjectDir1Plus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinProjectDir1Plus, 
	      Reference< QDPType<FVec,OLattice< FVec > > > >,
	      OLattice< HVec > > &rhs,
	      const OrderedSubset& s) 
{

  
  const OLattice< FVec >& a = static_cast<const OLattice< FVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned int n_vec=s.end() - s.start()+1;
  inlineSpinProjDir1Plus(aptr, bptr, n_vec);

}

// d = SpinProjectDir2Plus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinProjectDir2Plus, 
	      Reference< QDPType<FVec,OLattice< FVec > > > >,
	      OLattice< HVec > > &rhs,
	      const OrderedSubset& s) 
{
  
  const OLattice< FVec >& a = static_cast<const OLattice< FVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned int n_vec=s.end() - s.start()+1;
  inlineSpinProjDir2Plus(aptr, bptr, n_vec);

}

// d = SpinProjectDir3Plus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinProjectDir3Plus, 
	      Reference< QDPType<FVec,OLattice< FVec > > > >,
	      OLattice< HVec > > &rhs,
	      const OrderedSubset& s) 
{
  const OLattice< FVec >& a = static_cast<const OLattice< FVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned int n_vec=s.end() - s.start()+1;
  inlineSpinProjDir3Plus(aptr, bptr, n_vec);
}

// d = SpinProjectDir0Minus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinProjectDir0Minus, 
	      Reference< QDPType<FVec,OLattice< FVec > > > >,
	      OLattice< HVec > > &rhs,
	      const OrderedSubset& s) 
{

  //  Get at pointer for 4 vec
  const OLattice< FVec >& a = static_cast<const OLattice< FVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned int n_vec=s.end() - s.start()+1;
  inlineSpinProjDir0Minus(aptr, bptr, n_vec);
}

// d = SpinProjectDir1Minus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinProjectDir1Minus, 
	      Reference< QDPType<FVec,OLattice< FVec > > > >,
	      OLattice< HVec > > &rhs,
	      const OrderedSubset& s) 
{

  const OLattice< FVec >& a = static_cast<const OLattice< FVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned int n_vec=s.end() - s.start()+1;
  inlineSpinProjDir1Minus(aptr, bptr, n_vec);

}

// d = SpinProjectDir2Minus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinProjectDir2Minus, 
	      Reference< QDPType<FVec,OLattice< FVec > > > >,
	      OLattice< HVec > > &rhs,
	      const OrderedSubset& s) 
{

  
  const OLattice< FVec >& a = static_cast<const OLattice< FVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned int n_vec=s.end() - s.start()+1;
  inlineSpinProjDir2Minus(aptr, bptr, n_vec);


}

// d = SpinProjectDir3Minus(Vec);
template<>
inline
void evaluate(OLattice< HVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinProjectDir3Minus, 
	      Reference< QDPType<FVec,OLattice< FVec > > > >,
	      OLattice< HVec > > &rhs,
	      const OrderedSubset& s) 
{
  const OLattice< FVec >& a = static_cast<const OLattice< FVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned int n_vec=s.end() - s.start()+1;
  inlineSpinProjDir3Minus(aptr, bptr, n_vec);
}




// d = SpinReconstructDir0Plus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir0Plus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{

  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned int n_vec=s.end() - s.start()+1;
  inlineSpinReconDir0Plus(aptr, bptr, n_vec);


  
}

// d = SpinReconstructDir1Plus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir1Plus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{

  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned int n_vec=s.end() - s.start()+1;
  inlineSpinReconDir1Plus(aptr, bptr, n_vec);

}

// d = SpinReconstructDir2Plus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir2Plus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{


  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned int n_vec=s.end() - s.start()+1;
  inlineSpinReconDir2Plus(aptr, bptr, n_vec);

}

// d = SpinReconstructDir3Plus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir3Plus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{

  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned int n_vec=s.end() - s.start()+1;
  inlineSpinReconDir3Plus(aptr, bptr, n_vec);

}

// d = SpinReconstructDir0Minus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir0Minus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{

  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());


  unsigned int n_vec=s.end() - s.start()+1;
  inlineSpinReconDir0Minus(aptr, bptr, n_vec);

  
}

// d = SpinReconstructDir1Minus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir1Minus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{

  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned int n_vec=s.end() - s.start()+1;
  inlineSpinReconDir1Minus(aptr, bptr, n_vec);
}

// d = SpinReconstructDir2Minus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir2Minus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{


  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned int n_vec=s.end() - s.start()+1;
  inlineSpinReconDir2Minus(aptr, bptr, n_vec);
}

// d = SpinReconstructDir3Minus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir3Minus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{

  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned int n_vec=s.end() - s.start()+1;
  inlineSpinReconDir3Minus(aptr, bptr, n_vec);
}



// d += SpinReconstructDir0Plus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAddAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir0Plus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{


  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned int n_vec=s.end() - s.start()+1;
  inlineAddSpinReconDir0Plus(aptr, bptr, n_vec);

}

// d += SpinReconstructDir1Plus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAddAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir1Plus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{

  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned int n_vec=s.end() - s.start()+1;
  inlineAddSpinReconDir1Plus(aptr, bptr, n_vec);

}

// d += SpinReconstructDir2Plus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAddAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir2Plus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{
  
  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned int n_vec=s.end() - s.start()+1;
  inlineAddSpinReconDir2Plus(aptr, bptr, n_vec);

}

// d += SpinReconstructDir3Plus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAddAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir3Plus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{

  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned int n_vec=s.end() - s.start()+1;
  inlineAddSpinReconDir3Plus(aptr, bptr, n_vec);

}

// d += SpinReconstructDir0Minus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAddAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir0Minus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{


  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned int n_vec=s.end() - s.start()+1;
  inlineAddSpinReconDir0Minus(aptr, bptr, n_vec);

  
}

// d += SpinReconstructDir1Minus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAddAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir1Minus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{

  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned int n_vec=s.end() - s.start()+1;
  inlineAddSpinReconDir1Minus(aptr, bptr, n_vec);

}

// d += SpinReconstructDir2Minus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAddAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir2Minus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{


  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned int n_vec=s.end() - s.start()+1;
  inlineAddSpinReconDir2Minus(aptr, bptr, n_vec);
}

// d += SpinReconstructDir3Minus(Vec);
template<>
inline
void evaluate(OLattice< FVec >& b,
              const OpAddAssign& op,
              const QDPExpr<                             
	              UnaryNode< FnSpinReconstructDir3Minus, 
	      Reference< QDPType<HVec,OLattice< HVec > > > >,
	      OLattice< FVec > > &rhs,
	      const OrderedSubset& s) 
{


  const OLattice< HVec >& a = static_cast<const OLattice< HVec > &>(rhs.expression().child());

  REAL *aptr =(REAL *)&(a.elem(s.start()).elem(0).elem(0).real());
  REAL *bptr =(REAL *)&(b.elem(s.start()).elem(0).elem(0).real());

  unsigned int n_vec=s.end() - s.start()+1;
  inlineAddSpinReconDir3Minus(aptr, bptr, n_vec);
}

QDP_END_NAMESPACE();

#endif
