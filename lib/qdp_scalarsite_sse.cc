// $Id: qdp_scalarsite_sse.cc,v 1.10 2004-05-09 11:54:44 bjoo Exp $

/*! @file
 * @brief Intel SSE optimizations
 * 
 * SSE optimizations of basic operations
 */


#include "qdp.h"

// These SSE asm instructions are only supported under GCC/G++
#if defined(__GNUC__)

QDP_BEGIN_NAMESPACE(QDP);

#if BASE_PRECISION==32

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
//  cout << "call single site QDP_M_eq_M_times_M" << endl;

  typedef OLattice<PScalar<PColorMatrix<RComplexFloat, 3> > >    C;

  const C& l = static_cast<const C&>(rhs.expression().left());
  const C& r = static_cast<const C&>(rhs.expression().right());

  for(int i=s.start(); i <= s.end(); ++i) 
  {
    _inline_sse_mult_su3_nn(l.elem(i).elem(),r.elem(i).elem(),d.elem(i).elem());
  }
}


// GNUC vector type
typedef float v4sf __attribute__((mode(V4SF),aligned(16)));


// AXPY and AXMY routines
void vaxpy3(REAL32 *Out,REAL32 *scalep,REAL32 *InScale, REAL32 *Add,int n_3vec)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "SSE_TEST: vaxpy3" << endl;
#endif

//  int n_loops = n_3vec >> 2;   // only works on multiple of length 4 vectors
  int n_loops = n_3vec / 24;   // only works on multiple of length 24 vectors

//  register v4sf va = load_v4sf((float *)&a);
  v4sf vscalep = __builtin_ia32_loadss(scalep);
  asm("shufps\t$0,%0,%0" : "+x" (vscalep));

  for (; n_loops-- > 0; )
  {
    __builtin_ia32_storeaps(Out+ 0, __builtin_ia32_addps(__builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 0)), __builtin_ia32_loadaps(Add+ 0)));
    __builtin_ia32_storeaps(Out+ 4, __builtin_ia32_addps(__builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 4)), __builtin_ia32_loadaps(Add+ 4)));
    __builtin_ia32_storeaps(Out+ 8, __builtin_ia32_addps(__builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 8)), __builtin_ia32_loadaps(Add+ 8)));
    __builtin_ia32_storeaps(Out+12, __builtin_ia32_addps(__builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+12)), __builtin_ia32_loadaps(Add+12)));
    __builtin_ia32_storeaps(Out+16, __builtin_ia32_addps(__builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+16)), __builtin_ia32_loadaps(Add+16)));
    __builtin_ia32_storeaps(Out+20, __builtin_ia32_addps(__builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+20)), __builtin_ia32_loadaps(Add+20)));

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
  v4sf vscalep = __builtin_ia32_loadss(scalep);
  asm("shufps\t$0,%0,%0" : "+x" (vscalep));

  for (; n_loops-- > 0; )
  {
    __builtin_ia32_storeaps(Out+ 0, __builtin_ia32_subps(__builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 0)), __builtin_ia32_loadaps(Sub+ 0)));
    __builtin_ia32_storeaps(Out+ 4, __builtin_ia32_subps(__builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 4)), __builtin_ia32_loadaps(Sub+ 4)));
    __builtin_ia32_storeaps(Out+ 8, __builtin_ia32_subps(__builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 8)), __builtin_ia32_loadaps(Sub+ 8)));
    __builtin_ia32_storeaps(Out+12, __builtin_ia32_subps(__builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+12)), __builtin_ia32_loadaps(Sub+12)));
    __builtin_ia32_storeaps(Out+16, __builtin_ia32_subps(__builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+16)), __builtin_ia32_loadaps(Sub+16)));
    __builtin_ia32_storeaps(Out+20, __builtin_ia32_subps(__builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+20)), __builtin_ia32_loadaps(Sub+20)));

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
    __builtin_ia32_storeaps(Out+ 0, __builtin_ia32_addps(__builtin_ia32_loadaps(In1+ 0), __builtin_ia32_loadaps(In2+ 0)));
    __builtin_ia32_storeaps(Out+ 4, __builtin_ia32_addps(__builtin_ia32_loadaps(In1+ 4), __builtin_ia32_loadaps(In2+ 4)));
    __builtin_ia32_storeaps(Out+ 8, __builtin_ia32_addps(__builtin_ia32_loadaps(In1+ 8), __builtin_ia32_loadaps(In2+ 8)));
    __builtin_ia32_storeaps(Out+12, __builtin_ia32_addps(__builtin_ia32_loadaps(In1+12), __builtin_ia32_loadaps(In2+12)));
    __builtin_ia32_storeaps(Out+16, __builtin_ia32_addps(__builtin_ia32_loadaps(In1+16), __builtin_ia32_loadaps(In2+16)));
    __builtin_ia32_storeaps(Out+20, __builtin_ia32_addps(__builtin_ia32_loadaps(In1+20), __builtin_ia32_loadaps(In2+20)));

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
    __builtin_ia32_storeaps(Out+ 0, __builtin_ia32_subps(__builtin_ia32_loadaps(In1+ 0), __builtin_ia32_loadaps(In2+ 0)));
    __builtin_ia32_storeaps(Out+ 4, __builtin_ia32_subps(__builtin_ia32_loadaps(In1+ 4), __builtin_ia32_loadaps(In2+ 4)));
    __builtin_ia32_storeaps(Out+ 8, __builtin_ia32_subps(__builtin_ia32_loadaps(In1+ 8), __builtin_ia32_loadaps(In2+ 8)));
    __builtin_ia32_storeaps(Out+12, __builtin_ia32_subps(__builtin_ia32_loadaps(In1+12), __builtin_ia32_loadaps(In2+12)));
    __builtin_ia32_storeaps(Out+16, __builtin_ia32_subps(__builtin_ia32_loadaps(In1+16), __builtin_ia32_loadaps(In2+16)));
    __builtin_ia32_storeaps(Out+20, __builtin_ia32_subps(__builtin_ia32_loadaps(In1+20), __builtin_ia32_loadaps(In2+20)));

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
  v4sf vscalep = __builtin_ia32_loadss(scalep);
  asm("shufps\t$0,%0,%0" : "+x" (vscalep));

  for (; n_loops-- > 0; )
  {
    __builtin_ia32_storeaps(Out+ 0, __builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(In+ 0)));
    __builtin_ia32_storeaps(Out+ 4, __builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(In+ 4)));
    __builtin_ia32_storeaps(Out+ 8, __builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(In+ 8)));
    __builtin_ia32_storeaps(Out+12, __builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(In+12)));
    __builtin_ia32_storeaps(Out+16, __builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(In+16)));
    __builtin_ia32_storeaps(Out+20, __builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(In+20)));

    Out += 24; In += 24;
  }
}  

void local_sumsq(REAL32 *Out, REAL32 *In, int n_3vec)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "SSE_TEST: local_sumsq" << endl;
#endif

//  int n_loops = n_3vec >> 2;   // only works on multiple of length 4 vectors
  int n_loops = n_3vec / 24;   // only works on multiple of length 24 vectors

  *Out = 0.0;
  register v4sf vsum = __builtin_ia32_loadss(Out);
  asm("shufps\t$0,%0,%0" : "+x" (vsum));

  for (; n_loops-- > 0; )
  {
    register v4sf vtmp;

    vtmp = __builtin_ia32_loadaps(In+0);
    vsum = __builtin_ia32_addps(vsum, __builtin_ia32_mulps(vtmp, vtmp));
    vtmp = __builtin_ia32_loadaps(In+4);
    vsum = __builtin_ia32_addps(vsum, __builtin_ia32_mulps(vtmp, vtmp));
    vtmp = __builtin_ia32_loadaps(In+8);
    vsum = __builtin_ia32_addps(vsum, __builtin_ia32_mulps(vtmp, vtmp));
    vtmp = __builtin_ia32_loadaps(In+12);
    vsum = __builtin_ia32_addps(vsum, __builtin_ia32_mulps(vtmp, vtmp));
    vtmp = __builtin_ia32_loadaps(In+16);
    vsum = __builtin_ia32_addps(vsum, __builtin_ia32_mulps(vtmp, vtmp));
    vtmp = __builtin_ia32_loadaps(In+20);
    vsum = __builtin_ia32_addps(vsum, __builtin_ia32_mulps(vtmp, vtmp));

    In += 24;
  }

  REAL32 fsum[4];
  __builtin_ia32_storeaps(fsum, vsum);
  *Out = fsum[0] + fsum[1] + fsum[2] + fsum[3];
}

#endif // BASE PRECISION==32

QDP_END_NAMESPACE();

#endif  // defined(__GNUC__)
