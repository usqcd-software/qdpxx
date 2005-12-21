// $Id: qdp_scalarsite_sse.cc,v 1.11 2005-12-21 16:04:25 bjoo Exp $

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

void local_sumsq(REAL32 *Out, REAL32 *In, int n_3vec)
{
#ifdef DEBUG_BLAS
  QDPIO::cout << "SSE_TEST: local_sumsq" << endl;
#endif

//  int n_loops = n_3vec >> 2;   // only works on multiple of length 4 vectors
  int n_loops = n_3vec / 24;   // only works on multiple of length 24 vectors

  *Out = 0.0;
  register v4sf vsum = _mm_load_ss(Out);
  asm("shufps\t$0,%0,%0" : "+x" (vsum));

  for (; n_loops-- > 0; )
  {
    register v4sf vtmp;

    vtmp = _mm_load_ps(In+0);
    vsum = _mm_add_ps(vsum, _mm_mul_ps(vtmp, vtmp));
    vtmp = _mm_load_ps(In+4);
    vsum = _mm_add_ps(vsum, _mm_mul_ps(vtmp, vtmp));
    vtmp = _mm_load_ps(In+8);
    vsum = _mm_add_ps(vsum, _mm_mul_ps(vtmp, vtmp));
    vtmp = _mm_load_ps(In+12);
    vsum = _mm_add_ps(vsum, _mm_mul_ps(vtmp, vtmp));
    vtmp = _mm_load_ps(In+16);
    vsum = _mm_add_ps(vsum, _mm_mul_ps(vtmp, vtmp));
    vtmp = _mm_load_ps(In+20);
    vsum = _mm_add_ps(vsum, _mm_mul_ps(vtmp, vtmp));

    In += 24;
  }

  REAL32 fsum[4];
  _mm_store_ps(fsum, vsum);
  *Out = fsum[0] + fsum[1] + fsum[2] + fsum[3];
}

#endif // BASE PRECISION==32

QDP_END_NAMESPACE();

#endif  // defined(__GNUC__)
