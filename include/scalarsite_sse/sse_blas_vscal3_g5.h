// $Id: sse_blas_vscal3_g5.h,v 1.1 2005-03-18 11:55:29 bjoo Exp $

/*! @file
 *  @brief Generic Scalar VAXPY routine
 *
 */

#ifndef QDP_SSE_BLAS_VSCAL_G5
#define QDP_SSE_BLAS_VSCAL_G5

#if defined(__GNUC__)

#include "qdp_config.h"

QDP_BEGIN_NAMESPACE(QDP);

#if BASE_PRECISION==32



// (Vector) out = (*scalep)* P_{+} X
inline
void scal_g5ProjPlus(REAL32 *Out, REAL32* scalep, REAL32 *X, int n_4vec)
{
  // GNUC vector type
  typedef float v4sf __attribute__((mode(V4SF),aligned(16)));

  v4sf vscalep = __builtin_ia32_loadss(scalep);
  asm("shufps\t$0,%0,%0" : "+x" (vscalep)); 
  REAL32 rzero = (REAL32)0;
  
  // A zero vector
  v4sf vzero = __builtin_ia32_loadss(&rzero);
  asm("shufps\t$0,%0,%0" : "+x" (vzero)); 
  
  for(int i=0; i < n_4vec; i++) {

    // Spin Component 0: z0r, z0i, z1r, z1i
    __builtin_ia32_storeaps(Out+ 0, __builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(X+ 0)));
     
    // Spin Component 0: z2r, z2i, SpinComponent 1: z0r, z0i
    __builtin_ia32_storeaps(Out+ 4, __builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(X+ 4)));


    // Spin Component 1: z1r, z1i, z2r, z2i
    __builtin_ia32_storeaps(Out+ 8, __builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(X+ 8)));

    
    // Spin Component 2: z0r, z0i, z1r, z1i
    __builtin_ia32_storeaps(Out+12, vzero);

    // Spin Component 2: z2r, z2i, z0r, z0r
    __builtin_ia32_storeaps(Out+16, vzero);

    // Spin Component 3: z1r, z1i, z2r, z2i
    __builtin_ia32_storeaps(Out+20, vzero);

    // Update offsets
    Out += 24;  X += 24;
  }

}


// (Vector) out = (*scalep)* P_{-} X
inline
void scal_g5ProjMinus(REAL32 *Out, REAL32* scalep, REAL32 *X, int n_4vec)
{
  // GNUC vector type
  typedef float v4sf __attribute__((mode(V4SF),aligned(16)));

  v4sf vscalep = __builtin_ia32_loadss(scalep);
  asm("shufps\t$0,%0,%0" : "+x" (vscalep)); 
  REAL32 rzero = (REAL32)0;
  
  // A zero vector
  v4sf vzero = __builtin_ia32_loadss(&rzero);
  asm("shufps\t$0,%0,%0" : "+x" (vzero)); 
  
  for(int i=0; i < n_4vec; i++) {
    // Spin Component 0: z0r, z0i, z1r, z1i
    __builtin_ia32_storeaps(Out+0, vzero);

    // Spin Component 0: z2r, z2i, Spin Component 1: z0r, z0r
    __builtin_ia32_storeaps(Out+4, vzero);

    // Spin Component 1: z1r, z1i, z2r, z2i
    __builtin_ia32_storeaps(Out+8, vzero);


    // Spin Component 2: z0r, z0i, z1r, z1i
    __builtin_ia32_storeaps(Out+ 12, __builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(X+ 12)));
     
    // Spin Component 2: z2r, z2i, SpinComponent 3: z0r, z0i
    __builtin_ia32_storeaps(Out+ 16, __builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(X+ 16)));


    // Spin Component 3: z1r, z1i, z2r, z2i
    __builtin_ia32_storeaps(Out+ 20, __builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(X+ 20)));

    

    // Update offsets
    Out += 24;  X += 24;
  }

}


#endif

QDP_END_NAMESPACE(QDP);


#endif // GNUC

#endif // guard
