// $Id: sse_blas_vaxpy3_g5.h,v 1.1 2005-03-17 14:42:55 bjoo Exp $

/*! @file
 *  @brief Generic Scalar VAXPY routine
 *
 */

#ifndef QDP_SSE_BLAS_VAXPY3_G5
#define QDP_SSE_BLAS_VAXPY3_G5

#if defined(__GNUC__)

#include "qdp_config.h"

QDP_BEGIN_NAMESPACE(QDP);

#if BASE_PRECISION==32



// (Vector) out = (Scalar) (*scalep) * (Vector) InScale + (Vector) P{+} Add
inline
void axpyz_g5ProjPlus(REAL32 *Out,REAL32 *scalep,REAL32 *InScale, REAL32 *Add,int n_4vec)
{
  // GNUC vector type
  typedef float v4sf __attribute__((mode(V4SF),aligned(16)));

  // Load Vscalep
  v4sf vscalep = __builtin_ia32_loadss(scalep);
  asm("shufps\t$0,%0,%0" : "+x" (vscalep));

  for(int i=0; i < n_4vec; i++) {

    // Spin Component 0: z0r, z0i, z1r, z1i
    __builtin_ia32_storeaps(Out+ 0, __builtin_ia32_addps(__builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 0)), __builtin_ia32_loadaps(Add+ 0)));
     
    // Spin Component 0: z2r, z2i, SpinComponent 1: z0r, z0i
    __builtin_ia32_storeaps(Out+ 4, __builtin_ia32_addps(__builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 4)), __builtin_ia32_loadaps(Add+ 4)));

    // Spin Component 1: z1r, z1i, z2r, z2i
    __builtin_ia32_storeaps(Out+ 8, __builtin_ia32_addps(__builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 8)), __builtin_ia32_loadaps(Add+ 8)));

    // Spin Component 2: z0r, z0i, z1r, z1i
    __builtin_ia32_storeaps(Out+12, __builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 12)));

    // Spin Component 2: z2r, z2i, z0r, z0r
    __builtin_ia32_storeaps(Out+16, __builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 16)));

    // Spin Component 3: z1r, z1i, z2r, z2i
    __builtin_ia32_storeaps(Out+20, __builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 20)));

    // Update offsets
    Out += 24; InScale += 24; Add += 24;
  }

}

// (Vector) out = (Scalar) (*scalep) * (Vector) InScale + (Vector) P{+} Add
inline
void axpyz_g5ProjMinus(REAL32 *Out,REAL32 *scalep,REAL32 *InScale, REAL32 *Add,int n_4vec)
{
  // GNUC vector type
  typedef float v4sf __attribute__((mode(V4SF),aligned(16)));

  // Load Vscalep
  v4sf vscalep = __builtin_ia32_loadss(scalep);
  asm("shufps\t$0,%0,%0" : "+x" (vscalep));

  for(int i=0; i < n_4vec; i++) {
    // Spin Component 0: z0r, z0i, z1r, z1i
    __builtin_ia32_storeaps(Out+0, __builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 0)));

    // Spin Component 0: z2r, z2i, Spin Component 1: z0r, z0r
    __builtin_ia32_storeaps(Out+4, __builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 4)));

    // Spin Component 1: z1r, z1i, z2r, z2i
    __builtin_ia32_storeaps(Out+8, __builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 8)));


    // Spin Component 2: z0r, z0i, z1r, z1i
    __builtin_ia32_storeaps(Out+12, __builtin_ia32_addps(__builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 12)), __builtin_ia32_loadaps(Add+ 12)));
     
    // Spin Component 2: z2r, z2i, SpinComponent 3: z0r, z0i
    __builtin_ia32_storeaps(Out+ 16, __builtin_ia32_addps(__builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 16)), __builtin_ia32_loadaps(Add+ 16)));

    // Spin Component 3: z1r, z1i, z2r, z2i
    __builtin_ia32_storeaps(Out+ 20, __builtin_ia32_addps(__builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 20)), __builtin_ia32_loadaps(Add+ 20)));
    // Update offsets
    Out += 24; InScale += 24; Add += 24;
  }

}



// AXMY  versions
// (Vector) out = (Scalar) (*scalep) * (Vector) InScale - (Vector) P{+} Add
inline
void axmyz_g5ProjPlus(REAL32 *Out,REAL32 *scalep,REAL32 *InScale, REAL32 *Add,int n_4vec)
{
  // GNUC vector type
  typedef float v4sf __attribute__((mode(V4SF),aligned(16)));

  // Load Vscalep

  v4sf vscalep = __builtin_ia32_loadss(scalep);
  asm("shufps\t$0,%0,%0" : "+x" (vscalep));  

  for(int i=0; i < n_4vec; i++) {
    
    // Spin Component 0: z0r, z0i, z1r, z1i
    __builtin_ia32_storeaps(Out+ 0, __builtin_ia32_subps(__builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 0)), __builtin_ia32_loadaps(Add+ 0)));
    
     
    // Spin Component 0: z2r, z2i, SpinComponent 1: z0r, z0i
    __builtin_ia32_storeaps(Out+ 4, __builtin_ia32_subps(__builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 4)), __builtin_ia32_loadaps(Add+ 4)));
    
     // Spin Component 1: z1r, z1i, z2r, z2i
    __builtin_ia32_storeaps(Out+ 8, __builtin_ia32_subps(__builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 8)), __builtin_ia32_loadaps(Add+ 8)));
  
    // Spin Component 2: z0r, z0i, z1r, z1i
    __builtin_ia32_storeaps(Out+12, __builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 12)));
    
    
    
    // Spin Component 2: z2r, z2i, z0r, z0r
    __builtin_ia32_storeaps(Out+16, __builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 16)));

    // Spin Component 3: z1r, z1i, z2r, z2i
    __builtin_ia32_storeaps(Out+20, __builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 20)));

    // Update offsets
    Out += 24; InScale += 24; Add += 24;
  }


}

// (Vector) out = (Scalar) (*scalep) * (Vector) InScale - (Vector) P{-} Add
inline
void axmyz_g5ProjMinus(REAL32 *Out,REAL32 *scalep,REAL32 *InScale, REAL32 *Add,int n_4vec)
{
  // GNUC vector type
  typedef float v4sf __attribute__((mode(V4SF),aligned(16)));

  // Load Vscalep
  v4sf vscalep = __builtin_ia32_loadss(scalep);
  asm("shufps\t$0,%0,%0" : "+x" (vscalep));

  for(int i=0; i < n_4vec; i++) {
    // Spin Component 0: z0r, z0i, z1r, z1i
    __builtin_ia32_storeaps(Out+0, __builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 0)));

    // Spin Component 0: z2r, z2i, Spin Component 1: z0r, z0r
    __builtin_ia32_storeaps(Out+4, __builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 4)));

    // Spin Component 1: z1r, z1i, z2r, z2i
    __builtin_ia32_storeaps(Out+8, __builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 8)));


    // Spin Component 2: z0r, z0i, z1r, z1i
    __builtin_ia32_storeaps(Out+12, __builtin_ia32_subps(__builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 12)), __builtin_ia32_loadaps(Add+ 12)));
     
    // Spin Component 2: z2r, z2i, SpinComponent 3: z0r, z0i
    __builtin_ia32_storeaps(Out+ 16, __builtin_ia32_subps(__builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 16)), __builtin_ia32_loadaps(Add+ 16)));

    // Spin Component 3: z1r, z1i, z2r, z2i
    __builtin_ia32_storeaps(Out+ 20, __builtin_ia32_subps(__builtin_ia32_mulps(vscalep, __builtin_ia32_loadaps(InScale+ 20)), __builtin_ia32_loadaps(Add+ 20)));
    // Update offsets
    Out += 24; InScale += 24; Add += 24;
  }

}


#endif

QDP_END_NAMESPACE(QDP);


#endif // GNUC

#endif // guard
