// $Id: sse_blas_vadd3_g5.h,v 1.1 2005-03-17 14:42:55 bjoo Exp $

/*! @file
 *  @brief Generic Scalar VAXPY routine
 *
 */

#ifndef QDP_SSE_BLAS_VADD_G5
#define QDP_SSE_BLAS_VADD_G5

#if defined(__GNUC__)

#include "qdp_config.h"

QDP_BEGIN_NAMESPACE(QDP);

#if BASE_PRECISION==32



// (Vector) out = X + P_{+} (Vector) Y 
inline
void add_g5ProjPlus(REAL32 *Out, REAL32 *X, REAL32 *Y,int n_4vec)
{

  for(int i=0; i < n_4vec; i++) {

    // Spin Component 0: z0r, z0i, z1r, z1i
    __builtin_ia32_storeaps(Out+ 0, __builtin_ia32_addps(__builtin_ia32_loadaps(Y+ 0), __builtin_ia32_loadaps(X+ 0)));
     
    // Spin Component 0: z2r, z2i, SpinComponent 1: z0r, z0i
    __builtin_ia32_storeaps(Out+ 4, __builtin_ia32_addps(__builtin_ia32_loadaps(Y+ 4), __builtin_ia32_loadaps(X+ 4)));


    // Spin Component 1: z1r, z1i, z2r, z2i
    __builtin_ia32_storeaps(Out+ 8, __builtin_ia32_addps(__builtin_ia32_loadaps(Y+ 8), __builtin_ia32_loadaps(X+ 8)));

    // Spin Component 2: z0r, z0i, z1r, z1i
    __builtin_ia32_storeaps(Out+12, __builtin_ia32_loadaps(X+ 12));

    // Spin Component 2: z2r, z2i, z0r, z0r
    __builtin_ia32_storeaps(Out+16, __builtin_ia32_loadaps(X+ 16));

    // Spin Component 3: z1r, z1i, z2r, z2i
    __builtin_ia32_storeaps(Out+20, __builtin_ia32_loadaps(X+ 20));

    // Update offsets
    Out += 24; Y += 24; X += 24;
  }

}

// (Vector) out = X + (Vector) P{-} Y
inline
void add_g5ProjMinus(REAL32 *Out,REAL32 *X, REAL32 *Y,int n_4vec)
{

  for(int i=0; i < n_4vec; i++) {
    // Spin Component 0: z0r, z0i, z1r, z1i
    __builtin_ia32_storeaps(Out+0,__builtin_ia32_loadaps(X+ 0));

    // Spin Component 0: z2r, z2i, Spin Component 1: z0r, z0r
    __builtin_ia32_storeaps(Out+4,__builtin_ia32_loadaps(X+ 4));

    // Spin Component 1: z1r, z1i, z2r, z2i
    __builtin_ia32_storeaps(Out+8, __builtin_ia32_loadaps(X+ 8));


    // Spin Component 2: z0r, z0i, z1r, z1i
    __builtin_ia32_storeaps(Out+12, __builtin_ia32_addps(__builtin_ia32_loadaps(X+ 12), __builtin_ia32_loadaps(Y+ 12)));
     
    // Spin Component 2: z2r, z2i, SpinComponent 3: z0r, z0i
    __builtin_ia32_storeaps(Out+16, __builtin_ia32_addps(__builtin_ia32_loadaps(X+ 16), __builtin_ia32_loadaps(Y+ 16)));

    // Spin Component 3: z1r, z1i, z2r, z2i
    __builtin_ia32_storeaps(Out+20, __builtin_ia32_addps(__builtin_ia32_loadaps(X+ 20), __builtin_ia32_loadaps(Y+ 20)));

    // Update offsets
    Out += 24; Y += 24; X += 24;
  }

}



// AXMY  versions
// (Vector) out = (Scalar) (*scalep) * (Vector) Y - (Vector) P{+} X
inline
void sub_g5ProjPlus(REAL32 *Out, REAL32 *X, REAL32 *Y,int n_4vec)
{
  for(int i=0; i < n_4vec; i++) {
    // Spin Component 0: z0r, z0i, z1r, z1i
    __builtin_ia32_storeaps(Out+ 0, __builtin_ia32_subps(__builtin_ia32_loadaps(X+ 0), __builtin_ia32_loadaps(Y + 0)));
     
    // Spin Component 0: z2r, z2i, SpinComponent 1: z0r, z0i
    __builtin_ia32_storeaps(Out+ 4, __builtin_ia32_subps(__builtin_ia32_loadaps(X+ 4), __builtin_ia32_loadaps(Y + 4)));


    // Spin Component 1: z1r, z1i, z2r, z2i
    __builtin_ia32_storeaps(Out+ 8, __builtin_ia32_subps(__builtin_ia32_loadaps(X+ 8), __builtin_ia32_loadaps(Y+ 8)));
    
  
    // Spin Component 2: z0r, z0i, z1r, z1i
    __builtin_ia32_storeaps(Out+12, __builtin_ia32_loadaps(X + 12));
        
    
    // Spin Component 2: z2r, z2i, z0r, z0r
    __builtin_ia32_storeaps(Out+16, __builtin_ia32_loadaps(X+ 16));


    // Spin Component 3: z1r, z1i, z2r, z2i
    __builtin_ia32_storeaps(Out+20, __builtin_ia32_loadaps(X+20));

    // Update offsets
    Out += 24; Y += 24; X += 24;
  }


}

// (Vector) out = Y - (Vector) P{-} X
inline
void sub_g5ProjMinus(REAL32 *Out,REAL32 *X, REAL32 *Y,int n_4vec)
{

  for(int i=0; i < n_4vec; i++) {
    // Spin Component 0: z0r, z0i, z1r, z1i
    __builtin_ia32_storeaps(Out+0, __builtin_ia32_loadaps(X+0));

    // Spin Component 0: z2r, z2i, Spin Component 1: z0r, z0r
    __builtin_ia32_storeaps(Out+4, __builtin_ia32_loadaps(X+4));

    // Spin Component 1: z1r, z1i, z2r, z2i
    __builtin_ia32_storeaps(Out+8, __builtin_ia32_loadaps(X+8));

    // Spin Component 2: z0r, z0i, z1r, z1i
    __builtin_ia32_storeaps(Out+12, __builtin_ia32_subps(__builtin_ia32_loadaps(X+ 12), __builtin_ia32_loadaps(Y+ 12)));
     
    // Spin Component 2: z2r, z2i, SpinComponent 3: z0r, z0i
    __builtin_ia32_storeaps(Out+16, __builtin_ia32_subps(__builtin_ia32_loadaps(X+ 16), __builtin_ia32_loadaps(Y+ 16)));

    // Spin Component 3: z1r, z1i, z2r, z2i
    __builtin_ia32_storeaps(Out+20, __builtin_ia32_subps(__builtin_ia32_loadaps(X+ 20), __builtin_ia32_loadaps(Y+ 20)));

    // Update offsets
    Out += 24; Y += 24; X += 24;
  }

}


#endif

QDP_END_NAMESPACE(QDP);


#endif // GNUC

#endif // guard
