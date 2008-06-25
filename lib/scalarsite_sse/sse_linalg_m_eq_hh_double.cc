// $Id: sse_linalg_m_eq_hh_double.cc,v 1.1 2008-06-25 20:34:30 bjoo Exp $

/*! @file
 *  @brief Generic Scalar VAXPY routine
 *
 */

#include "scalarsite_sse/sse_linalg_mm_su3_double.h"

namespace QDP {

#include <xmmintrin.h>
#include <pmmintrin.h>

#if 0

  /* SSE 2 */

#define CONJMUL(z,x,y)		\
  { \
    __m128d t1,t2,t3,t4; \
    t1 = _mm_mul_pd(x,y); \
    t2 = _mm_shuffle_pd(t1,t1,0x1); \
    t3 = _mm_shuffle_pd(y,y,0x1);\
    z = _mm_add_pd(t1,t2); \
    t2 = _mm_mul_pd(x,t3); \
    t3 = _mm_shuffle_pd(t2,t2,0x1); \
    t3 = _mm_sub_pd(t2,t3);	    \
    z= _mm_shuffle_pd(z,t3,0x2); \
  }

#define CONJMADD(z,x,y)				\
  { \
    __m128d t1,t2,t3,t4; \
    t1 = _mm_mul_pd(x,y); \
    t2 = _mm_shuffle_pd(t1,t1,0x1); \
    t3 = _mm_shuffle_pd(y,y,0x1);\
    t4 = _mm_add_pd(t1,t2); \
    t2 = _mm_mul_pd(x,t3); \
    t3 = _mm_shuffle_pd(t2,t2,0x1); \
    t3 = _mm_sub_pd(t2,t3); \
    t4= _mm_shuffle_pd(t4,t3,0x2); \
    z = _mm_add_pd(z,t4); \
  }

#else

  /* SSE 3 */
#include <pmmintrin.h>

#define CONJMUL(z,x,y)		\
  { \
    __m128d t1; \
    t1 = _mm_mul_pd((x),(y)); \
    (z) = _mm_hadd_pd(t1,t1);			\
    t1 = _mm_shuffle_pd((x),(x),0x1);\
    t1 = _mm_mul_pd((y),t1); \
    t1 = _mm_hsub_pd(t1,t1); \
    (z)= _mm_shuffle_pd((z),t1,0x2);		\
  }

#define CONJMADD(z,x,y)				\
  { \
    __m128d t1,t2; \
    t1 = _mm_mul_pd((x),(y)); \
    t1 = _mm_hadd_pd(t1,t1); \
    t2 = _mm_shuffle_pd((x),(x),0x1);\
    t2 = _mm_mul_pd((y),t2); \
    t2 = _mm_hsub_pd(t2,t2); \
    t1= _mm_shuffle_pd(t1,t2,0x2);		\
    (z) = _mm_add_pd((z),t1);			\
  }



#endif


  /* M3 = M1*adj(M2) */
  void ssed_m_eq_hh(REAL64* m3, REAL64* m2, REAL64* m1, int n_mat)
  {
    __m128d m1_1;
    __m128d m1_2;
    __m128d m1_3;

    __m128d m2_1;
    __m128d m2_2;
    __m128d m2_3;

    __m128d m3_11;
    __m128d m3_12;
    __m128d m3_13;


    REAL64* m1_p=m1;
    REAL64* m2_p=m2;
    REAL64* m3_p=m3;



    for(int i=0; i < n_mat; i++) { 

      /* Next matrix */
      m1_p += 18; m2_p += 18; m3_p += 18;
    }

  }

  /* M3 += a M1*M2 */
  void ssed_m_peq_ahh(REAL64* m3, REAL64* a, REAL64* m2, REAL64* m1, int n_mat)
  {
    __m128d m1_1;
    __m128d m1_2;
    __m128d m1_3;

    __m128d m2_1;
    __m128d m2_2;
    __m128d m2_3;

    __m128d m3_11;
    __m128d tmp1;
    __m128d tmp2;

    __m128d scalar;
    
    scalar = _mm_loaddup_pd(a);

    REAL64* m1_p=m1;
    REAL64* m2_p=m2;
    REAL64* m3_p=m3;

    for(int i =0; i < n_mat; i++) { 
      m1_p += 18; m2_p += 18; m3_p += 18;
    }

  }



} // namespace QDP;

