// $Id: sse_linalg_m_eq_mm_double.cc,v 1.1 2008-06-24 20:37:46 bjoo Exp $

/*! @file
 *  @brief Generic Scalar VAXPY routine
 *
 */

#include "scalarsite_sse/sse_linalg_mm_su3_double.h"

namespace QDP {

#include <xmmintrin.h>


#if 0 
#define CMUL(z,x,y)		\
  { \
    __m128d t1,t2,t3; \
    t1 = _mm_mul_pd((x),(y)); \
    t2 = _mm_shuffle_pd(t1,t1,0x1); \
    t3 = _mm_shuffle_pd((y),(y),0x1);\
    (z) = _mm_sub_pd(t1,t2); \
    t2 = _mm_mul_pd((x),t3); \
    t3 = _mm_shuffle_pd(t2,t2,0x1); \
    t3 = _mm_add_pd(t2,t3); \
    (z)= _mm_shuffle_pd((z),t3,0x2); \
  }

#define CMADD(z,x,y)				\
  { \
    __m128d t1,t2,t3,t4; \
    t1 = _mm_mul_pd((x),(y)); \
    t2 = _mm_shuffle_pd(t1,t1,0x1); \
    t3 = _mm_shuffle_pd((y),(y),0x1);\
    t4 = _mm_sub_pd(t1,t2); \
    t2 = _mm_mul_pd((x),t3); \
    t3 = _mm_shuffle_pd(t2,t2,0x1); \
    t3 = _mm_add_pd(t2,t3); \
    t4= _mm_shuffle_pd(t4,t3,0x2); \
    (z) = _mm_add_pd((z),t4); \
  }

#endif

#include <pmmintrin.h>

/* SSE 3? */
#define CMUL(z,x,y)		\
  { \
    __m128d t1,t2; \
    t1 = _mm_mul_pd((x),(y)); \
    (z) = _mm_hsub_pd(t1,t1);			\
    t2 = _mm_shuffle_pd((y),(y),0x1);\
    t2 = _mm_mul_pd((x),t2); \
    t2 = _mm_hadd_pd(t2,t2); \
    (z)= _mm_shuffle_pd((z),t2,0x2);		\
  }

#define CMADD(z,x,y)				\
  { \
    __m128d t1,t2; \
    t1 = _mm_mul_pd((x),(y)); \
    t1 = _mm_hsub_pd(t1,t1); \
    t2 = _mm_shuffle_pd((y),(y),0x1);\
    t2 = _mm_mul_pd((x),t2); \
    t2 = _mm_hadd_pd(t2,t2); \
    t1= _mm_shuffle_pd(t1,t2,0x2);		\
    (z) = _mm_add_pd((z),t1);			\
  }

  /* M3 = M1*M2 */
  void ssed_m_eq_mm(REAL64* m3, REAL64* m2, REAL64* m1, int n_mat)
  {
    __m128d m1_1;
    __m128d m1_2;
    __m128d m1_3;

    __m128d m2_1;
    __m128d m2_2;
    
    __m128d m3_11;
    __m128d m3_12;
    __m128d m3_13;

    __m128d m3_21;
    __m128d m3_22;
    __m128d m3_23;

   

    REAL64* m1_p=m1;
    REAL64* m2_p=m2;
    REAL64* m3_p=m3;

    for(int i=0; i < n_mat; i++) { 

      // First row of M2 into all columns of M1
    
      m2_1 = _mm_load_pd(m2_p);
      m2_2 = _mm_load_pd(m2_p+6);
      m1_1 = _mm_load_pd(m1_p);      
      m1_2 = _mm_load_pd(m1_p+2);
      m1_3 = _mm_load_pd(m1_p+4);

      CMUL(m3_11, m2_1, m1_1);
      CMUL(m3_12, m2_1, m1_2);
      CMUL(m3_13, m2_1, m1_3);

      CMUL(m3_21, m2_2, m1_1);
      CMUL(m3_22, m2_2, m1_2);
      CMUL(m3_23, m2_2, m1_3);

      m2_1 = _mm_load_pd(m2_p+2);
      m2_2 = _mm_load_pd(m2_p+8);
      m1_1 = _mm_load_pd(m1_p+6);
      m1_2 = _mm_load_pd(m1_p+8);
      m1_3 = _mm_load_pd(m1_p+10);

      CMADD(m3_11, m2_1, m1_1);
      CMADD(m3_12, m2_1, m1_2);
      CMADD(m3_13, m2_1, m1_3);

      CMADD(m3_21, m2_2, m1_1);
      CMADD(m3_22, m2_2, m1_2);
      CMADD(m3_23, m2_2, m1_3);

      m2_1 = _mm_load_pd(m2_p+4);
      m2_2 = _mm_load_pd(m2_p+10);
      m1_1 = _mm_load_pd(m1_p+12);
      m1_2 = _mm_load_pd(m1_p+14);
      m1_3 = _mm_load_pd(m1_p+16);

      CMADD(m3_11, m2_1, m1_1);
      _mm_store_pd(m3_p, m3_11);

      CMADD(m3_12, m2_1, m1_2);
      _mm_store_pd(m3_p+2, m3_12);

      CMADD(m3_13, m2_1, m1_3);
      _mm_store_pd(m3_p+4, m3_13);

      CMADD(m3_21, m2_2, m1_1);
      _mm_store_pd(m3_p+6, m3_21);

      CMADD(m3_22, m2_2, m1_2);
      _mm_store_pd(m3_p+8, m3_22);

      CMADD(m3_23, m2_2, m1_3);
      _mm_store_pd(m3_p+10, m3_23);


      m2_1 = _mm_load_pd(m2_p+12);
      m1_1 = _mm_load_pd(m1_p);      
      m1_2 = _mm_load_pd(m1_p+2);
      m1_3 = _mm_load_pd(m1_p+4);

      CMUL(m3_11, m2_1, m1_1);
      CMUL(m3_12, m2_1, m1_2);
      CMUL(m3_13, m2_1, m1_3);


      m2_1 = _mm_load_pd(m2_p+14);
      m1_1 = _mm_load_pd(m1_p+6);
      m1_2 = _mm_load_pd(m1_p+8);
      m1_3 = _mm_load_pd(m1_p+10);

      CMADD(m3_11, m2_1, m1_1);
      CMADD(m3_12, m2_1, m1_2);
      CMADD(m3_13, m2_1, m1_3);

      m2_1 = _mm_load_pd(m2_p+16);
      m1_1 = _mm_load_pd(m1_p+12);
      m1_2 = _mm_load_pd(m1_p+14);
      m1_3 = _mm_load_pd(m1_p+16);

      CMADD(m3_11, m2_1, m1_1);
      _mm_store_pd(m3_p+12, m3_11);

      CMADD(m3_12, m2_1, m1_2);
      _mm_store_pd(m3_p+14, m3_12);

      CMADD(m3_13, m2_1, m1_3);
      _mm_store_pd(m3_p+16, m3_13);

      /* Next matrix */
      m1_p += 18; m2_p += 18; m3_p += 18;
    }

  }



} // namespace QDP;

