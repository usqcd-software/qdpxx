// $Id: generic_blas_local_vcdot.h,v 1.6 2009-09-15 20:48:41 bjoo Exp $

/*! @file
 *  @brief Generic Scalar, CDOT  routine
 *
 */

#ifndef QDP_GENERIC_BLAS_LOCAL_VCDOT
#define QDP_GENERIC_BLAS_LOCAL_VCDOT


namespace QDP {

// Out = < V1, V2 > = V1^{dagger} V2
// Out is complex: Out_re, Out_im 
// V1 V2 are complex vectors of length 3*n_3vec
//volatile
inline
void l_vcdot(DOUBLE *Out_re, DOUBLE *Out_im, REAL *V1, REAL *V2, int n_3vec)
{
   double result_re;
   double result_im;

  
   double v1_0r;
   double v1_0i;
   double v1_1r;
   double v1_1i;
   double v1_2r;
   double v1_2i;

   double v2_0r;
   double v2_0i;
   double v2_1r;
   double v2_1i;
   double v2_2r;
   double v2_2i;

   int counter=0;
   unsigned long vecptr1=0;
   unsigned long vecptr2=0;
  result_re= 0;
  result_im= 0;

  
  if( n_3vec > 0 ) { 

    // Prefetch 
    v1_0r = (DOUBLE)V1[vecptr1++];
    v2_0r = (DOUBLE)V2[vecptr2++];

    v1_0i = (DOUBLE)V1[vecptr1++];
    v2_0i = (DOUBLE)V2[vecptr2++];

    v1_1r = (DOUBLE)V1[vecptr1++];
    v2_1r = (DOUBLE)V2[vecptr2++];
    int len = 4*n_3vec;

    for(counter=0; counter < len-1; counter++) {
      result_re = result_re + v1_0r*v2_0r;
      v1_1i =(DOUBLE)V1[vecptr1++];
      result_im = result_im - v1_0i*v2_0r;
      v2_1i = (DOUBLE)V2[vecptr2++];
      result_im = result_im + v1_0r*v2_0i;
      v1_2r = (DOUBLE)V1[vecptr1++];
      result_re = result_re + v1_0i*v2_0i;
      v2_2r = (DOUBLE)V2[vecptr2++];
      
      result_re = result_re + v1_1r*v2_1r;
      v1_2i = (DOUBLE)V1[vecptr1++];
      result_im = result_im - v1_1i*v2_1r;
      v2_2i = (DOUBLE)V2[vecptr2++];
      result_im = result_im + v1_1r*v2_1i;
      v1_0r = (DOUBLE)V1[vecptr1++];
      result_re = result_re + v1_1i*v2_1i;
      v2_0r = (DOUBLE)V2[vecptr2++];

      result_re = result_re + v1_2r*v2_2r;
      v1_0i = (DOUBLE)V1[vecptr1++];
      result_im = result_im - v1_2i*v2_2r;
      v2_0i = (DOUBLE)V2[vecptr2++];
      result_im = result_im + v1_2r*v2_2i;
      v1_1r = (DOUBLE)V1[vecptr1++];
      result_re = result_re + v1_2i*v2_2i;
      v2_1r = (DOUBLE)V2[vecptr2++];

    }

    // Last one plus drain...
    result_re = result_re + v1_0r*v2_0r;
    v1_1i =(DOUBLE)V1[vecptr1++];
    result_im = result_im - v1_0i*v2_0r;
    v2_1i = (DOUBLE)V2[vecptr2++];
    result_im = result_im + v1_0r*v2_0i;
    v1_2r = (DOUBLE)V1[vecptr1++];
    result_re = result_re + v1_0i*v2_0i;
    v2_2r = (DOUBLE)V2[vecptr2++];
      
    result_re = result_re + v1_1r*v2_1r;
    v1_2i = (DOUBLE)V1[vecptr1++];
    result_im = result_im - v1_1i*v2_1r;
    v2_2i = (DOUBLE)V2[vecptr2++];
    result_im = result_im + v1_1r*v2_1i;
    result_re = result_re + v1_1i*v2_1i;
    

    result_re = result_re + v1_2r*v2_2r;
    result_im = result_im - v1_2i*v2_2r;
    result_im = result_im + v1_2r*v2_2i;
    result_re = result_re + v1_2i*v2_2i;    

  }
  
  *Out_re=(DOUBLE)result_re;
  *Out_im=(DOUBLE)result_im;
}


} // namespace QDP;

#endif // guard
