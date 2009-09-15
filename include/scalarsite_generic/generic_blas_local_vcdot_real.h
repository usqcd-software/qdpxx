// $Id: generic_blas_local_vcdot_real.h,v 1.6 2009-09-15 20:48:41 bjoo Exp $

/*! @file
 *  @brief Generic Scalar, CDOT  routine
 *
 */

#ifndef QDP_GENERIC_BLAS_LOCAL_VCDOT_REAL
#define QDP_GENERIC_BLAS_LOCAL_VCDOT_REAL


namespace QDP {

// Out = Re (< V1, V2 >) = Re(V1^{dagger} V2)
// Out  REAL
// V1 V2 are complex vectors of length 3*n_3vec
// volatile
inline
void l_vcdot_real(DOUBLE *Out, REAL *V1, REAL *V2, int n_3vec)
{
  register double result;
  
  register double v1_0r;
  register double v1_0i;
  register double v1_1r;
  register double v1_1i;
  register double v1_2r;
  register double v1_2i;

  register double v2_0r;
  register double v2_0i;
  register double v2_1r;
  register double v2_1i;
  register double v2_2r;
  register double v2_2i;

  register int counter=0;
  register unsigned long vecptr1=0;
  register unsigned long vecptr2=0;
  result= 0;

  int len = 4*n_3vec;

  if( n_3vec > 0 ) { 

    // Prefetch 
    v1_0r = (DOUBLE)V1[vecptr1++];
    v2_0r = (DOUBLE)V2[vecptr2++];

    v1_0i = (DOUBLE)V1[vecptr1++];
    v2_0i = (DOUBLE)V2[vecptr2++];

    v1_1r = (DOUBLE)V1[vecptr1++];
    v2_1r = (DOUBLE)V2[vecptr2++];

    v1_1i =(DOUBLE)V1[vecptr1++];
    v2_1i = (DOUBLE)V2[vecptr2++];
    
    v1_2r = (DOUBLE)V1[vecptr1++];
    v2_2r = (DOUBLE)V2[vecptr2++];
    
    v1_2i = (DOUBLE)V1[vecptr1++];
    v2_2i = (DOUBLE)V2[vecptr2++];

    for(counter=0; counter < len-1; counter++) {
      result = result + v1_0r*v2_0r;
      v1_0r = (DOUBLE)V1[vecptr1++];
      v2_0r = (DOUBLE)V2[vecptr2++];    

      result = result + v1_0i*v2_0i;
      v1_0i = (DOUBLE)V1[vecptr1++];
      v2_0i = (DOUBLE)V2[vecptr2++];
      
      result = result + v1_1r*v2_1r;
      v1_1r = (DOUBLE)V1[vecptr1++];
      v2_1r = (DOUBLE)V2[vecptr2++];

      result = result + v1_1i*v2_1i;
      v1_1i =(DOUBLE)V1[vecptr1++];
      v2_1i = (DOUBLE)V2[vecptr2++];
      
      result = result + v1_2r*v2_2r;
      v1_2r = (DOUBLE)V1[vecptr1++];
      v2_2r = (DOUBLE)V2[vecptr2++];

      result = result + v1_2i*v2_2i;
      v1_2i = (DOUBLE)V1[vecptr1++];
      v2_2i = (DOUBLE)V2[vecptr2++];


    }

    // Last one plus drain...
    result = result + v1_0r*v2_0r;
    result = result + v1_0i*v2_0i;
    result = result + v1_1r*v2_1r;
    result = result + v1_1i*v2_1i;
    result = result + v1_2r*v2_2r;
    result = result + v1_2i*v2_2i;    

  }
  
  *Out=(DOUBLE)result;
}


} // namespace QDP;

#endif // guard
