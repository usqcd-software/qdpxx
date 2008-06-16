// $Id: sse_blas_local_vcdot_double.h,v 1.1 2008-06-16 20:09:06 bjoo Exp $

/*! @file
 *  @brief Generic Scalar VAXPY routine
 *
 */

#ifndef QDP_SSE_BLAS_LOCAL_VCDOT_DOUBLE
#define QDP_SSE_BLAS_LOCAL_VCDOT_DOUBLE

namespace QDP {

#include <xmmintrin.h>
#include <iostream>
  using namespace std;

  // Re < y^\dag , x > 
  // =  sum ( y.re x.re + y.im x.im )
  // =  sum ( y.re x.re ) + sum( y.im x.im )
  //
  // Load in [ x.re | x.im ]
  //         [ y.re | y.im ]
  // Make    [ x.re y.re | x.im y.im ]
  // accumulate sum
  // 
  // At the end do a single crossing:
  //
  //       [ sum (x.re y.re) | sum(x.im y.im) ]
  //     + [ sum (x.im y.im) | sum(x.re y.re) ]
  //  =    [ innerProdReal   | innerProdReal  ]
  // 
  // then srore either half.
inline
  void local_vcdot4(REAL64 *sum, REAL64 *y, REAL64* x,int n_4spin)
{
  __m128d sum_real1;
  __m128d sum_imag1;

  __m128d sum_real2;
  __m128d sum_imag2;

  __m128d tmp1;
  __m128d tmp2;
  __m128d tmp3;
  __m128d tmp4;

  __m128d tmp5;
  __m128d tmp6;
  __m128d tmp7;
  __m128d tmp8;

  __m128d tmp9;
  __m128d tmp10;

  typedef union
  {
    double d[2];
    __m128d xmm;
  } vd;

  vd sign  = { (double)1,(double)-1};

  // Zero out sums
  sum_real1 = _mm_xor_pd(sum_real1, sum_real1); 
  sum_imag1 = _mm_xor_pd(sum_imag1, sum_imag1); 

  sum_real2 = _mm_xor_pd(sum_real2, sum_real2); 
  sum_imag2 = _mm_xor_pd(sum_imag2, sum_imag2); 
  
  double *x_p=x;
  double *y_p=y;


  for(int i=0; i < n_4spin; i++) { 
      
    tmp1 = _mm_load_pd(x_p);  // tmp1 = x
    tmp2 = _mm_load_pd(y_p);  // tmp2 = y
      
    // tmp3 = [ y_r x_r , y_i x_i ]
    tmp3 = _mm_mul_pd(tmp2, tmp1);
    
    // Accumulate: [ sum y_r x_r, sum y_i x_i ]
    sum_real1 = _mm_add_pd(sum_real1, tmp3);
    
    // Negate y_i: tmp2 = [ y_r, -y_i ]
    tmp2 = _mm_mul_pd(tmp2, sign.xmm);
    
    // Cross x:    tmp3 = [ x_i,  x_r ]
    tmp3 = _mm_shuffle_pd(tmp1, tmp1, 0x1);
    
    // Tmp 4    [ y_r x_i, - y_i x_r ]
    tmp4 = _mm_mul_pd(tmp2, tmp3);
    
    // Accumulate: [ sum y_r x i, -sum y_i x_r ]
    sum_imag1 = _mm_add_pd(sum_imag1, tmp4);
    
    
    tmp5 = _mm_load_pd(x_p+2);  // tmp1 = x
    tmp6 = _mm_load_pd(y_p+2);  // tmp2 = y
    
    // tmp3 = [ y_r x_r , y_i x_i ]
    tmp7 = _mm_mul_pd(tmp6, tmp5);
    
    // Accumulate: [ sum y_r x_r, sum y_i x_i ]
    sum_real2 = _mm_add_pd(sum_real2, tmp7);
    
    // Negate y_i: tmp2 = [ y_r, -y_i ]
    tmp6 = _mm_mul_pd(tmp6, sign.xmm);
    
    // Cross x:    tmp3 = [ x_i,  x_r ]
    tmp7 = _mm_shuffle_pd(tmp5, tmp5, 0x1);
    
    // Tmp 4    [ y_r x_i, - y_i x_r ]
    tmp8 = _mm_mul_pd(tmp6, tmp7);
    
    // Accumulate: [ sum y_r x i, -sum y_i x_r ]
    sum_imag2 = _mm_add_pd(sum_imag2, tmp8);
    
    tmp1 = _mm_load_pd(x_p+4);  // tmp1 = x
    tmp2 = _mm_load_pd(y_p+4);  // tmp2 = y
    
    // tmp3 = [ y_r x_r , y_i x_i ]
    tmp3 = _mm_mul_pd(tmp2, tmp1);
    
    // Accumulate: [ sum y_r x_r, sum y_i x_i ]
    sum_real1 = _mm_add_pd(sum_real1, tmp3);
    
    // Negate y_i: tmp2 = [ y_r, -y_i ]
    tmp2 = _mm_mul_pd(tmp2, sign.xmm);
    
    // Cross x:    tmp3 = [ x_i,  x_r ]
    tmp3 = _mm_shuffle_pd(tmp1, tmp1, 0x1);
    
    // Tmp 4    [ y_r x_i, - y_i x_r ]
    tmp4 = _mm_mul_pd(tmp2, tmp3);
    
    // Accumulate: [ sum y_r x i, -sum y_i x_r ]
    sum_imag1 = _mm_add_pd(sum_imag1, tmp4);
    
    
    tmp5 = _mm_load_pd(x_p+6);  // tmp1 = x
    tmp6 = _mm_load_pd(y_p+6);  // tmp2 = y
    
    // tmp3 = [ y_r x_r , y_i x_i ]
    tmp7 = _mm_mul_pd(tmp6, tmp5);
    
    // Accumulate: [ sum y_r x_r, sum y_i x_i ]
    sum_real2 = _mm_add_pd(sum_real2, tmp7);
    
    // Negate y_i: tmp2 = [ y_r, -y_i ]
    tmp6 = _mm_mul_pd(tmp6, sign.xmm);
    
    // Cross x:    tmp3 = [ x_i,  x_r ]
    tmp7 = _mm_shuffle_pd(tmp5, tmp5, 0x1);
    
    // Tmp 4    [ y_r x_i, - y_i x_r ]
    tmp8 = _mm_mul_pd(tmp6, tmp7);
    
    // Accumulate: [ sum y_r x i, -sum y_i x_r ]
    sum_imag2 = _mm_add_pd(sum_imag2, tmp8);
    
    
    
    tmp1 = _mm_load_pd(x_p+8);  // tmp1 = x
    tmp2 = _mm_load_pd(y_p+8);  // tmp2 = y
    
    // tmp3 = [ y_r x_r , y_i x_i ]
    tmp3 = _mm_mul_pd(tmp2, tmp1);
    
    // Accumulate: [ sum y_r x_r, sum y_i x_i ]
    sum_real1 = _mm_add_pd(sum_real1, tmp3);
    
    // Negate y_i: tmp2 = [ y_r, -y_i ]
    tmp2 = _mm_mul_pd(tmp2, sign.xmm);
    
    // Cross x:    tmp3 = [ x_i,  x_r ]
    tmp3 = _mm_shuffle_pd(tmp1, tmp1, 0x1);
    
    // Tmp 4    [ y_r x_i, - y_i x_r ]
    tmp4 = _mm_mul_pd(tmp2, tmp3);
    
    // Accumulate: [ sum y_r x i, -sum y_i x_r ]
    sum_imag1 = _mm_add_pd(sum_imag1, tmp4);
    
    
    tmp5 = _mm_load_pd(x_p+10);  // tmp1 = x
    tmp6 = _mm_load_pd(y_p+10);  // tmp2 = y
    
    // tmp3 = [ y_r x_r , y_i x_i ]
    tmp7 = _mm_mul_pd(tmp6, tmp5);
    
    // Accumulate: [ sum y_r x_r, sum y_i x_i ]
    sum_real2 = _mm_add_pd(sum_real2, tmp7);
    
    // Negate y_i: tmp2 = [ y_r, -y_i ]
    tmp6 = _mm_mul_pd(tmp6, sign.xmm);
    
    // Cross x:    tmp3 = [ x_i,  x_r ]
    tmp7 = _mm_shuffle_pd(tmp5, tmp5, 0x1);
    
    // Tmp 4    [ y_r x_i, - y_i x_r ]
    tmp8 = _mm_mul_pd(tmp6, tmp7);
    
    // Accumulate: [ sum y_r x i, -sum y_i x_r ]
    sum_imag2 = _mm_add_pd(sum_imag2, tmp8);
    
    tmp1 = _mm_load_pd(x_p+12);  // tmp1 = x
    tmp2 = _mm_load_pd(y_p+12);  // tmp2 = y
    
    // tmp3 = [ y_r x_r , y_i x_i ]
    tmp3 = _mm_mul_pd(tmp2, tmp1);
    
    // Accumulate: [ sum y_r x_r, sum y_i x_i ]
    sum_real1 = _mm_add_pd(sum_real1, tmp3);
    
    // Negate y_i: tmp2 = [ y_r, -y_i ]
    tmp2 = _mm_mul_pd(tmp2, sign.xmm);
    
    // Cross x:    tmp3 = [ x_i,  x_r ]
    tmp3 = _mm_shuffle_pd(tmp1, tmp1, 0x1);
    
    // Tmp 4    [ y_r x_i, - y_i x_r ]
    tmp4 = _mm_mul_pd(tmp2, tmp3);
    
    // Accumulate: [ sum y_r x i, -sum y_i x_r ]
    sum_imag1 = _mm_add_pd(sum_imag1, tmp4);
    
    
    tmp5 = _mm_load_pd(x_p+14);  // tmp1 = x
    tmp6 = _mm_load_pd(y_p+14);  // tmp2 = y
    
    // tmp3 = [ y_r x_r , y_i x_i ]
    tmp7 = _mm_mul_pd(tmp6, tmp5);
    
    // Accumulate: [ sum y_r x_r, sum y_i x_i ]
    sum_real2 = _mm_add_pd(sum_real2, tmp7);
    
    // Negate y_i: tmp2 = [ y_r, -y_i ]
    tmp6 = _mm_mul_pd(tmp6, sign.xmm);
    
    // Cross x:    tmp3 = [ x_i,  x_r ]
    tmp7 = _mm_shuffle_pd(tmp5, tmp5, 0x1);
    
    // Tmp 4    [ y_r x_i, - y_i x_r ]
    tmp8 = _mm_mul_pd(tmp6, tmp7);
    
    // Accumulate: [ sum y_r x i, -sum y_i x_r ]
    sum_imag2 = _mm_add_pd(sum_imag2, tmp8);
    
    
    
    tmp1 = _mm_load_pd(x_p+16);  // tmp1 = x
    tmp2 = _mm_load_pd(y_p+16);  // tmp2 = y
    
    // tmp3 = [ y_r x_r , y_i x_i ]
    tmp3 = _mm_mul_pd(tmp2, tmp1);
    
    // Accumulate: [ sum y_r x_r, sum y_i x_i ]
    sum_real1 = _mm_add_pd(sum_real1, tmp3);
    
    // Negate y_i: tmp2 = [ y_r, -y_i ]
    tmp2 = _mm_mul_pd(tmp2, sign.xmm);
    
    // Cross x:    tmp3 = [ x_i,  x_r ]
    tmp3 = _mm_shuffle_pd(tmp1, tmp1, 0x1);
    
    // Tmp 4    [ y_r x_i, - y_i x_r ]
    tmp4 = _mm_mul_pd(tmp2, tmp3);
    
    // Accumulate: [ sum y_r x i, -sum y_i x_r ]
    sum_imag1 = _mm_add_pd(sum_imag1, tmp4);
    
    
    tmp5 = _mm_load_pd(x_p+18);  // tmp1 = x
    tmp6 = _mm_load_pd(y_p+18);  // tmp2 = y
    
    // tmp3 = [ y_r x_r , y_i x_i ]
    tmp7 = _mm_mul_pd(tmp6, tmp5);
    
    // Accumulate: [ sum y_r x_r, sum y_i x_i ]
    sum_real2 = _mm_add_pd(sum_real2, tmp7);
    
    // Negate y_i: tmp2 = [ y_r, -y_i ]
    tmp6 = _mm_mul_pd(tmp6, sign.xmm);
    
    // Cross x:    tmp3 = [ x_i,  x_r ]
    tmp7 = _mm_shuffle_pd(tmp5, tmp5, 0x1);
    
    // Tmp 4    [ y_r x_i, - y_i x_r ]
    tmp8 = _mm_mul_pd(tmp6, tmp7);
    
    // Accumulate: [ sum y_r x i, -sum y_i x_r ]
    sum_imag2 = _mm_add_pd(sum_imag2, tmp8);
    
    tmp1 = _mm_load_pd(x_p+20);  // tmp1 = x
    tmp2 = _mm_load_pd(y_p+20);  // tmp2 = y
    
    // tmp3 = [ y_r x_r , y_i x_i ]
    tmp3 = _mm_mul_pd(tmp2, tmp1);
    
    // Accumulate: [ sum y_r x_r, sum y_i x_i ]
    sum_real1 = _mm_add_pd(sum_real1, tmp3);
    
    // Negate y_i: tmp2 = [ y_r, -y_i ]
    tmp2 = _mm_mul_pd(tmp2, sign.xmm);
    
    // Cross x:    tmp3 = [ x_i,  x_r ]
    tmp3 = _mm_shuffle_pd(tmp1, tmp1, 0x1);
    
    // Tmp 4    [ y_r x_i, - y_i x_r ]
    tmp4 = _mm_mul_pd(tmp2, tmp3);
    
    // Accumulate: [ sum y_r x i, -sum y_i x_r ]
    sum_imag1 = _mm_add_pd(sum_imag1, tmp4);
    
    
    tmp5 = _mm_load_pd(x_p+22);  // tmp1 = x
    tmp6 = _mm_load_pd(y_p+22);  // tmp2 = y
    
    // tmp3 = [ y_r x_r , y_i x_i ]
    tmp7 = _mm_mul_pd(tmp6, tmp5);
    
    // Accumulate: [ sum y_r x_r, sum y_i x_i ]
    sum_real2 = _mm_add_pd(sum_real2, tmp7);
    
    // Negate y_i: tmp2 = [ y_r, -y_i ]
    tmp6 = _mm_mul_pd(tmp6, sign.xmm);
    
    // Cross x:    tmp3 = [ x_i,  x_r ]
    tmp7 = _mm_shuffle_pd(tmp5, tmp5, 0x1);
    
    // Tmp 4    [ y_r x_i, - y_i x_r ]
    tmp8 = _mm_mul_pd(tmp6, tmp7);
    
    // Accumulate: [ sum y_r x i, -sum y_i x_r ]
    sum_imag2 = _mm_add_pd(sum_imag2, tmp8);
    
    

    x_p+=24; y_p+=24;
  }


  // Collect the sums
  sum_real1 = _mm_add_pd(sum_real1,sum_real2);
  sum_imag1 = _mm_add_pd(sum_imag1, sum_imag2);

  // Cross and add the real part
  tmp1 = _mm_shuffle_pd(sum_real1, sum_real1, 0x1);
  sum_real1 = _mm_add_pd(sum_real1, tmp1);
  

  // Cross and add the imag part
  tmp2 = _mm_shuffle_pd(sum_imag1, sum_imag1, 0x1);
  sum_imag1 = _mm_add_pd(sum_imag1, tmp2);

  // Take top half of sum1 and bottom half of sum2
  sum_real1 = _mm_shuffle_pd(sum_real1, sum_imag1, 0x2);

  // Single store
  _mm_store_pd(sum,sum_real1);
  
}



} // namespace QDP;

#endif // guard
