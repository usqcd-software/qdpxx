// $Id: generic_blas_local_sumsq.h,v 1.2 2007-06-10 14:32:10 edwards Exp $

/*! @file
 *  @brief Generic Scalar, local sum squared routine
 *
 */

#ifndef QDP_GENERIC_BLAS_LOCAL_SUMSQ
#define QDP_GENERIC_BLAS_LOCAL_SUMSQ


namespace QDP {

// (Double) (*out) = || (Vector) In ||^2 (local to node)
inline
void local_sumsq(DOUBLE *Out, REAL *In, int n_3vec)
{
  register double result;
  
  register double i1;
  register double i2;
  register double i3;
  register double i4;
  register double i5;
  register double i6;
  
  int counter;
  int vecptr=0;
  result = 0;

  if( n_3vec > 0 ) { 
    i1 = (double)In[vecptr++];
    i2 = (double)In[vecptr++];
    result = i1*i1 + result;
    for(counter=0; counter < n_3vec-1; counter++) {
      i3 = (double)In[vecptr++];
      result = i2*i2 + result;
      i4 = (double)In[vecptr++];
      result = i3*i3 + result;
      i5 = (double)In[vecptr++];
      result = i4*i4 + result;
      i6 = (double)In[vecptr++];
      result = i5*i5 + result;
      i1 = (double)In[vecptr++];
      result = i6*i6 + result;
      i2 = (double)In[vecptr++];
      result = i1*i1 + result;
    }
    
    i3 = (double)In[vecptr++];
    result = i2*i2 + result;
    i4 = (double)In[vecptr++];
    result = i3*i3 + result;
    i5 = (double)In[vecptr++];
    result = i4*i4 + result;
    i6 = (double)In[vecptr++];
    result = i5*i5 + result;
    result = i6*i6 + result;
  }
  
  *Out=(DOUBLE)result;
}


} // namespace QDP;

#endif // guard
