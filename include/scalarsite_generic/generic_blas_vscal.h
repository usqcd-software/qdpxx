// $Id: generic_blas_vscal.h,v 1.2 2007-06-10 14:32:10 edwards Exp $

/*! @file
 *  @brief Generic Scalar VSCAL routine
 *
 */

#ifndef QDP_GENERIC_BLAS_VSCAL
#define QDP_GENERIC_BLAS_VSCAL

namespace QDP {

// (Vector) out = (Scalar) (*scalep) * (Vector) In
inline
void vscal(REAL *Out, REAL *scalep, REAL *In, int n_3vec)
{
  register double a = *scalep;

  register double i0r;
  register double i0i;
  register double i1r;
  register double i1i;
  register double i2r;
  register double i2i;

  register double o0r;
  register double o0i;
  register double o1r;
  register double o1i;
  register double o2r;
  register double o2i;

  register int counter=0;
  register int inptr=0;
  register int outptr=0;

  if( n_3vec > 0 ) {
    i0r = (double)In[inptr++];
    i0i = (double)In[inptr++];
    i1r = (double)In[inptr++];
    for(counter = 0; counter < n_3vec-1 ; counter++) {
      o0r = a*i0r;
      Out[outptr++] = (REAL)o0r;
      
      i1i = (double)In[inptr++];
      i2r = (double)In[inptr++];
      o0i = a*i0i;
      Out[outptr++] = (REAL)o0i;
      
      i2i = (double)In[inptr++];
      i0r = (double)In[inptr++];
      o1r = a*i1r;
      Out[outptr++] = (REAL)o1r;
      
      i0i = (double)In[inptr++];
      i1r = (double)In[inptr++]; // Last prefetched
      
      o1i = a*i1i;
      Out[outptr++] = (REAL)o1i;
      
      o2r= a*i2r;
      Out[outptr++] = (REAL)o2r;
      
      o2i= a*i2i;
      Out[outptr++] = (REAL)o2i;
    }

    o0r = a*i0r;
    Out[outptr++] =(REAL) o0r;
    
    i1i = (double)In[inptr++];
    i2r = (double)In[inptr++];
    o0i = a*i0i;
    Out[outptr++] = (REAL)o0i;
    
    i2i = (double)In[inptr++];
    o1r = a*i1r;
    Out[outptr++] = (REAL)o1r;
    
    o1i = a*i1i;
    Out[outptr++] = (REAL)o1i;
    
    o2r= a*i2r;
    Out[outptr++] = (REAL)o2r;
    
    o2i= a*i2i;
    Out[outptr++] = (REAL)o2i;
    
  }
}  

} // namespace QDP;

#endif // guard
