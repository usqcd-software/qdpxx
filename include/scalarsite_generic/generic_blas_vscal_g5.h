// $Id: generic_blas_vscal_g5.h,v 1.2 2007-06-10 14:32:10 edwards Exp $

/*! @file
 *  @brief Generic Scalar VSCAL routine
 *
 */

#ifndef QDP_GENERIC_BLAS_VSCAL_G5
#define QDP_GENERIC_BLAS_VSCAL_G5

namespace QDP {

// (Vector) out = (Scalar) (*scalep) * P+ (Vector) In
inline
void scal_g5ProjPlus(REAL *Out, REAL *scalep, REAL *In, int n_4vec)
{
  register double a;
  register double x0r;
  register double x0i;
  
  register double x1r;
  register double x1i;
  
  register double x2r;
  register double x2i;
  
  register double z0r;
  register double z0i;
  
  register double z1r;
  register double z1i;
  
  register double z2r;
  register double z2i;
  
  a = *scalep;
  
  register int index_x = 0;
  register int index_z = 0;
  
  register int counter;
  
  for( counter = 0; counter < n_4vec; counter++) {
    // Spin Component 0
    x0r = (double)In[index_x++];
    z0r = a*x0r;
    Out[index_z++] =(REAL) z0r;
    
    x0i = (double)In[index_x++];
    z0i = a*x0i;
    Out[index_z++] =(REAL) z0i;
    
    x1r = (double)In[index_x++];
    z1r = a*x1r;
    Out[index_z++] = (REAL)z1r;
    
    x1i = (double)In[index_x++];
    z1i = a*x1i;
    Out[index_z++] = (REAL)z1i;
    
    x2r = (double)In[index_x++];     
    z2r = a*x2r;
    Out[index_z++] = (REAL)z2r;
    
    x2i = (double)In[index_x++];
    z2i = a*x2i;
    Out[index_z++] = (REAL)z2i;

    // Spin Component 1
    x0r = (double)In[index_x++];
    z0r = a*x0r;
    Out[index_z++] =(REAL) z0r;
    
    x0i = (double)In[index_x++];
    z0i = a*x0i;
    Out[index_z++] =(REAL) z0i;
    
    x1r = (double)In[index_x++];
    z1r = a*x1r;
    Out[index_z++] = (REAL)z1r;
    
    x1i = (double)In[index_x++];
    z1i = a*x1i;
    Out[index_z++] = (REAL)z1i;
    
    x2r = (double)In[index_x++];     
    z2r = a*x2r;
    Out[index_z++] = (REAL)z2r;
    
    x2i = (double)In[index_x++];
    z2i = a*x2i;
    Out[index_z++] = (REAL)z2i;

    index_x+=12;

    Out[index_z++] = (REAL)0;
    Out[index_z++] = (REAL)0;
    Out[index_z++] = (REAL)0;
    Out[index_z++] = (REAL)0;
    Out[index_z++] = (REAL)0;
    Out[index_z++] = (REAL)0;
    
    Out[index_z++] = (REAL)0;
    Out[index_z++] = (REAL)0;
    Out[index_z++] = (REAL)0;
    Out[index_z++] = (REAL)0;
    Out[index_z++] = (REAL)0;
    Out[index_z++] = (REAL)0;

  }
}  

// (Vector) out = (Scalar) (*scalep) * P- (Vector) In
inline
void scal_g5ProjMinus(REAL *Out, REAL *scalep, REAL *In, int n_4vec)
{
  register double a;
  register double x0r;
  register double x0i;
  
  register double x1r;
  register double x1i;
  
  register double x2r;
  register double x2i;
  
  register double z0r;
  register double z0i;
  
  register double z1r;
  register double z1i;
  
  register double z2r;
  register double z2i;
  
  a = *scalep;
  
  register int index_x = 0;
  register int index_z = 0;
  
  register int counter;
  
  for( counter = 0; counter < n_4vec; counter++) {
    index_x+=12;

    // Spin Component 0
    Out[index_z++] = (REAL)0;
    Out[index_z++] = (REAL)0;
    Out[index_z++] = (REAL)0;
    Out[index_z++] = (REAL)0;
    Out[index_z++] = (REAL)0;
    Out[index_z++] = (REAL)0;
    
    // Spin Component 1
    Out[index_z++] = (REAL)0;
    Out[index_z++] = (REAL)0;
    Out[index_z++] = (REAL)0;
    Out[index_z++] = (REAL)0;
    Out[index_z++] = (REAL)0;
    Out[index_z++] = (REAL)0;


    // Spin Component 2
    x0r = (double)In[index_x++];
    z0r = a*x0r;
    Out[index_z++] =(REAL) z0r;
    
    x0i = (double)In[index_x++];
    z0i = a*x0i;
    Out[index_z++] =(REAL) z0i;
    
    x1r = (double)In[index_x++];
    z1r = a*x1r;
    Out[index_z++] = (REAL)z1r;
    
    x1i = (double)In[index_x++];
    z1i = a*x1i;
    Out[index_z++] = (REAL)z1i;
    
    x2r = (double)In[index_x++];     
    z2r = a*x2r;
    Out[index_z++] = (REAL)z2r;
    
    x2i = (double)In[index_x++];
    z2i = a*x2i;
    Out[index_z++] = (REAL)z2i;

    // Spin Component 3
    x0r = (double)In[index_x++];
    z0r = a*x0r;
    Out[index_z++] =(REAL) z0r;
    
    x0i = (double)In[index_x++];
    z0i = a*x0i;
    Out[index_z++] =(REAL) z0i;
    
    x1r = (double)In[index_x++];
    z1r = a*x1r;
    Out[index_z++] = (REAL)z1r;
    
    x1i = (double)In[index_x++];
    z1i = a*x1i;
    Out[index_z++] = (REAL)z1i;
    
    x2r = (double)In[index_x++];     
    z2r = a*x2r;
    Out[index_z++] = (REAL)z2r;
    
    x2i = (double)In[index_x++];
    z2i = a*x2i;
    Out[index_z++] = (REAL)z2i;


  }
}  

} // namespace QDP;

#endif // guard
