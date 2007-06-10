// $Id: generic_blas_vaxpy3_norm.h,v 1.2 2007-06-10 14:32:10 edwards Exp $

/*! @file
 *  @brief Generic Scalar VAXPY NORM  routine
 *
 */

#ifndef QDP_GENERIC_BLAS_VAXPY3_NORM
#define QDP_GENERIC_BLAS_VAXPY3_NORM

namespace QDP {

inline
void vaxpy3_norm(REAL *Out,REAL *scalep,REAL *InScale, REAL *Add,int n_3vec, REAL *norm)
{
  register double a;
  register double x0r;
  register double x0i;
  
  register double x1r;
  register double x1i;
  
  register double x2r;
  register double x2i;
  
  register double y0r;
  register double y0i;
  
  register double y1r;
  register double y1i;
  
  register double y2r;
  register double y2i;
  
  register double z0r;
  register double z0i;
  
  register double z1r;
  register double z1i;
  
  register double z2r;
  register double z2i;
  register double norm_out=0;

  a = *scalep;
  
  register int index_x = 0;
  register int index_y = 0;
  register int index_z = 0;
  
  register int counter;
  
  for( counter = 0; counter < n_3vec; counter++) {
    x0r = (double)InScale[index_x++];
    y0r = (double)Add[index_y++];
    z0r = a*x0r + y0r;
    norm_out += z0r*z0r;
    Out[index_z++] =(REAL) z0r;
    
    x0i = (double)InScale[index_x++];
    y0i = (double)Add[index_y++];
    z0i = a*x0i + y0i;
    norm_out += z0i*z0i;
    Out[index_z++] =(REAL) z0i;
    

    x1r = (double)InScale[index_x++];
    y1r = (double)Add[index_y++];
    z1r = a*x1r + y1r;
    norm_out += z1r * z1r; 
    Out[index_z++] = (REAL)z1r;
    
    x1i = (double)InScale[index_x++];
    y1i = (double)Add[index_y++];
    z1i = a*x1i + y1i;
    norm_out += z1i*z1i;
    Out[index_z++] = (REAL)z1i;
    
    x2r = (double)InScale[index_x++];     
    y2r = (double)Add[index_y++];
    z2r = a*x2r + y2r;
    norm_out += z2r*z2r;
    Out[index_z++] = (REAL)z2r;
    
    x2i = (double)InScale[index_x++];
    y2i = (double)Add[index_y++];
    z2i = a*x2i + y2i;  
    norm_out += z2i*z2i;
    Out[index_z++] = (REAL)z2i;
    
  }
  *norm=(REAL)norm_out;
}

} // namespace QDP;

#endif // guard
