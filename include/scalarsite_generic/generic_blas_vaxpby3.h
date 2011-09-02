// $Id: generic_blas_vaxpby3.h,v 1.3 2009-09-15 20:48:41 bjoo Exp $

/*! @file
 *  @brief Generic Scalar VAXPBY routine
 *
 */

#ifndef QDP_GENERIC_BLAS_VAXPBY3
#define QDP_GENERIC_BLAS_VAXPBY3

namespace QDP {

// (Vector) Out = (Scalar) (*ap) * (Vector)xp + (Scalar)(*bp) * (Vector)yp
inline
void vaxpby3(REAL *Out,REAL *ap ,REAL *xp, REAL *bp,  REAL *yp ,int n_3vec)
{
  REAL a = (*ap);
  REAL b = (*bp);
  int len = 24*n_3vec;
  for(int i=0; i < len; i++) { 
    Out[i] = a*xp[i] + b*yp[i];
  }
#if 0
  register double a;
  register double b;

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
  
  a = *ap;
  b = *bp;

  register int index_x = 0;
  register int index_y = 0;
  register int index_z = 0;
  
  register int counter;
  int len = 4*n_3vec;

  for( counter = 0; counter < len; counter++) {
    x0r = (double)xp[index_x++];
    y0r = (double)yp[index_y++];
    z0r = a*x0r;
    z0r = z0r + b*y0r;
    Out[index_z++] =(REAL) z0r;
    
    x0i = (double)xp[index_x++];
    y0i = (double)yp[index_y++];
    z0i = a*x0i;
    z0i = z0i + b*y0i;
    Out[index_z++] =(REAL) z0i;
    
    x1r = (double)xp[index_x++];
    y1r = (double)yp[index_y++];
    z1r = a*x1r;
    z1r = z1r + b*y1r;
    Out[index_z++] = (REAL)z1r;
    
    x1i = (double)xp[index_x++];
    y1i = (double)yp[index_y++];
    z1i = a*x1i;
    z1i = z1i + b*y1i;

    Out[index_z++] = (REAL)z1i;
    
    x2r = (double)xp[index_x++];     
    y2r = (double)yp[index_y++];
    z2r = a*x2r;
    z2r = z2r + b*y2r;
    Out[index_z++] = (REAL)z2r;
    
    x2i = (double)xp[index_x++];
    y2i = (double)yp[index_y++];
    z2i = a*x2i;
    z2i = z2i + b*y2i;
    Out[index_z++] = (REAL)z2i;
  }
#endif
}


} // namespace QDP;

#endif // guard
