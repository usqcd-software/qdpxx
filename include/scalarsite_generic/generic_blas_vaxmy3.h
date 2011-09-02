// $Id: generic_blas_vaxmy3.h,v 1.3 2009-09-15 20:48:41 bjoo Exp $

/*! @file
 *  @brief Generic Scalar VAXMY routine
 *
 */

#ifndef QDP_GENERIC_BLAS_VAXMY3
#define QDP_GENERIC_BLAS_VAXMY3

namespace QDP {

// (Vector) Out = (Scalar) (*scalep) * (Vector) InScale - (Vector) Add
inline
void vaxmy3(REAL *Out,REAL *scalep,REAL *InScale, REAL *Sub,int n_3vec)
{

  REAL a = (*scalep);
  int len = n_3vec * 24;
  for(int i=0; i < len; i++) { 
    Out[i] = a*InScale[i] - Sub[i];
  }

#if 0
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
  
  a = *scalep;
  
  register int index_x = 0;
  register int index_y = 0;
  register int index_z = 0;
  
  register int counter;
  int len = n_3vec*4; // 6 deep unrolled loop 
                      // multiply by 4 to get 24.

  for( counter = 0; counter < len; counter++) {
    x0r = (double)InScale[index_x++];
    y0r = (double)Sub[index_y++];
    z0r = a*x0r - y0r;
    Out[index_z++] = (REAL)z0r;
    
    x0i = (double)InScale[index_x++];
    y0i = (double)Sub[index_y++];
    z0i = a*x0i - y0i;
    Out[index_z++] = (REAL)z0i;
    
    x1r = (double)InScale[index_x++];
    y1r = (double)Sub[index_y++];
    z1r = a*x1r - y1r;
    Out[index_z++] = (REAL)z1r;
    
    x1i = (double)InScale[index_x++];
    y1i = (double)Sub[index_y++];
    z1i = a*x1i - y1i;
    Out[index_z++] = (REAL)z1i;
    
    x2r = (double)InScale[index_x++];     
    y2r = (double)Sub[index_y++];
    z2r = a*x2r - y2r;
    Out[index_z++] = (REAL)z2r;
    
    x2i = (double)InScale[index_x++];
    y2i = (double)Sub[index_y++];
    z2i = a*x2i - y2i;  
    Out[index_z++] = (REAL)z2i;
  }
#endif

}

} // namespace QDP;

#endif // guard
