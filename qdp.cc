// $Id: qdp.cc,v 1.1 2002-09-12 18:22:16 edwards Exp $
//
// QDP data parallel interface
//

#include "qdp.h"

QDP_BEGIN_NAMESPACE(QDP);

//! Hack a-rooney - for now barf on boolean subset representation - need better method 
void diefunc() {fprintf(stderr,"Boolean rep not implemented\n");exit(1);}

//! Another Hack a-rooney - for now barf on boolean subset representation - need better method 
void SZ_ERROR(const char *s, ...) {fprintf(stderr,"%s\n",s);exit(1);}
  

//! Decompose a lexicographic site into coordinates
multi1d<int> crtesn(int ipos, const multi1d<int>& latt_size)
{
  multi1d<int> coord(latt_size.size());

  /* Calculate the Cartesian coordinates of the VALUE of IPOS where the 
   * value is defined by
   *
   *     for i = 0 to NDIM-1  {
   *        X_i  <- mod( IPOS, L(i) )
   *        IPOS <- int( IPOS / L(i) )
   *     }
   *
   * NOTE: here the coord(i) and IPOS have their origin at 0. 
   */
  for(int i=0; i < latt_size.size(); ++i)
  {
    coord[i] = ipos % latt_size[i];
    ipos = ipos / latt_size[i];
  }

  return coord;
}

QDP_END_NAMESPACE();
