// $Id: qdp.cc,v 1.6 2002-12-05 21:27:10 edwards Exp $
//
// QDP data parallel interface
//

#include "qdp.h"

QDP_BEGIN_NAMESPACE(QDP);

//-----------------------------------------------------------------------------
//! Su2_extract: r_0,r_1,r_2,r_3 <- source(su2_index)  [SU(N) field]  under a subset
/*! 
 * Extract components r_k proportional to SU(2) submatrix su2_index
 * from the "SU(Nc)" matrix V. The SU(2) matrix is parametrized in the
 * sigma matrix basis.
 *
 * There are Nc*(Nc-1)/2 unique SU(2) submatrices in an SU(Nc) matrix.
 * The user does not need to know exactly which one is which, just that
 * they are unique.
 */
multi1d<LatticeReal> 
su2Extract(const LatticeColorMatrix& source, 
	   int su2_index, 
	   const Subset& s)
{
  multi1d<LatticeReal> r(4);

  /* Determine the SU(N) indices corresponding to the SU(2) indices */
  /* of the SU(2) subgroup $3 */
  int i1, i2;
  int found = 0;
  int del_i = 0;
  int index = -1;

  while ( del_i < (Nc-1) && found == 0 )
  {
    del_i++;
    for ( i1 = 0; i1 < (Nc-del_i); i1++ )
    {
      index++;
      if ( index == su2_index )
      {
	found = 1;
	break;
      }
    }
  }
  i2 = i1 + del_i;

  if ( found == 0 )
    QDP_error_exit("Trouble with SU2 subgroup index");

  /* Compute the b(k) of A_SU(2) = b0 + i sum_k bk sigma_k */ 
  r[0](s) = real(peekColor(source,i1,i1)) + real(peekColor(source,i2,i2));
  r[1](s) = imag(peekColor(source,i1,i2)) + imag(peekColor(source,i2,i1));
  r[2](s) = real(peekColor(source,i1,i2)) - real(peekColor(source,i2,i1));
  r[3](s) = imag(peekColor(source,i1,i1)) - imag(peekColor(source,i2,i2));

  return r;
}
  

//-----------------------------------------------
//! Sun_fill: dest(su2_index) <- r_0,r_1,r_2,r_3  under a subset
/*!
 * Fill an SU(Nc) matrix V with the SU(2) submatrix su2_index
 * paramtrized by b_k in the sigma matrix basis.
 *
 * Fill in B from B_SU(2) = b0 + i sum_k bk sigma_k
 *
 * There are Nc*(Nc-1)/2 unique SU(2) submatrices in an SU(Nc) matrix.
 * The user does not need to know exactly which one is which, just that
 * they are unique.
 */
LatticeColorMatrix
sunFill(const multi1d<LatticeReal> r,
	     int su2_index,
	     const Subset& s)
{
  LatticeColorMatrix dest;

  /* Determine the SU(N) indices corresponding to the SU(2) indices */
  /* of the SU(2) subgroup $3 */
  int i1, i2;
  int found = 0;
  int del_i = 0;
  int index = -1;

  while ( del_i < (Nc-1) && found == 0 )
  {
    del_i++;
    for ( i1 = 0; i1 < (Nc-del_i); i1++ )
    {
      index++;
      if ( index == su2_index )
      {
	found = 1;
	break;
      }
    }
  }
  i2 = i1 + del_i;

  if ( found == 0 )
    QDP_error_exit("Trouble with SU2 subgroup index");

  /* 
   * Insert the b(k) of A_SU(2) = b0 + i sum_k bk sigma_k 
   * back into the SU(N) matrix
   */ 
  dest(s) = 1.0;

  pokeColor(dest(s), cmplx( r[0], r[3]), i1, i1);
  pokeColor(dest(s), cmplx( r[2], r[1]), i1, i2);
  pokeColor(dest(s), cmplx(-r[2], r[1]), i2, i1);
  pokeColor(dest(s), cmplx( r[0],-r[3]), i2, i2);

  return dest;
}
  

QDP_END_NAMESPACE();
