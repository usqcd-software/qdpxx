// $Id: qdp.cc,v 1.3 2002-10-28 03:08:44 edwards Exp $
//
// QDP data parallel interface
//

#include "qdp.h"

QDP_BEGIN_NAMESPACE(QDP);

//! General death routine
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


//! Calculates the lexicographic site index from the coordinate of a site
/*! 
 * Nothing specific about the actual lattice size, can be used for 
 * any kind of latt size 
 */
int local_site(const multi1d<int>& coord, const multi1d<int>& latt_size)
{
  int order = 0;

  for(int mmu=latt_size.size()-1; mmu >= 1; --mmu)
    order = latt_size[mmu-1]*(coord[mmu] + order);

  order += coord[0];

  return order;
}



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
su2Extract(const LatticeGauge& source, 
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
    SZ_ERROR("Trouble with SU2 subgroup index");

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
LatticeGauge
sunFill(const multi1d<LatticeReal> r,
	     int su2_index,
	     const Subset& s)
{
  LatticeGauge dest;

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
    SZ_ERROR("Trouble with SU2 subgroup index");

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
