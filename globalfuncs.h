// -*- C++ -*-
// $Id: globalfuncs.h,v 1.1 2002-09-12 18:22:16 edwards Exp $
//
// QDP data parallel interface
//
QDP_BEGIN_NAMESPACE(QDP);

//-----------------------------------------------------------------------------

//! Su2_extract: r_0,r_1,r_2,r_3 <- source(su2_index)  [SU(N) field]
/*! 
 * Extract components r_k proportional to SU(2) submatrix su2_index
 * from the "SU(Nc)" matrix V. The SU(2) matrix is parametrized in the
 * sigma matrix basis.
   *
   * There are Nc*(Nc-1)/2 unique SU(2) submatrices in an SU(Nc) matrix.
   * The user does not need to know exactly which one is which, just that
   * they are unique.
   */
template<class T, class C, class T1, class C1> inline
void su2_extract(QDPType<T,C>& r_0, QDPType<T,C>& r_1, QDPType<T,C>& r_2, QDPType<T,C>& r_3, 
		 int su2_index, QDPType<T1,C1>& source)
{
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
  su2_extract(static_cast<C&>(r_0), static_cast<C&>(r_1), 
	      static_cast<C&>(r_2), static_cast<C&>(r_3), 
	      static_cast<C1 &>(source), 
	      i1, i2);
}
  

//! Sun_fill: dest(su2_index) <- r_0,r_1,r_2,r_3
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
template<class T, class C, class T1, class C1> inline
void sun_fill(QDPType<T,C>& dest, 
	      int su2_index,
	      const QDPType<T1,C1>& r_0, const QDPType<T1,C1>& r_1, 
	      const QDPType<T1,C1>& r_2, const QDPType<T1,C1>& r_3)
{
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
  sun_fill(static_cast<C&>(dest), 
	   static_cast<const C1&>(r_0), static_cast<const C1&>(r_1), 
	   static_cast<const C1&>(r_2), static_cast<const C1&>(r_3), 
	   i1, i2);
}
  

// Spin projection
//! dest  = spinProject(source1) 
/*! Boneheaded simple implementation till I get a better one... */
template<class T, class C>
typename UnaryReturn<C, FnSpinProject>::Type_t
spinProject(const QDPType<T,C>& s1, int mu, int isign)
{
  typedef typename UnaryReturn<C, FnSpinProject>::Type_t  Ret_t;
  Ret_t  d;

  switch (isign)
  {
  case -1:
    switch (mu)
    {
    case 0:
      d = spinProjectDir0Minus(s1);
      break;
    case 1:
      d = spinProjectDir1Minus(s1);
      break;
    case 2:
      d = spinProjectDir2Minus(s1);
      break;
    case 3:
      d = spinProjectDir3Minus(s1);
      break;
    default:
      cerr << "Spin project: illegal direction\n";
      exit(1);
    }
    break;

  case +1:
    switch (mu)
    {
    case 0:
      d = spinProjectDir0Plus(s1);
      break;
    case 1:
      d = spinProjectDir1Plus(s1);
      break;
    case 2:
      d = spinProjectDir2Plus(s1);
      break;
    case 3:
      d = spinProjectDir3Plus(s1);
      break;
    default:
      cerr << "Spin project: illegal direction\n";
      exit(1);
    }
    break;

  default:
    cerr << "Spin project: isign must be pos or neg.\n";
    exit(1);
  }

  return d;
}


// Spin reconstruction
//! dest  = spinReconstruct(source1) 
/*! Boneheaded simple implementation till I get a better one... */
template<class T, class C>
typename UnaryReturn<C, FnSpinReconstruct>::Type_t
spinReconstruct(const QDPType<T,C>& s1, int mu, int isign)
{
  typedef typename UnaryReturn<C, FnSpinReconstruct>::Type_t  Ret_t;
  Ret_t  d;

  switch (isign)
  {
  case -1:
    switch (mu)
    {
    case 0:
      d = spinReconstructDir0Minus(s1);
      break;
    case 1:
      d = spinReconstructDir1Minus(s1);
      break;
    case 2:
      d = spinReconstructDir2Minus(s1);
      break;
    case 3:
      d = spinReconstructDir3Minus(s1);
      break;
    default:
      cerr << "Spin reconstruct: illegal direction\n";
      exit(1);
    }
    break;

  case +1:
    switch (mu)
    {
    case 0:
      d = spinReconstructDir0Plus(s1);
      break;
    case 1:
      d = spinReconstructDir1Plus(s1);
      break;
    case 2:
      d = spinReconstructDir2Plus(s1);
      break;
    case 3:
      d = spinReconstructDir3Plus(s1);
      break;
    default:
      cerr << "Spin reconstruct: illegal direction\n";
      exit(1);
    }
    break;

  default:
    cerr << "Spin reconstruct: isign must be pos or neg.\n";
    exit(1);
  }

  return d;
}



#if 0
//-----------------------------------------------------------------------------
/** @addtogroup group5
 * Simple unary functions that return the same type as the source
 *  @{
 */
//! QDP Shift return function: implements  dest(x) = s1(x+isign*dir)
/*!
 * shift(source,isign,dir)
 * isign = parity of direction (+1 or -1)
 * dir   = direction ([0,...,Nd-1])
 *
 * Implements:  dest(x) = s1(x+isign*dir)
 * There are cpp macros called  FORWARD and BACKWARD that are +1,-1 resp.
 * that are often used as arguments
 */
template<class T> QDPType<T> Shift(const QDPType<T>& s1, int isign, int dir) 
{
  QDPType<T> d;
  fprintf(stderr,"explicit Shift\n"); 
  d.elem().shift_rep(s1.elem(), isign, dir, global_context->Sub());
  return d;
}
/** @} */ // end of group5
#endif


QDP_END_NAMESPACE();

