// -*- C++ -*-
// $Id: globalfuncs.h,v 1.7 2002-10-12 04:10:15 edwards Exp $

/*! \file
 * \brief Global functions on QDPType
 */

QDP_BEGIN_NAMESPACE(QDP);

// NOTE: there is no inlining anywhere in this file - some
// small effort at controlling code bloat.


/** \defgroup group3 QDP global reductions
 *  
 *  Global reductions, like sum, norm2, etc.
 *  @{
 */
//-----------------------------------------------
// Global sums
//! OScalar = sum(source)
/*!
 * Allow a global sum that sums over the lattice, but returns an object
 * of the same primitive type. E.g., contract only over lattice indices
 */
template<class T, class C>
typename UnaryReturn<C, FnSum>::Type_t
sum(const QDPType<T,C>& s1)
{
  return sum(PETE_identity(s1));
}


//! OScalar = sum(source)  under an explicit subset
/*!
 * Allow a global sum that sums over the lattice, but returns an object
 * of the same primitive type. E.g., contract only over lattice indices
 */
template<class T, class C>
typename UnaryReturn<C, FnSum>::Type_t
sum(const QDPType<T,C>& s1, const Subset& s)
{
  return sum(PETE_identity(s1),s);
}


//! OScalar = norm2(trace(conj(source)*source))
/*!
 * return  num(trace(conj(source)*source))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T, class C>
typename UnaryReturn<C, FnNorm2>::Type_t
norm2(const QDPType<T,C>& s1)
{
  return sum(localNorm2(s1));
}

template<class T, class C>
typename UnaryReturn<C, FnNorm2>::Type_t
norm2(const QDPExpr<T,C>& s1)
{
  return sum(localNorm2(s1));
}


//! OScalar = norm2(trace(conj(source)*source)) under an explicit subset
/*!
 * return  num(trace(conj(source)*source))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T, class C>
typename UnaryReturn<C, FnNorm2>::Type_t
norm2(const QDPType<T,C>& s1, const Subset& s)
{
  return sum(localNorm2(s1),s);
}

template<class T, class C>
typename UnaryReturn<C, FnNorm2>::Type_t
norm2(const QDPExpr<T,C>& s1, const Subset& s)
{
  return sum(localNorm2(s1),s);
}


//! OScalar = innerproduct(conj(source1)*source2)
/*!
 * return  sum(trace(conj(source1)*source2))
 *
 * Sum over the lattice
 */
template<class T1, class C1, class T2, class C2>
typename BinaryReturn<C1, C2, FnInnerproduct>::Type_t
innerproduct(const QDPType<T1,C1>& s1, const QDPType<T2,C2>& s2)
{
  return sum(localInnerproduct(s1,s2));
}

template<class T1, class C1, class T2, class C2>
typename BinaryReturn<C1, C2, FnInnerproduct>::Type_t
innerproduct(const QDPType<T1,C1>& s1, const QDPExpr<T2,C2>& s2)
{
  return sum(localInnerproduct(s1,s2));
}

template<class T1, class C1, class T2, class C2>
typename BinaryReturn<C1, C2, FnInnerproduct>::Type_t
innerproduct(const QDPExpr<T1,C1>& s1, const QDPType<T2,C2>& s2)
{
  return sum(localInnerproduct(s1,s2));
}

template<class T1, class C1, class T2, class C2>
typename BinaryReturn<C1, C2, FnInnerproduct>::Type_t
innerproduct(const QDPExpr<T1,C1>& s1, const QDPExpr<T2,C2>& s2)
{
  return sum(localInnerproduct(s1,s2));
}


//! OScalar = innerproduct(conj(source1)*source2) under an explicit subset
/*!
 * return  sum(trace(conj(source1)*source2))
 *
 * Sum over the lattice
 */
template<class T1, class C1, class T2, class C2>
typename BinaryReturn<C1, C2, FnInnerproduct>::Type_t
innerproduct(const QDPType<T1,C1>& s1, const QDPType<T2,C2>& s2,
	     const Subset& s)
{
  return sum(localInnerproduct(s1,s2),s);
}

template<class T1, class C1, class T2, class C2>
typename BinaryReturn<C1, C2, FnInnerproduct>::Type_t
innerproduct(const QDPType<T1,C1>& s1, const QDPExpr<T2,C2>& s2,
  	     const Subset& s)
{
  return sum(localInnerproduct(s1,s2),s);
}

template<class T1, class C1, class T2, class C2>
typename BinaryReturn<C1, C2, FnInnerproduct>::Type_t
innerproduct(const QDPExpr<T1,C1>& s1, const QDPType<T2,C2>& s2,
  	     const Subset& s)
{
  return sum(localInnerproduct(s1,s2),s);
}

template<class T1, class C1, class T2, class C2>
typename BinaryReturn<C1, C2, FnInnerproduct>::Type_t
innerproduct(const QDPExpr<T1,C1>& s1, const QDPExpr<T2,C2>& s2,
  	     const Subset& s)
{
  return sum(localInnerproduct(s1,s2),s);
}


//! OScalar = innerproductReal(conj(source1)*source2)
/*!
 * return  sum(trace(conj(source1)*source2))
 *
 * Sum over the lattice
 */
template<class T1, class C1, class T2, class C2>
typename BinaryReturn<C1, C2, FnInnerproductReal>::Type_t
innerproductReal(const QDPType<T1,C1>& s1, const QDPType<T2,C2>& s2)
{
  return sum(localInnerproductReal(s1,s2));
}

template<class T1, class C1, class T2, class C2>
typename BinaryReturn<C1, C2, FnInnerproductReal>::Type_t
innerproductReal(const QDPType<T1,C1>& s1, const QDPExpr<T2,C2>& s2)
{
  return sum(localInnerproductReal(s1,s2));
}

template<class T1, class C1, class T2, class C2>
typename BinaryReturn<C1, C2, FnInnerproductReal>::Type_t
innerproductReal(const QDPExpr<T1,C1>& s1, const QDPType<T2,C2>& s2)
{
  return sum(localInnerproductReal(s1,s2));
}

template<class T1, class C1, class T2, class C2>
typename BinaryReturn<C1, C2, FnInnerproductReal>::Type_t
innerproductReal(const QDPExpr<T1,C1>& s1, const QDPExpr<T2,C2>& s2)
{
  return sum(localInnerproductReal(s1,s2));
}


//! OScalar = innerproductReal(conj(source1)*source2) under an explicit subset
/*!
 * return  sum(trace(conj(source1)*source2))
 *
 * Sum over the lattice
 */
template<class T1, class C1, class T2, class C2>
typename BinaryReturn<C1, C2, FnInnerproductReal>::Type_t
innerproductReal(const QDPType<T1,C1>& s1, const QDPType<T2,C2>& s2,
		 const Subset& s)
{
  return sum(localInnerproductReal(s1,s2),s);
}

template<class T1, class C1, class T2, class C2>
typename BinaryReturn<C1, C2, FnInnerproductReal>::Type_t
innerproductReal(const QDPType<T1,C1>& s1, const QDPExpr<T2,C2>& s2,
		 const Subset& s)
{
  return sum(localInnerproductReal(s1,s2),s);
}

template<class T1, class C1, class T2, class C2>
typename BinaryReturn<C1, C2, FnInnerproductReal>::Type_t
innerproductReal(const QDPExpr<T1,C1>& s1, const QDPType<T2,C2>& s2,
		 const Subset& s)
{
  return sum(localInnerproductReal(s1,s2),s);
}

template<class T1, class C1, class T2, class C2>
typename BinaryReturn<C1, C2, FnInnerproductReal>::Type_t
innerproductReal(const QDPExpr<T1,C1>& s1, const QDPExpr<T2,C2>& s2,
		 const Subset& s)
{
  return sum(localInnerproductReal(s1,s2),s);
}


//-----------------------------------------------------------------------------
// Multiple global sums 
//! dest  = sumMulti(source1,Set) 
/*!
 * Compute the global sum on multiple subsets specified by Set 
 *
 * This is a very simple implementation. There is no need for
 * anything fancier unless global sums are just so extraordinarily
 * slow. Otherwise, generalized sums happen so infrequently the slow
 * version is fine.
 */
template<class T, class C>
typename UnaryReturn<C, FnSumMulti>::Type_t
sumMulti(const QDPType<T,C>& s1, const Set& ss)
{
  return sumMulti(PETE_identity(s1), ss);
}


/** @} */ // end of group3


/** \defgroup group5 QDP auxilliary functions
 *  
 *  SU(N) operations, spin projections, etc.
 *  @{
 */
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
template<class T, class C, class T1, class C1> 
void su2_extract(QDPType<T,C>& r_0, QDPType<T,C>& r_1, QDPType<T,C>& r_2, QDPType<T,C>& r_3, 
		 int su2_index, QDPType<T1,C1>& source, 
		 const Subset& s)
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
	      i1, i2, s);
}
  

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
template<class T, class C, class T1, class C1> 
void su2_extract(QDPType<T,C>& r_0, QDPType<T,C>& r_1, QDPType<T,C>& r_2, QDPType<T,C>& r_3, 
		 int su2_index, QDPType<T1,C1>& source)
{
  su2_extract(r_0, r_1, r_2, r_3, su2_index, source, all);
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
template<class T, class C, class T1, class C1> 
void sun_fill(QDPType<T,C>& dest, 
	      int su2_index,
	      const QDPType<T1,C1>& r_0, const QDPType<T1,C1>& r_1, 
	      const QDPType<T1,C1>& r_2, const QDPType<T1,C1>& r_3,
	      const Subset& s)
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
	   i1, i2, s);
}
  

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
template<class T, class C, class T1, class C1> 
void sun_fill(QDPType<T,C>& dest, 
	      int su2_index,
	      const QDPType<T1,C1>& r_0, const QDPType<T1,C1>& r_1, 
	      const QDPType<T1,C1>& r_2, const QDPType<T1,C1>& r_3)
{
  sun_fill(dest, su2_index, r_0, r_1, r_2, r_3, all);
}




//-----------------------------------------------
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


//-----------------------------------------------
// Spin reconstruction
//! dest  = spinReconstruct(source1) 
/*! Boneheaded simple implementation till I get a better one... */
template<class T, class C>
typename UnaryReturn<C, FnSpinReconstruct>::Type_t
spinReconstruct(const QDPType<T,C>& s1, int mu, int isign)
{
//  typedef typename UnaryReturn<C, FnSpinReconstruct>::Type_t  Ret_t;
//  Ret_t  d;

  switch (isign)
  {
  case -1:
    switch (mu)
    {
    case 0:
      return spinReconstructDir0Minus(s1);
    case 1:
      return spinReconstructDir1Minus(s1);
    case 2:
      return spinReconstructDir2Minus(s1);
    case 3:
      return spinReconstructDir3Minus(s1);
    default:
      cerr << "Spin reconstruct: illegal direction\n";
      exit(1);
    }
    break;

  case +1:
    switch (mu)
    {
    case 0:
      return spinReconstructDir0Plus(s1);
    case 1:
      return spinReconstructDir1Plus(s1);
    case 2:
      return spinReconstructDir2Plus(s1);
    case 3:
      return spinReconstructDir3Plus(s1);
    default:
      cerr << "Spin reconstruct: illegal direction\n";
      exit(1);
    }
    break;

  default:
    cerr << "Spin reconstruct: isign must be pos or neg.\n";
    exit(1);
  }

//  return d;
}


/** @} */ // end of group5

QDP_END_NAMESPACE();

