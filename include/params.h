// -*- C++ -*-
// $Id: params.h,v 1.7 2002-11-28 02:57:56 edwards Exp $

/*! @file
 * @brief Fundamental parameters
 */

QDP_BEGIN_NAMESPACE(QDP);

/*! @defgroup params Fundamental parameters for QDP
 *
 * The user can change the compile time number of dimensions,
 * colors and spin components
 *
 * @{
 */

//! Number of dimensions
#define ND  2

//! Number of colors
#define NC  3

//! Number of spin components
#define NS  4

//! Compile time max sub-lattice size in  OLattice:F (private member) - run-time can be smaller
/*! NOTE, the use of this macro is controlled by the   NO_MEM  variable 
 * in outer.h */
#define  VOLUME   256


const int Nd = ND;
const int Nc = NC;
const int Ns = NS;



/*! @} */  // end of group params

QDP_END_NAMESPACE();
