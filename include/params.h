// -*- C++ -*-
// $Id: params.h,v 1.8 2002-12-14 04:49:01 edwards Exp $

/*! @file
 * @brief Fundamental parameters
 */

QDP_BEGIN_NAMESPACE(QDP);

/*! @defgroup params Fundamental parameters for QDP
 *
 * The user can change the compile time number of dimensions,
 * colors and spin components -- note these are now determined
 * by configure
 *
 * @{
 */

#include <qdp_config.h>

//! Compile time max sub-lattice size in  OLattice:F (private member) - run-time can be smaller
/*! NOTE, the use of this macro is controlled by the   NO_MEM  variable 
 * in outer.h */

const int Nd = ND;
const int Nc = NC;
const int Ns = NS;

/*! @} */  // end of group params

QDP_END_NAMESPACE();


