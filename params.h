// -*- C++ -*-
// $Id: params.h,v 1.5 2002-10-26 01:54:30 edwards Exp $

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
#define ND  4

//! Number of colors
#define NC  3

//! Number of spin components
#define NS  4

//! Lattice size
#define  LX0   2
#define  LX1   2
#define  LX2   2
#define  LX3   4


const int Nd = ND;
const int Nc = NC;
const int Ns = NS;


#if ND == 2
#define  VOLUME   LX0*LX1
#elif ND == 3
#define  VOLUME   LX0*LX1*LX2
#elif ND == 4
#define  VOLUME   LX0*LX1*LX2*LX3
#else
#error "unsupported number of dimensions"
#endif

/*! @} */  // end of group params

QDP_END_NAMESPACE();
