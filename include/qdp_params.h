// -*- C++ -*-
// $Id: qdp_params.h,v 1.1 2003-05-22 20:06:27 edwards Exp $

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

const int Nd = ND;
const int Nc = NC;
const int Ns = NS;

/*! @} */  // end of group params

QDP_END_NAMESPACE();


