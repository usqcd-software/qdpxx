// -*- C++ -*-
// $Id: qdp_params.h,v 1.2 2003-10-17 16:06:39 edwards Exp $

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

const int Nd = QDP_ND;
const int Nc = QDP_NC;
const int Ns = QDP_NS;

/*! @} */  // end of group params

QDP_END_NAMESPACE();


