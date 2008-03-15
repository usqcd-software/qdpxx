// -*- C++ -*- 
// $Id: qdp_defs.h,v 1.14.4.1 2008-03-15 14:28:54 edwards Exp $
/*! file 
 * \brief QDP Typedef switcharoo 
 *
 * This file acts as a switchbox for including qdp_scalarsite_defs.h
 * or qdp_scalarvecsite_defs.h for common datatypes depending on the 
 * parallel architecture chosen.
 *
 * The purpose of this file is so that this switching 
 *   i) Done in a single place only  AND
 *  ii) Other QDP++ header files can include this file with impunity 
 *      (it'll have an include guard)
 */

#ifndef QDP_DEFS_H
#define QDP_DEFS_H

#include <qdp_config.h>
#include "qdp_precision.h"
#include "qdp_params.h"

#if defined(ARCH_SCALAR) || defined(ARCH_PARSCALAR)
#include "qdp_scalarsite_defs.h"

#elif defined(ARCH_SCALARVEC) || defined(ARCH_PARSCALARVEC)
#include "qdp_scalarvecsite_defs.h"

#else
#error "Unknown architecture ARCH"
#endif

#endif
