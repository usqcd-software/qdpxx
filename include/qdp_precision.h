// -*- C++ -*-
// $Id: qdp_precision.h,v 1.4 2004-08-07 01:38:57 edwards Exp $

/*! \file
 * \brief PRECISION ISSUES
 */

#ifndef QDP_PRECISION_H
#define QDP_PRECISION_H


// Fix Definitions
#include <qdp_config.h>

// Fix default precision
#if ! defined(BASE_PRECISION)
#define BASE_PRECISION 32
#endif

// These are fixed precision versions
#define INTEGER32 int
#define REAL32    float
#define REAL64    double
#define LOGICAL   bool

// Set the base floating precision
#if BASE_PRECISION == 32
// Use single precision for base precision
#define REAL      REAL32
#define DOUBLE    REAL64

#define INNER_LOG 2

#elif BASE_PRECISION == 64
// Use double precision for base precision
#define REAL      REAL64
#define DOUBLE    REAL64

#define INNER_LOG 1

#else
#error "Unknown BASE_PRECISION"
#endif

#endif
