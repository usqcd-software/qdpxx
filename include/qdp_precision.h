// -*- C++ -*-

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
typedef int       INTEGER32;
typedef float     REAL32;
typedef double    REAL64;
typedef bool      LOGICAL;

// Set the base floating precision
#if BASE_PRECISION == 32
// Use single precision for base precision
typedef REAL32    REAL;
typedef REAL64    DOUBLE;

#ifndef QDP_INNER_LOG
#if defined(__MIC__)
#warning "Using INNER_LOG of 4"
#define INNER_LOG 4
#elif defined(__AVX__)
#warning "Using INNER_LOG of 3"
#define INNER_LOG 3
#else
#warning "Using INNER_LOG of 2"
#define INNER_LOG 2
#endif
#else 
#warning user defined inner log QDP_INNER_LOG
#define INNER_LOG (QDP_INNER_LOG)
#endif

#elif BASE_PRECISION == 64
// Use double precision for base precision
typedef REAL64    REAL;
typedef REAL64    DOUBLE;
#ifndef QDP_INNER_LOG
#if defined(__MIC__)
#warning "Using INNER_LOG of 3"
#define INNER_LOG 3
#elif defined(__AVX__)
#warning "Using INNER_LOG of 2"
#define INNER_LOG 2
#else
#warning "Using INNER_LOG of 1"
#define INNER_LOG 1
#endif
#else 
#warning user defined inner log QDP_INNER_LOG
#define INNER_LOG (QDP_INNER_LOG)
#endif

#else
#error "Unknown BASE_PRECISION"
#endif

#endif
