// -*- C++ -*-
// $Id: qdp_scalarsite_sse.h,v 1.19 2004-04-07 17:36:01 edwards Exp $

/*! @file
 * @brief Intel SSE optimizations
 *
 * SSE optimizations of basic operations
 */

#ifndef QDP_SCALARSITE_SSE_H
#define QDP_SCALARSITE_SSE_H

// These SSE asm instructions are only supported under GCC/G++
#if defined(__GNUC__)

// Use SSE specific Linalg stuff (inline assembler etc)
#include "scalarsite_sse/qdp_scalarsite_sse_linalg.h"

#if __GNUC_MINOR__ >= 2
// Use SSE specific blas stuff (inline assembler etc)
// Only supported on gcc >= 3.2
#include "scalarsite_sse/qdp_scalarsite_sse_blas.h"
#else
#warning "This version of gcc does not support vector types - not using SSE blas code"
#endif

#else

#error "This is not a GNUC compiler, and therefore does not support the GNU specific asm directives."

#endif  // gnuc

#endif  // guard

