// -*- C++ -*-
// $Id: qdp_scalarsite_sse.h,v 1.17 2004-03-24 03:40:14 edwards Exp $

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

// Use SSE specific blas stuff (inline assembler etc)
#include "scalarsite_sse/qdp_scalarsite_sse_blas.h"

#else

#error "This is not a GNUC compiler, and therefore does not support the GNU specific asm directives."

#endif  // gnuc

#endif  // guard

