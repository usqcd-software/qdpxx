// -*- C++ -*-
// $Id: qdp_scalarsite_sse.h,v 1.20 2004-05-09 11:54:44 bjoo Exp $

/*! @file
 * @brief Intel SSE optimizations
 *
 * SSE optimizations of basic operations
 */

#ifndef QDP_SCALARSITE_SSE_H
#define QDP_SCALARSITE_SSE_H


#if BASE_PRECISION == 32
// DO THE SSE STUFF

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

// Use Complex BLAS from Generics. It is better than nothing
#include "scalarsite_generic/qdp_scalarsite_generic_cblas.h"


#else

#error "This is not a GNUC compiler, and therefore does not support the GNU specific asm directives."

#endif  // gnuc

#else // BASE PRECISION 

// Double precision. We don't actually have any double precision BLAS
// coded in SSE so use the Generics instead. THey are better than nowt.
#include "scalarsite_generic/qdp_scalarsite_generic_linalg.h"
#include "scalarsite_generic/qdp_scalarsite_generic_blas.h"
#include "scalarsite_generic/qdp_scalarsite_generic_cblas.h"

#endif  // BASE PRECISION

#endif  // guard

