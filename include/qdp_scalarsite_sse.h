// -*- C++ -*-
// $Id: qdp_scalarsite_sse.h,v 1.25 2007-01-31 00:32:12 bjoo Exp $

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

#if __GNUC_MAJOR__ == 3                                   /* If we are using GCC 3.x */

#if __GNUC_MINOR__ >= 2                                   /* SSE only above v 3.2    */
// Use SSE specific blas stuff (inline assembler etc)
// Only supported on gcc >= 3.2
#include "scalarsite_sse/qdp_scalarsite_sse_blas.h"
#else 
#warning "This version of gcc does not support vector types - not using SSE blas code"
#endif

#else 
#include "scalarsite_sse/qdp_scalarsite_sse_blas.h"
#endif

// Use Complex BLAS from Generics. It is better than nothing
#include "scalarsite_generic/qdp_scalarsite_generic_cblas.h"
#include "scalarsite_generic/qdp_scalarsite_spin_project.h"
#include "scalarsite_generic/generic_spin_proj.h"
#include "scalarsite_generic/qdp_generic_fused_spin_proj.h"

// Use chiralProject BLAS from Generics. There seems little difference
// between it and the SSE... Memory Bandwidth limits?

#else

#error "This is not a GNUC compiler, and therefore does not support the GNU specific asm directives."

#endif  // gnuc

#else // BASE PRECISION 

// Double precision. We don't actually have any double precision BLAS
// coded in SSE so use the Generics instead. THey are better than nowt.
#include "scalarsite_generic/qdp_scalarsite_generic_linalg.h"
#include "scalarsite_generic/qdp_scalarsite_generic_blas.h"
#include "scalarsite_generic/qdp_scalarsite_generic_cblas.h"
#include "scalarsite_generic/qdp_scalarsite_spin_project.h"
#include "scalarsite_generic/generic_spin_proj.h"
#include "scalarsite_generic/qdp_generic_fused_spin_proj.h"
#endif  // BASE PRECISION

// This dude takes care of the GNUC and PRECISION and stuff internally
#include "scalarsite_sse/qdp_scalarsite_sse_blas_g5.h"

#endif  // guard

