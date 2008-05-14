// -*- C++ -*-
// $Id: qdp_scalarsite_sse.h,v 1.28 2008-05-14 18:58:35 bjoo Exp $

/*! @file
 * @brief Intel SSE optimizations
 *
 * SSE optimizations of basic operations
 */

#ifndef QDP_SCALARSITE_SSE_H
#define QDP_SCALARSITE_SSE_H


#if BASE_PRECISION == 32

/* 32 Bits - we have SSE so use it */
#warning "Using SSE BLAS. If your compiler cant handle intrinsics your build will break"
#include "scalarsite_sse/qdp_scalarsite_sse_linalg.h"
#include "scalarsite_sse/qdp_scalarsite_sse_blas.h"
#include "scalarsite_generic/qdp_scalarsite_generic_cblas.h"
#include "scalarsite_sse/sse_spin_aggregate.h"

#else

#warning "Non 32-bit precision: Using Generics"
#include "scalarsite_generic/qdp_scalarsite_generic_linalg.h"
#include "scalarsite_generic/qdp_scalarsite_generic_blas.h"
#include "scalarsite_generic/qdp_scalarsite_generic_cblas.h"
#include "scalarsite_generic/generic_spin_aggregate.h"

#endif // Base precision

#endif  // guard

