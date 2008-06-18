// $Id: sse_blas_vaxpbyz4_double.h,v 1.2 2008-06-18 16:03:02 bjoo Exp $

/*! @file
 *  @brief Generic Scalar VAXPY routine
 *
 */

#ifndef QDP_SSE_BLAS_VAXPBYZ4_DOUBLE
#define QDP_SSE_BLAS_VAXPBYZ4_DOUBLE

#include "qdp_precision.h"

namespace QDP {

  void vaxpbyz4(REAL64 *z, REAL64 *a, REAL64 *x, REAL64 *b, REAL64 *y, int n_4vec);

} // namespace QDP;

#endif // guard
