#ifndef SSE_MV_SWITCHBOX_H
#define SSE_MV_SWITCHBOX_H
// Switchbox for Matrix vector routines for SSE Builds
// 
// If we have single precision and suitable GNU Compiler then 
// use the SSE otherwise fall back to generics
//
// Author: $Id: sse_mv_switchbox.h,v 1.2 2007-02-06 15:01:58 bjoo Exp $


#if (BASE_PRECISION == 32) && defined(__GNUC__) 
// We are in single prec and  we are using suitable GCC Compiler
#include "scalarsite_sse/sse_mat_vec.h"
#include "scalarsite_sse/sse_adj_mat_vec.h"
#define _inline_mult_su3_mat_vec(aa,bb,cc) \
{\
  _inline_sse_mult_su3_mat_vec(aa,bb,cc) \
}

#define _inline_mult_adj_su3_mat_vec(aa,bb,cc) \
{\
 _inline_sse_mult_adj_su3_mat_vec(aa,bb,cc) \
}

#else // end _GNUC_MAJOR_

// We are either not in single precision or don't have a suitable compiler
#include "scalarsite_generic/generic_mv_switchbox.h"

#endif

#endif
