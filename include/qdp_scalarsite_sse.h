// -*- C++ -*-
// $Id: qdp_scalarsite_sse.h,v 1.26 2007-02-06 15:01:57 bjoo Exp $

/*! @file
 * @brief Intel SSE optimizations
 *
 * SSE optimizations of basic operations
 */

#ifndef QDP_SCALARSITE_SSE_H
#define QDP_SCALARSITE_SSE_H


#if BASE_PRECISION == 32


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
      #endif // END _GNUC_MINOR
    #else    // ELSE __GNUC_MAJOR == 3
      #if __GNUC_MAJOR__ >= 4 

         // Use SSE BLAS FOR GCC 4 and above
         #include "scalarsite_sse/qdp_scalarsite_sse_blas.h"
      #endif
    #endif // END _GNUC_MAJOR == 3

   // Use Complex BLAS from Generics. It is better than nothing
   #include "scalarsite_generic/qdp_scalarsite_generic_cblas.h"

   // Use the SSE SPIN Aggregator (will use SSE MATVEC with GNU inlines
   #include "scalarsite_sse/sse_spin_aggregate.h"
#else // ELSE_GNUC

  #warning Non GNU Compiler: Using Generics
  #include "scalarsite_generic/qdp_scalarsite_generic_linalg.h"
  #include "scalarsite_generic/qdp_scalarsite_generic_blas.h"
  #include "scalarsite_generic/qdp_scalarsite_generic_cblas.h"
  #include "scalarsite_generic/generic_spin_aggregate.h"

#endif  // ENDIF GNUC

#else // ELSE BASE PRECISION 

// Double precision. We dont care about the compiler
  #include "scalarsite_generic/qdp_scalarsite_generic_linalg.h"
  #include "scalarsite_generic/qdp_scalarsite_generic_blas.h"
  #include "scalarsite_generic/qdp_scalarsite_generic_cblas.h"
  #include "scalarsite_generic/generic_spin_aggregate.h"
#endif  // BASE PRECISION

#endif  // guard

