// -*- C++ -*-
// $Id: qdp_scalarsite_qcdoc.h,v 1.3 2004-03-21 21:19:44 bjoo Exp $

/*! @file
 * @brief Qcdoc optimizations
 *
 * Qcdoc version of optimized basic operations
 */

#ifndef QDP_SCALARSITE_QCDOC_H
#define QDP_SCALARSITE_QCDOC_H

#warning "Using QCDOC Specific optimisations" 

// Use QCDOC specific Linalg stuff (inline assembler etc)
#include "scalarsite_qcdoc/qdp_scalarsite_qcdoc_linalg.h"

// Use Generically optimized BLAS for now
#include "scalarsite_generic/qdp_scalarsite_generic_blas.h"


#endif
