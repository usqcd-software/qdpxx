// -*- C++ -*-
// $Id: qdp_scalarsite_qcdoc.h,v 1.4 2004-03-21 23:51:19 bjoo Exp $

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
#include "scalarsite_qcdoc/qdp_scalarsite_qcdoc_blas.h"


#endif
