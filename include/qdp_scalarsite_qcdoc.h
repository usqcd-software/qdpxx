// -*- C++ -*-
// $Id: qdp_scalarsite_qcdoc.h,v 1.7 2004-05-09 11:54:44 bjoo Exp $

/*! @file
 * @brief Qcdoc optimizations
 *
 * Qcdoc version of optimized basic operations
 */

#ifndef QDP_SCALARSITE_QCDOC_H
#define QDP_SCALARSITE_QCDOC_H

// Use QCDOC specific Linalg stuff (inline assembler etc)
#include "scalarsite_qcdoc/qdp_scalarsite_qcdoc_linalg.h"

// Use QCDOC specific BLAS for now -- use Pete's assembler
#include "scalarsite_qcdoc/qdp_scalarsite_qcdoc_blas.h"

// Use GENERIC Complex BLAS for now as there is no other yet
#include "scalarsite_generic/qdp_scalarsite_qcdoc_cblas.h"

#endif
