// -*- C++ -*-
// $Id: qdp_scalarsite_bagel_qdp.h,v 1.5 2007-02-26 22:49:21 bjoo Exp $

/*! @file
 * @brief Qcdoc optimizations
 *
 * Qcdoc version of optimized basic operations
 */

#ifndef QDP_SCALARSITE_BAGEL_QDP_H
#define QDP_SCALARSITE_BAGEL_QDP_H

#include "qdp_config.h"

#ifdef QDP_USE_QCDOC
// Use QCDOC specific Linalg stuff (inline assembler etc)
// This is a hack until I have the linalg stuff Bagellified
#include "scalarsite_qcdoc/qdp_scalarsite_qcdoc_linalg.h"
#else

// Use generic SU3 linalg

#include "scalarsite_generic/qdp_scalarsite_generic_linalg.h"
#endif

// Use QCDOC specific BLAS for now -- use Pete's assembler
#include "scalarsite_bagel_qdp/qdp_scalarsite_bagel_qdp_blas.h"
// Use GENERIC Chiral Projector BLAS for now, BAGEL should generate
// this eventually

#include "scalarsite_bagel_qdp/qdp_scalarsite_bagel_qdp_blas_g5.h"

// Use GENERIC Complex BLAS for now as there is no other yet
#include "scalarsite_generic/qdp_scalarsite_generic_cblas.h"

#include "scalarsite_generic/generic_spin_aggregate.h"
#endif
