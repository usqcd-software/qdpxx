// -*- C++ -*-
// $Id: qdp_primitive.h,v 1.2 2003-05-23 04:45:53 edwards Exp $

/*! \file
 * \brief Primitive classes
 *
 * Primitives are the various types on the fibers at the lattice sites
 */


// QDP_BEGIN_NAMESPACE(QDP);

/*! \defgroup fiber Fiber only types and operations
 * \ingroup fiberbundle
 *
 * Primitives are the various types on the fibers at the lattice sites.
 *
 * The primitive indices, including Reality (also known as complex or real),
 * is represented as a tensor product over various vector spaces. Different
 * kinds of object can transform in those vector spaces, like Scalar, Vector, and
 * Matrix.
 */

#include "qdp_primscalar.h"
#include "qdp_primmatrix.h"
#include "qdp_primvector.h"
#include "qdp_primseed.h"
#include "qdp_primcolormat.h"
#include "qdp_primcolorvec.h"
#include "qdp_primgamma.h"
#include "qdp_primspinmat.h"
#include "qdp_primspinvec.h"

// QDP_END_NAMESPACE();

