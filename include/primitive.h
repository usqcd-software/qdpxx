// -*- C++ -*-
// $Id: primitive.h,v 1.2 2002-10-12 04:10:15 edwards Exp $

/*! \file
 * \brief Primitive classes
 *
 * Primitives are the various types on the fibers at the lattice sites
 */


QDP_BEGIN_NAMESPACE(QDP);

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

#include "primscalar.h"
#include "primmatrix.h"
#include "primvector.h"
#include "primseed.h"
#include "primcolormat.h"
#include "primcolorvec.h"
#include "primgamma.h"
#include "primspinmat.h"
#include "primspinvec.h"

QDP_END_NAMESPACE();

