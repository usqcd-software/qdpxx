// $Id: layout.cc,v 1.6 2002-10-28 03:08:44 edwards Exp $
//
// QDP data parallel interface
//
// Layout
//
// This routine provides various layouts, including
//    lexicographic
//    2-checkerboard  (even/odd-checkerboarding of sites)
//    32-style checkerboard (even/odd-checkerboarding of hypercubes)

#include "qdp.h"
#include "proto.h"

#define  USE_LEXICO_LAYOUT
#undef   USE_CB2_LAYOUT
#undef   USE_CB32_LAYOUT

QDP_BEGIN_NAMESPACE(QDP);

namespace Layout
{
};

QDP_END_NAMESPACE();
