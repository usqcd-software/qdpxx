// -*- C++ -*-
// $Id: qdp.h,v 1.3 2002-10-02 20:29:37 edwards Exp $
//
// QDP data parallel interface
//
#include <cstdio>
#include <iostream.h>
#include <cstdlib>

#undef DEBUG

#ifndef QDP_INCLUDE
#define QDP_INCLUDE

// Move to another file eventually
#define FORWARD 1
#define BACKWARD -1


// HACK to make class names shorter
// Definitely do not want this in production
#define QDP std

#define QDP_NO_NAMESPACE

#if defined(QDP_NO_NAMESPACE)
#define QDP_BEGIN_NAMESPACE(a)
#define QDP_END_NAMESPACE()

#else // ! defined(QDP_NO_NAMESPACE)
#define BEGIN_NAMESPACE(a) namespace a {
#define END_NAMESPACE(a) };
#endif


QDP_BEGIN_NAMESPACE(QDP);
void SZ_ERROR(const char *s, ...);
void diefunc();
QDP_END_NAMESPACE();


// Basic includes
QDP_BEGIN_NAMESPACE(QDP);
#define PETE_USER_DEFINED_EXPRESSION
#include "./PETE/PETE.h"
QDP_END_NAMESPACE();

#include "forward.h"

#include "multi.h"
#include "params.h"
#include "layout.h"
#include "subset.h"

#include "traits.h"
#include "qdpexpr.h"
#include "qdptype.h"
#include "qdpsubtype.h"
#include "QDPOperators.h"
#include "newops.h"
//#include "word.h"
#include "simpleword.h"
#include "reality.h"
//#include "inner.h"
#include "primitive.h"
#include "outer.h"
#include "outersubtype.h"
#include "defs.h"
#include "globalfuncs.h"
#include "specializations.h"

//#include "special.h"
#include "random.h"
#include "io.h"

// Architectural specific code
#include "scalar_specific.h"


#endif
