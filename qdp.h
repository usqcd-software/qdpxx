// -*- C++ -*-
// $Id: qdp.h,v 1.9 2002-11-04 04:38:45 edwards Exp $

/*! \file
 * \brief Primary include file for QDP
 *
 * No other file should be included by the user
 */

/*! \mainpage  QDP
 *
 * \section Description
 *
 * QDP is a C++ data-parallel interface for Lattice field theory.
 * The QDP interface provides an environment somewhat similar to 
 * Fortran 90 - namely data-parallel operations (operator/infix form)
 * which can be applied on lattice wide objects. The interface provides a
 * level of abstraction such that high-level user code written using the
 * API can be run unchanged on a single processor workstation or a
 * collection of multiprocessor nodes with parallel communications.
 * Architectural dependencies are hidden below the interface. A variety
 * of types for the site elements are provided. To achieve good
 * performance, overlapping communication and computation primitives are
 * provided.
 */

/*! \namespace QDP
 * \brief Primary namespace holding all QDP types, operations and objects
 */

#include <cstdio>
#include <cstdlib>
#include <ostream>
#include <iostream>

using namespace std;   // I do not like this

using std::iostream;
using std::ostream;

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


// Info/error routines
QDP_BEGIN_NAMESPACE(QDP);

/**
 * Simple information display routine
 */
extern int  QDP_info (const char* format, ...);

/**
 * Simple error display routine
 */
extern int  QDP_error (const char* format, ...);

/**
 * Simple error display and abort routine
 */
extern void  QDP_error_exit (const char *format, ...);

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
#include "io.h"
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

#if defined(ARCH_SCALAR)
// Architectural specific code to a single node/single proc box
#warning "Using scalar architecture"
#include "scalar_specific.h"

#elif defined(ARCH_PARSCALAR)
// Architectural specific code to a parallel/single proc box
#warning "Using parallel scalar architecture"
#include "parscalar_specific.h"

#else
#error "Unknown architecture ARCH"
#endif


#endif
