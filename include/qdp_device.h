#ifndef QDP_DEVICE_H
#define QDP_DEVICE_H

#include <qdp_config.h>

typedef int       INTEGER32;
typedef float     REAL32;
typedef double    REAL64;
typedef bool      LOGICAL;

// Set the base floating precision
#if BASE_PRECISION == 32
// Use single precision for base precision
typedef REAL32    REAL;
typedef REAL64    DOUBLE;

#define INNER_LOG 2

#elif BASE_PRECISION == 64
// Use double precision for base precision
typedef REAL64    REAL;
typedef REAL64    DOUBLE;

#define INNER_LOG 1

#else
#error "Unknown BASE_PRECISION"
#endif


#define QDP_ALIGN8    __attribute__ ((aligned  (8)))
#define QDP_ALIGN16   __attribute__ ((aligned (16)))

//

#include "qdp_precision.h"

using namespace std;   // I do not like this - fix later

#define PETE_USER_DEFINED_EXPRESSION
namespace QDP {
#include <PETE/PETE.h>
}


#include "qdp_forward_gpu.h"

#include "qdp_traits.h"
#include "qdp_qdpexpr.h"
//#include "qdp_qdptype.h"

namespace QDP {
#include "QDPOperators.h"
}


#include "qdp_newops.h"
#include "qdp_optops.h"

#include "qdp_sharedmem.h"

#include "qdp_simpleword.h"
#include "qdp_reality.h"
#include "qdp_inner.h"

#include "qdp_primscalar.h"
#include "qdp_primmatrix.h"
#include "qdp_primvector.h"
#include "qdp_primseed.h"
#include "qdp_primcolormat.h"
#include "qdp_primcolorvec.h"
#include "qdp_primgamma.h"
#include "qdp_primspinmat.h"
#include "qdp_primspinvec.h"

#include "qdp_specializations.h"

#include "qdp_random_gpu.h"


#endif  // QDP_INCLUDE


