#ifndef QDP_SSE_INTRIN_H
#define QDP_SSE_INTRIN_H

#include <xmmintrin.h>

// Include the file with the SSE intrinsics  in it
namespace QDP {
typedef __m128 v4sf;

typedef union { 
  __m128 vector;
  float floats[4];
} SSEVec;
};
#endif 
