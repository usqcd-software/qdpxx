#ifndef QDP_SSE_INTRIN_H
#define QDP_SSE_INTRIN_H

// Include the file with the SSE intrinsics  in it
#include <xmmintrin.h>
namespace __QDP__ {

typedef union { 
  __m128 vector;
  float floats[4];
} SSEVec;
};
#endif 
