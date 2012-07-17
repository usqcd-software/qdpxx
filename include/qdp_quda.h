#ifndef QDP_QUDA
#define QDP_QUDA

#ifndef __CUDACC__

#warning "Using QDP-JIT macros"

//#include <qdp_quda.h>

#define QDP_IS_QDPJIT

#define QDP_ALIGNMENT_SIZE 4096

#include <qdp_init.h>
#include <qdp_cuda.h>
#include <qdp_singleton.h>
#include <qdp_forward.h>
#include <qdp_stopwatch.h>
#include <qdp_allocator.h>
#include <qdp_default_allocator.h>
#include <qdp_cuda_allocator.h>
#include <qdp_pool_allocator.h>
#include <qdp_cache.h>



#define cudaMalloc(dst, size) QDP_allocate(dst, size , __FILE__ , __LINE__ )
#define cudaFree(dst) QDP_free(dst)



inline cudaError_t QDP_allocate(void **dst, size_t size, char * cstrFile , int intLine )
{
  bool op;

  op = QDP::QDPCache::Instance().allocate_device_static( dst , size );

  if (!op)
    return cudaErrorMemoryAllocation;
  else
    return cudaSuccess;
}

inline void QDP_free(void *dst)
{
  QDP::QDPCache::Instance().free_device_static( dst );
}

#endif // __CUDACC__

#endif // QUDA_QDPJIT
