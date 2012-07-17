// -*- C++ -*-

/*! \file
 * \brief Cuda memory allocator for QDP
 *
 */



#ifndef QDP_CUDA_ALLOCATOR
#define QDP_CUDA_ALLOCATOR

namespace QDP
{

  class QDPCUDAAllocator {
  public:
    enum { ALIGNMENT_SIZE = 4096 };

    static bool allocate( void** ptr, const size_t n_bytes ) {
      return CudaMalloc( ptr , n_bytes );
    }

    static void free(const void *mem) {
      CudaFree( mem );
    }
  };


  class QDPCUDAHostAllocator {
  public:
    enum { ALIGNMENT_SIZE = 8 };

    static bool allocate( void** ptr, const size_t n_bytes ) {
#if 0
      return CudaHostAlloc( ptr , n_bytes , 0 );
#else 
      try {
	*ptr = (void*)QDP::Allocator::theQDPAllocator::Instance().allocate( n_bytes , QDP::Allocator::DEFAULT );
      }
      catch(std::bad_alloc) {
	QDP_error_exit("pool allocate: host memory allocation failed");
      }
      return true;
      //return posix_memalign( ptr , 4*1024 , n_bytes ) == 0;
#endif
    }

    static void free(const void *mem) {
#if 0
      CudaHostFree( mem );
#else
      //free((void*)mem);
      QDP::Allocator::theQDPAllocator::Instance().free( (void*)mem );
#endif
    }
  };




} // namespace QDP



#endif


