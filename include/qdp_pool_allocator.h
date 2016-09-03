// -*- C++ -*-

/*! \file
 * \brief Default memory allocator for QDP
 *
 */

#ifndef QDP_POOL_ALLOCATOR
#define QDP_POOL_ALLOCATOR


#define TBB_PREVIEW_MEMORY_POOL 1
#include "tbb/memory_pool.h"

#include "qdp_singleton.h"
namespace QDP
{
 	 namespace Allocator {

 	 // A Descrptor contains the start block and the number of blocks
 	 // In an object
 	 // The idea is that when we allocate n_blocks, they will be contiguous
 	 // so we can just note the start, and the number
 	 // and then pop them off the end of the free list.
 	 // When we return to the pool we can just add back the blocks.



 	 // Quick and Dirty Pool ALlocator
 	 class QDPPoolAllocator {
 	 public:
 		 QDPPoolAllocator(void);

 		 ~QDPPoolAllocator();

 		 void  init(size_t PoolSizeInGB);

 		 void* alloc(size_t size);
 		 void  free(void *mem);


 	 private:
 		 size_t _PoolSize;
 		 unsigned char* _MyMem;
 		 tbb::fixed_pool* _LargePool;

 	 };



    // Turn into a Singleton. Create with CreateUsingNew
    // Has NoDestroy lifetime, as it may be needed for 
    // the destruction policy is No Destroy, so the 
    // Singleton is not cleaned up on exit. This is so 
    // that static objects can refer to it with confidence
    // in their own destruction, not having to worry that
    // atexit() may have destroyed the allocator before
    // the static objects need to feed memory. 
    typedef SingletonHolder<QDP::Allocator::QDPPoolAllocator,
			    QDP::CreateUsingNew,
			    QDP::NoDestroy,
			    QDP::SingleThreaded> theQDPPoolAllocator;

  } // namespace Allocator
} // namespace QDP

#endif
