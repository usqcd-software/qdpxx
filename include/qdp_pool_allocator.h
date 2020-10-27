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
namespace __QDP__
{
 	 namespace Allocator {

 	 // A Descrptor contains the start block and the number of blocks
 	 // In an object
 	 // The idea is that when we allocate n_blocks, they will be contiguous
 	 // so we can just note the start, and the number
 	 // and then pop them off the end of the free list.
 	 // When we return to the pool we can just add back the blocks.

	   struct PoolMemInfo                                                                                                                                                                                          {                                                                                                                                                                                                
	     size_t Size;                                                                                                                                                                                 
	     unsigned char* Unaligned;                                                                                                                                                                    
	   };                                                                                                                                                                                                   
                                                                                                                                                                                                         
	   using PoolMapT = std::map<void *,PoolMemInfo>;       


 	 // Quick and Dirty Pool ALlocator
 	 class QDPPoolAllocator {
 	 private:
	   // Disallow Copies
	   QDPPoolAllocator(const QDPPoolAllocator& c) {}
	   
	   // Disallow assignments (copies by another name)
	   void operator=(const QDPPoolAllocator& c) {}
	   
	   // Disallow creation / destruction by anyone except
	   // 	the singleton CreateUsingNew policy which is a "friend"
	   // I don't like friends but this follows Alexandrescu's advice
	   // on p154 of Modern C++ Design (A. Alexandrescu)
	   QDPPoolAllocator(void);
	   ~QDPPoolAllocator();
	   friend class __QDP__::CreateUsingNew<__QDP__::Allocator::QDPPoolAllocator>;
	   friend class __QDP__::CreateStatic<__QDP__::Allocator::QDPPoolAllocator>;
 	 public:
	   // Init -- has to
	   void  init(size_t PoolSizeInMB);
	   
	   void* allocate(size_t n_bytes, const MemoryPoolHint& mem_pool_hint);
	   void  free(void *mem);

	   // Memory debugging Interface
	   void pushFunc(const char *func, int line);
	   void popFunc(void);
	   void dump();
	   

 	 private:
	   size_t _PoolSize;
	   unsigned char* _MyMem;
	   tbb::fixed_pool* _LargePool;
	   PoolMapT theAllocMap;

 	 };




  } // namespace Allocator
} // namespace __QDP__

#endif
