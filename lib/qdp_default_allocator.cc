/*! @file
 * @brief QCDOC memory allocator
 */


#include "qdp.h"
#include "qdp_default_allocator.h"

namespace __QDP__ {
namespace Allocator {
 
  // The type returned on map insertion, allows me to check
  // the insertion was successful.
  typedef std::pair<MapT::iterator, bool> InsertRetVal;


  //! Allocator function. Allocates n_bytes, into a memory pool
  //! This is a default implementation, with only 1 memory pool
  //! So we simply ignore the memory pool hint.
  void*
  QDPDefaultAllocator::allocate(size_t n_bytes,const MemoryPoolHint& mem_pool_hint) {
    
    //! The raw unaligned pointer returned by the allocator
    unsigned char *unaligned;

    //! The aligned pointer that we make out of the unaligned one.
    unsigned char *aligned;

    size_t bytes_to_alloc;
    bytes_to_alloc = n_bytes;
    if ( n_bytes % (32*1024) == 0 ) { 
      bytes_to_alloc += 0; // 2 lines bytes to kill cache aliasing
    }
    bytes_to_alloc += QDP_ALIGNMENT_SIZE;

    // Try and allocate the memory
    try { 
      unaligned = new unsigned char[ bytes_to_alloc ];
    }
    catch( std::bad_alloc ) { 
      QDPIO::cerr << "Unable to allocate memory in allocate()" << std::endl;
      throw;  // Re throw the bad alloc is the correct behaviour

    }

    // Work out the aligned pointer
    aligned = (unsigned char *)( ( (unsigned long)unaligned + (QDP_ALIGNMENT_SIZE-1) ) & ~(QDP_ALIGNMENT_SIZE - 1));

#if defined(QDP_DEBUG_MEMORY)
    // Current location
    FuncInfo_t& info = infostack.top();

    // Insert into the map
    InsertRetVal r = the_alignment_map.insert(
      std::make_pair(aligned, MapVal(unaligned, info.func, info.line, bytes_to_alloc)));
#else
    // Insert into the map
    InsertRetVal r = the_alignment_map.insert(std::make_pair(aligned, unaligned));
#endif

    // Check success of insertion.
    if( ! r.second ) { 
      QDPIO::cerr << "Failed to insert (unaligned,aligned) pair into map" << std::endl;
      QDP_abort(1);
    }

    // Return the aligned pointer
    return (void *)aligned;
  }


  //! Free an aligned pointer, which was allocated by us.
  void 
  QDPDefaultAllocator::free(void *mem) { 
    unsigned char* unaligned; 

    // Look up the original unaligned pointer in the memory. 
    MapT::iterator iter = the_alignment_map.find((unsigned char*)mem);
    if( iter != the_alignment_map.end() ) 
    { 
#if defined(QDP_DEBUG_MEMORY)
      // Find the original unaligned pointer
      unaligned = iter->second.unaligned;
#else
      // Find the original unaligned pointer
      unaligned = iter->second;
#endif
      
      // Remove its entry from the map
      the_alignment_map.erase(iter);

      // Delete the actual unaligned pointer
      delete [] unaligned;
    }
    else { 
      QDPIO::cerr << "Pointer not found in map" << std::endl;
      QDP_abort(1);
    }
  }


#if defined(QDP_DEBUG_MEMORY)
  //! Dump the map
  void
  QDPDefaultAllocator::dump()
  {
     if ( Layout::primaryNode() )
     {
       size_t sum = 0;
       typedef MapT::const_iterator CI;
       QDPIO::cout << "Dumping memory map" << std::endl;
       for( CI j = the_alignment_map.begin();
             j != the_alignment_map.end(); j++)
       {
	 sum += j->second.bytes;
         printf("mem= 0x%x  bytes= %d  bytes/site= %d  line= %d  func= %s\n", j->first, 
                j->second.bytes, j->second.bytes/Layout::sitesOnNode(), j->second.line, j->second.func.c_str());
       }
       printf("total bytes= %d\n", sum);
     }
  }

  // Setter
  void
  QDPDefaultAllocator::pushFunc(const char* func, int line)
  {
    infostack.push(FuncInfo_t(func,line));
  }

  // Nuker
  void
  QDPDefaultAllocator::popFunc()
  {
    if (infostack.empty())
    {
      QDPIO::cerr << __func__ << ": invalid pop" << std::endl;
      QDP_abort(1);
    }
  
    infostack.pop();
  }


  static const char* nowhere = "nowhere";

  // Init
  void
  QDPDefaultAllocator::init(size_t PoolSizeInMB)
  {
    infostack.push(FuncInfo_t(nowhere,0));
  }

#else

  //! Dump the map
  void
  QDPDefaultAllocator::dump()
  {
     if ( Layout::primaryNode() )
     {
       typedef MapT::const_iterator CI;
       QDPIO::cout << "Dumping memory map" << std::endl;
       for( CI j = the_alignment_map.begin();
             j != the_alignment_map.end(); j++)
       {
         printf("mem= 0x%lx  unaligned= 0x%lx\n", (unsigned long)j->first, 
		(unsigned long)j->second);
       }
     }
  }

  // Setter
  void
  QDPDefaultAllocator::pushFunc(const char* func, int line) {}

  // Nuker
  void
  QDPDefaultAllocator::popFunc() {}

  // Init
  void
  QDPDefaultAllocator::init(size_t PoolSizeInMB ) {
	  QDPIO::cout << "Initializing QDPDefaultAllocator." << std::endl;

  }

#endif

} // namespace Allocator
} // namespace __QDP__
