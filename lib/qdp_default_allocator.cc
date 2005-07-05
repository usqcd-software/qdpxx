#include "qdp.h"

QDP_BEGIN_NAMESPACE(QDP);
QDP_BEGIN_NAMESPACE(Allocator);

  //! Allocator function. Allocates n_bytes, into a memory pool
  //! This is a default implementation, with only 1 memory pool
  //! So we simply ignore the memory pool hint.
  void*
  QDPDefaultAllocator::allocate(size_t n_bytes,const MemoryPoolHint& mem_pool_hint) {
    
    //! The raw unaligned pointer returned by the allocator
    unsigned char *unaligned;

    //! The aligned pointer that we make out of the unaligned one.
    unsigned char *aligned;

    // Try and allocate the memory
    try { 
      unaligned = new unsigned char[ n_bytes + QDP_ALIGNMENT_SIZE ];
    }
    catch( std::bad_alloc ) { 
      QDPIO::cerr << "Unable to allocate memory in allocate()" << endl;
      QDP_abort(1);
    }

    // Work out the aligned pointer
    aligned = (unsigned char *)( ( (unsigned long)unaligned + (QDP_ALIGNMENT_SIZE-1) ) & ~(QDP_ALIGNMENT_SIZE - 1));

    
    // Insert into the map
    InsertRetVal r = the_alignment_map.insert(make_pair(aligned, unaligned));

    // Check success of insertion.
    if( ! r.second ) { 
      QDPIO::cerr << "Failed to insert (unaligned,aligned) pair into map" << endl;
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
    if( iter != the_alignment_map.end() ) { 
      // Find the original unaligned pointer
      unaligned = iter->second;
      
      // Remove its entry from the map
      the_alignment_map.erase(iter);

      // Delete the actual unaligned pointer
      delete [] unaligned;
    }
    else { 
      QDPIO::cerr << "Pointer not found in map" << endl;
      QDP_abort(1);
    }
  }

QDP_END_NAMESPACE();
QDP_END_NAMESPACE();
