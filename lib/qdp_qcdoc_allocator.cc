#include "qdp.h"

QDP_BEGIN_NAMESPACE(QDP);
QDP_BEGIN_NAMESPACE(Allocator);

 //! Allocator function. Allocates n_bytes, into a memory pool
  //! This is a default implementation, with only 1 memory pool
  //! So we simply ignore the memory pool hint.
  void*
  QDPQCDOCAllocator::allocate(size_t n_bytes,const MemoryPoolHint& mem_pool_hint) {

    //! QALLOC always returns aligned pointers
    unsigned char *aligned;
    int qalloc_flags;

    switch( mem_pool_hint ) {
    case DEFAULT:
      qalloc_flags = QCOMMS;
      break;
    case FAST:
      qalloc_flags = (QCOMMS|QFAST);
      break;
    default:
      QDPIO::cerr << "Unsupported mem pool hint " << mem_pool_hint << endl;
      QDP_abort(1);
      break;
    }

    aligned=(unsigned char *)qalloc(qalloc_flags, n_bytes);
    if( aligned == (unsigned char *)NULL ) {
      aligned = (unsigned char *)qalloc(QCOMMS, n_bytes);
      if( aligned == (unsigned char *)NULL ) {
        dump();
        QDPIO::cerr << "Unable to allocate memory with qalloc" << endl;
        QDP_abort(1);
     }
    }

    // Insert into the map
    InsertRetVal r = the_alignment_map.insert(make_pair(aligned, n_bytes));

    // Check success of insertion.
    if( ! r.second ) {
      QDPIO::cerr << "Failed to insert (aligned,n_bytes) pair into map" << endl;
      QDP_abort(1);
    }

    // Return the aligned pointer
    return (void *)aligned;
  }


  //! Free an aligned pointer, which was allocated by us.
  void
  QDPQCDOCAllocator::free(void *mem) {
    // Look up the original aligned pointer in the memory.
    MapT::iterator iter = the_alignment_map.find((unsigned char*)mem);
    if( iter != the_alignment_map.end() ) {
      // Find the original aligned pointer
      unsigned char *aligned = iter->first;

      // Remove its entry from the map
      the_alignment_map.erase(iter);

      // Delete the actual aligned pointer
      qfree(aligned);
    }
    else {
      QDPIO::cerr << "Pointer not found in map" << endl;
      QDP_abort(1);
    }
  }

  //! Dump the map
  void
  QDPQCDOCAllocator::dump()
  {
     if ( Layout::primaryNode() )
     {
       typedef MapT::const_iterator CI;
       QDPIO::cout << "Dumping memory map" << endl;
       for( CI j = the_alignment_map.begin();
             j != the_alignment_map.end(); j++)
       {
         printf("mem= 0x%x  bytes = %d\n", j->first, j->second);
       }
     }
  }

QDP_END_NAMESPACE();
QDP_END_NAMESPACE();
