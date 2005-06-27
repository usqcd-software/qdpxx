#ifndef QDP_DEFAULT_ALLOCATOR
#define QDP_DEFAULT_ALLOCATOR

#include "qdp_allocator.h"
#include "qdp_stdio.h"
#include "qdp_singleton.h"
#include <string>
#include <map>

using namespace std;
QDP_BEGIN_NAMESPACE(QDP);
QDP_BEGIN_NAMESPACE(Allocator);

// Specialise allocator to the default case
class QDPDefaultAllocator {
private:
  // Convenience typedefs to save typing

  // The type of the map to hold the aligned unaligned values
  typedef map<unsigned char*, unsigned char *> MapT;

  // The type returned on map insertion, allows me to check
  // the insertion was successful.
  typedef pair<MapT::iterator, bool> InsertRetVal;

  // Disallow Copies
  QDPDefaultAllocator(const QDPDefaultAllocator& c) {};

  // Disallow assignments (copies by another name)
  QDPDefaultAllocator& operator=(const QDPDefaultAllocator& c) {};

 public:
  QDPDefaultAllocator() {};
  //! Allocator function. Allocates n_bytes, into a memory pool
  //! This is a default implementation, with only 1 memory pool
  //! So we simply ignore the memory pool hint.
  inline void*
  allocate(size_t n_bytes,const MemoryPoolHint& mem_pool_hint) {
    
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
  inline void 
  free(void *mem) { 
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

private:
  MapT the_alignment_map;
};

// Turn into a Singleton, CreateUsingNew, DefaultLifetime, SingleThreaded
typedef SingletonHolder<QDPDefaultAllocator> theQDPAllocator;

QDP_END_NAMESPACE();
QDP_END_NAMESPACE();

#endif
