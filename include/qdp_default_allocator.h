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
  void*
  allocate(size_t n_bytes,const MemoryPoolHint& mem_pool_hint);


  //! Free an aligned pointer, which was allocated by us.
  void 
  free(void *mem);
private:
  MapT the_alignment_map;
};

// Turn into a Singleton, CreateUsingNew, DefaultLifetime, SingleThreaded
typedef SingletonHolder<QDPDefaultAllocator> theQDPAllocator;

QDP_END_NAMESPACE();
QDP_END_NAMESPACE();

#endif
