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

  // Disallow creation / destruction by anyone except 
  // the singleton CreateUsingNew policy which is a "friend"
  // I don't like friends but this follows Alexandrescu's advice
  // on p154 of Modern C++ Design (A. Alexandrescu)
  QDPDefaultAllocator() {};
  ~QDPDefaultAllocator() {};

  friend class QDP::CreateUsingNew<QDP::Allocator::QDPDefaultAllocator>;
 public:

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

// Turn into a Singleton. Create with CreateUsingNew
// Has NoDestroy lifetime, as it may be needed for 
// the destruction policy is No Destroy, so the 
// Singleton is not cleaned up on exit. This is so 
// that static objects can refer to it with confidence
// in their own destruction, not having to worry that
// atexit() may have destroyed the allocator before
// the static objects need to feed memory. 
typedef SingletonHolder<QDP::Allocator::QDPDefaultAllocator,
			QDP::CreateUsingNew,
			QDP::NoDestroy,
			QDP::SingleThreaded> theQDPAllocator;

QDP_END_NAMESPACE();
QDP_END_NAMESPACE();

#endif
