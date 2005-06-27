#ifndef QDP_QCDOC_ALLOCATOR
#define QDP_QCDOC_ALLOCATOR

#include "qdp_allocator.h"
#include "qdp_stdio.h"
#include "qdp_singleton.h"
#include <string>
#include <qalloc.h>

using namespace std;
QDP_BEGIN_NAMESPACE(QDP);
QDP_BEGIN_NAMESPACE(Allocator);


class QDPQCDOCAllocator {
private:
  // Disallow Copies
  QDPQCDOCAllocator(const QDPQCDOCAllocator& c) {};

  // Disallow assignments (copies by another name)
  QDPQCDOCAllocator& operator=(const QDPQCDOCAllocator& c) {};

 public:
  QDPQCDOCAllocator() {};
  //! Allocator function. Allocates n_bytes, into a memory pool
  //! This is a default implementation, with only 1 memory pool
  //! So we simply ignore the memory pool hint.
  inline void*
  allocate(size_t n_bytes,const MemoryPoolHint& mem_pool_hint) {
    
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
	QDPIO::cerr << "Unable to allocate memory with qalloc" << endl;
	QDP_abort(1);
      }
    }
    // Return the aligned pointer
    return (void *)aligned;
  }


  //! Free an aligned pointer, which was allocated by us.
  inline void 
  free(void *mem) { 
    qfree(mem);
  }
};

// Turn into a Singleton, CreateUsingNew, DefaultLifetime, SingleThreaded
typedef SingletonHolder<QDPQCDOCAllocator> theQDPAllocator;

QDP_END_NAMESPACE();
QDP_END_NAMESPACE();

#endif
