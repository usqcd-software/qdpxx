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

  // Disallow construction for everyone except friends
  QDPQCDOCAllocator() {};
  
  // Disallow destruction for everyone except Friends
  ~QDPQCDOCAllocator() {};

  // The only friend is the singleton creation policy 
  friend class QDP::CreateUsingNew<QDP::Allocator::QDPQCDOCAllocator>;

 public:

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

// Turn into a Singleton. Create with CreateUsingNew
// Has NoDestroy lifetime, as it may be needed for 
// the destruction policy is No Destroy, so the 
// Singleton is not cleaned up on exit. This is so 
// that static objects can refer to it with confidence
// in their own destruction, not having to worry that
// atexit() may have destroyed the allocator before
// the static objects need to feed memory. 
typedef SingletonHolder<QDPQCDOCAllocator,
			QDP::CreateUsingNew,
			QDP::NoDestroy,
			QDP::SingleThreaded> theQDPAllocator;

QDP_END_NAMESPACE();

QDP_BEGIN_NAMESPACE(Hints);
//! Hint to move an object of type OLattice to fast memory. 
/*!
 * \ingroup QDP Memory management hints
 * 
 * This is a specialised function for OLattice objects (eg fermion fields,
 *  lattice color matrix fields and the like) hinting to place them into 
 *  fast memory. The hint is forwarded to the object.
 *
 * \param x   The object for which the hint is meant
 * \param copy Whether to copy the object's slow memory contents to 
 *             its new fast memory home. Default is no.
 */
template<typename T>
inline
void moveToFastMemoryHint(OLattice<T>& x, bool copy=false) {
  x.moveToFastMemoryHint(copy);
}

//! Hint to revert an object of type OLattice from fast memory. 
/* \ingroup QDP Memory management hints
 * 
 *  This is a specialised function for OLattice objects (eg fermion fields,
 *  lattice color matrix fields and the like) hinting to revert them from 
 *  fast memory into back into slow memory. The hint is forwarded to the 
 *  object.
 *
 * \param x   The object for which the hint is meant
 * \param copy Whether to copy the object's fast memory contents back to 
 *             its new slow memory home. Default is no.
 */
template<typename T>
inline
void revertFromFastMemoryHint(OLattice<T>& x, bool copy=false) { 
  x.revertFromFastMemoryHint(copy);
}

//! Hint to move a multi1d<OLattice<T> > object to fast memory.
/*!
 * \ingroup QDP Memory management hints
 * 
 * This is a specialised function for multi1d<OLattice> objects 
 * (eg 5D fermion fields and arrays LatticeColorMatrix fields)
 *  hinting to place them into 
 *  fast memory. The hint is forwarded to the object.
 *
 * \param x   The object for which the hint is meant
 * \param copy Whether to copy the object's slow memory contents to 
 *             its new fast memory home. Default is no.
 */
template<typename T>
inline
void moveToFastMemoryHint(multi1d<OLattice<T> >&x, bool copy=false) { 
  x.moveToFastMemoryHint(copy);
}

//! Hint to revert a multi1d<OLattice<T> > object from fast memory.
/* \ingroup QDP Memory management hints
 * 
 *  This is a specialised function for multi1d<OLattice> objects 
 *  (eg 5D fermion fields and arrays of LatticeColorMatrix fields and the 
 *  like) hinting to revert them from  fast memory into back into slow memory.
 *  The hint is forwarded to the object.
 *
 * \param x   The object for which the hint is meant
 * \param copy Whether to copy the object's fast memory contents back to 
 *             its new slow memory home. Default is no.
 */
template<typename T>
inline
void revertFromFastMemoryHint(multi1d<OLattice<T> >&x, bool copy=false) { 
  x.revertFromFastMemoryHint(copy);
}
QDP_END_NAMESPACE();

QDP_END_NAMESPACE();

#endif
