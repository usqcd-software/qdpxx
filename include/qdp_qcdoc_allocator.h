// -*- C++ -*-

/*! @file
 * @brief QCDOC memory allocator
 */

#ifndef QDP_QCDOC_ALLOCATOR
#define QDP_QCDOC_ALLOCATOR

#include "qdp_allocator.h"
#include "qdp_stdio.h"
#include "qdp_singleton.h"
#include <string>
#include <qalloc.h>
#include <map>

using namespace std;


namespace QDP {
namespace Allocator {


class QDPQCDOCAllocator {
private:

  // Disallow Copies
  QDPQCDOCAllocator(const QDPQCDOCAllocator& c) {}

  // Disallow assignments (copies by another name)
  void operator=(const QDPQCDOCAllocator& c) {}

  // Disallow construction for everyone except friends
  QDPQCDOCAllocator() {init();}
  
  // Disallow destruction for everyone except Friends
  ~QDPQCDOCAllocator() {}

  // The only friend is the singleton creation policy 
  friend class QDP::CreateUsingNew<QDP::Allocator::QDPQCDOCAllocator>;

 public:

  // Pusher
  void pushFunc(const char* func, int line);
  
  // Popper
  void popFunc();
  
  //! Allocator function. Allocates n_bytes, into a memory pool
  void*
  allocate(size_t n_bytes,const MemoryPoolHint& mem_pool_hint);

  //! Free an aligned pointer, which was allocated by us.
  void 
  free(void *mem);

  //! Dump the map
  void
  dump();

protected:
  void init();
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

} // namespace Allocator

namespace Hints {

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

} // namespace Hints
} // namespace QDP

#endif
