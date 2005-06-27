#ifndef QDP_ALLOCATOR
#define QDP_ALLOCATOR

/*! QDP Allocator
 *  A raw memory allocator for QDP, for particular use 
 *  with OLattice Objects. The pointers returned by allocate
 *  are all allocated with the correct alignment
 *  On normal targets this should be QDP_ALIGNMENT
 *  whereas on other targets (QCDOC) this should be the default
 *  alignment
 */
using namespace std;

QDP_BEGIN_NAMESPACE(QDP);
QDP_BEGIN_NAMESPACE(Allocator);

enum MemoryPoolHint { DEFAULT, FAST };

QDP_END_NAMESPACE(); // Allocator
QDP_END_NAMESPACE(); // QDP

#ifdef QDP_USE_QCDOC
// Include the QCDOC specialisation
#include "qdp_qcdoc_allocator.h"
#else
// Include the default specialisation
#include "qdp_default_allocator.h"
#endif


#endif
