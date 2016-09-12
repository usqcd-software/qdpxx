// -*- C++ -*-

/*! \file
 * \brief Default memory allocator for QDP
 *
 */

#ifndef QDP_DEFAULT_ALLOCATOR
#define QDP_DEFAULT_ALLOCATOR

#include "qdp_allocator.h"
#include "qdp_stdio.h"

#include <string>
#include <map>

namespace QDP
{
  namespace Allocator
  {

    // Specialise allocator to the default case
    class QDPDefaultAllocator {
    private:
      // Disallow Copies
      QDPDefaultAllocator(const QDPDefaultAllocator& c) {}

      // Disallow assignments (copies by another name)
      void operator=(const QDPDefaultAllocator& c) {}

      // Disallow creation / destruction by anyone except 
      // the singleton CreateUsingNew policy which is a "friend"
      // I don't like friends but this follows Alexandrescu's advice
      // on p154 of Modern C++ Design (A. Alexandrescu)
      QDPDefaultAllocator() {}
      ~QDPDefaultAllocator() {}

      friend class QDP::CreateUsingNew<QDP::Allocator::QDPDefaultAllocator>;
    public:

      void init(size_t PoolSizeinMB);

      // Pusher
      void pushFunc(const char* func, int line);
  
      // Popper
      void popFunc();
  
      //! Allocator function. Allocates n_bytes, into a memory pool
      //! This is a default implementation, with only 1 memory pool
      //! So we simply ignore the memory pool hint.
      void*
      allocate(size_t n_bytes,const MemoryPoolHint& mem_pool_hint);

      //! Free an aligned pointer, which was allocated by us.
      void 
      free(void *mem);

      //! Dump the map
      void
      dump();



    };


  } // namespace Allocator
} // namespace QDP

#endif
