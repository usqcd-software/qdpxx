// -*- C++ -*-

/*! \file
 * \brief Default memory allocator for QDP
 *
 */

#ifndef QDP_DEFAULT_ALLOCATOR
#define QDP_DEFAULT_ALLOCATOR

#include "qdp_config.h"

#if defined(QDP_DEBUG_MEMORY)
#include <stack>
#endif

#include "qdp_allocator.h"
#include "qdp_stdio.h"

#include <string>
#include <map>

namespace __QDP__
{
  namespace Allocator
  {

#if defined(QDP_DEBUG_MEMORY)
    // Struct to hold in map
    struct MapVal {
      MapVal(unsigned char* u, const std::string& f, int l, size_t b) :
	unaligned(u), func(f), line(l), bytes(b) {}

      unsigned char* unaligned;
      std::string    func;
      int            line;
      size_t         bytes;
    };

    // Convenience typedefs to save typing

    // The type of the map to hold the aligned unaligned values
    typedef std::map<unsigned char*, MapVal> MapT;

    // Func info
    struct FuncInfo_t {
      FuncInfo_t(const char* f, int l) : func(f), line(l) {}

      std::string  func;
      int          line;
    };
#else
    typedef std::map<unsigned char*, unsigned char *> MapT;
#endif


    // Specialise allocator to the default case
    class QDPDefaultAllocator {
    private:

#if defined(QDP_DEBUG_MEMORY)
      std::stack<FuncInfo_t> infostack;
#endif

      MapT the_alignment_map;

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

      friend class __QDP__::CreateUsingNew<__QDP__::Allocator::QDPDefaultAllocator>;
      friend class __QDP__::CreateStatic<__QDP__::Allocator::QDPDefaultAllocator>;
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
} // namespace __QDP__

#endif
