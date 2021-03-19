// -*- C++ -*-

/*! \file
 * \brief Default memory allocator for QDP
 *
 */

#ifndef QDP_DEFAULT_ALLOCATOR
#define QDP_DEFAULT_ALLOCATOR

#include "qdp_config.h"
#include "qdp_singleton.h"

#if defined(QDP_DEBUG_MEMORY)
#include <stack>
#endif

#include "qdp_allocator.h"
#include "qdp_stdio.h"

#include <map>
#include <stdlib.h> // aligned_alloc is in cstdlib since c++17 and in stdlib.h since C11
#include <string>

namespace QDP
{
  namespace Allocator
  {

    namespace detail
    {
      inline std::map<void*, std::size_t>& getAllocs()
      {
	static std::map<void*, std::size_t> allocs;
	return allocs;
      }

      inline std::size_t& getCurrentlyAllocated()
      {
	static std::size_t s = 0;
	return s;
      }
    }

    //! Allocate and initialize n instances of T, at least with alignment QDP_ALIGNMENT_SIZE
    template <typename T>
    T* new_aligned(std::size_t n)
    {
      if (n == 0)
	return nullptr;
      T* p =
	(T*)aligned_alloc(std::max((std::size_t)QDP_ALIGNMENT_SIZE, alignof(T)), sizeof(T) * n);
      if (p == nullptr)
	QDP_error_exit("Bad allocation! Currently there are %g MiB allocated",
		       (double)detail::getCurrentlyAllocated() / 1024 / 1024);
      detail::getCurrentlyAllocated() += sizeof(T) * n;
      new (p) T[n];
      detail::getAllocs()[(void*)p] = n;
      return p;
    }

    //! Destroy the instances allocated with new_aligned
    template <typename T>
    void delete_aligned(T* p)
    {
      if (p == nullptr)
	return;
      std::size_t n = detail::getAllocs()[(void*)p];
      if (detail::getAllocs().erase((void*)p) != 1)
	QDP_error_exit("Pointer not previously allocated with new_aligned");
      while (n--)
	p[n].~T();
      std::free(p);
      detail::getCurrentlyAllocated() -= sizeof(T) * n;
    }

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

      // Disallow creation / destruction by anyone except SingletonHolder
      friend class QDP::SingletonHolder<QDP::Allocator::QDPDefaultAllocator>;
      QDPDefaultAllocator() {}
      ~QDPDefaultAllocator()
      {
	if (the_alignment_map.size() > 0)
	  QDPIO::cerr << "warning: QDPDefaultAllocator still has allocations" << std::endl;
      }

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
