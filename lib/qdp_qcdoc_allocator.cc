// $Id: qdp_qcdoc_allocator.cc,v 1.8 2008-06-27 13:31:22 bjoo Exp $
/*! @file
 * @brief QCDOC memory allocator
 */

#include "qdp.h"

#if defined(QDP_DEBUG_MEMORY)
#include "stack"
#endif


namespace QDP {
namespace Allocator {

#if defined(QDP_DEBUG_MEMORY)
 // Struct to hold in map
  struct MapVal {
    MapVal(const std::string& f, int l, size_t b) : func(f), line(l), bytes(b) {}

    std::string   func;
    int           line;
    size_t        bytes;
  };

  // The type of the map to hold the aligned size values
  typedef map<unsigned char*, MapVal> MapT;

  // Func info
  struct FuncInfo_t {
    FuncInfo_t(const char* f, int l) : func(f), line(l) {}

    std::string  func;
    int          line;
  };

  // A stack to hold fun info
  std::stack<FuncInfo_t> infostack;

#else

  // The type of the map to hold the aligned size values
  typedef map<unsigned char*, size_t> MapT;
#endif


  // The type returned on map insertion, allows me to check
  // the insertion was successful.
  typedef pair<MapT::iterator, bool> InsertRetVal;

  // Anonymous namespace
  namespace {
    MapT the_alignment_map;
  }

  //! Allocator function. Allocates n_bytes, into a memory pool
  //! This is a default implementation, with only 1 memory pool
  //! So we simply ignore the memory pool hint.
  void*
  QDPQCDOCAllocator::allocate(size_t n_bytes,const MemoryPoolHint& mem_pool_hint) 
  {
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

    // If we cannot get the memory we want, we throw an exception
    // in the standard 'new' way.

    // Interface is therefore changed tho
    if( aligned == (unsigned char *)NULL ) { 
      throw std::bad_alloc();
    }

#if defined(QDP_DEBUG_MEMORY)
    // Current location
    FuncInfo_t& info = infostack.top();

    // Insert into the map
    InsertRetVal r = the_alignment_map.insert(make_pair(aligned, MapVal(info.func, info.line, n_bytes)));
#else
    // Insert into the map
    InsertRetVal r = the_alignment_map.insert(make_pair(aligned, n_bytes));
#endif

    // Check success of insertion.
    if( ! r.second ) {
      QDPIO::cerr << "Failed to insert (aligned,n_bytes) pair into map" << endl;
      QDP_abort(1);
    }

    // Return the aligned pointer
    return (void *)aligned;
  }


  //! Free an aligned pointer, which was allocated by us.
  void
  QDPQCDOCAllocator::free(void *mem)
  {
    // Look up the original aligned pointer in the memory.
    MapT::iterator iter = the_alignment_map.find((unsigned char*)mem);
    if( iter != the_alignment_map.end() ) {
      // Find the original aligned pointer
      unsigned char *aligned = iter->first;

      // Remove its entry from the map
      the_alignment_map.erase(iter);

      // Delete the actual aligned pointer
      qfree(aligned);
    }
    else {
      QDPIO::cerr << "Pointer not found in map" << endl;
      QDP_abort(1);
    }
  }


#if defined(QDP_DEBUG_MEMORY)
  //! Dump the map
  void
  QDPQCDOCAllocator::dump()
  {
     QMP_barrier();
     if ( Layout::primaryNode() )
     {
       size_t sum = 0;
       typedef MapT::const_iterator CI;
       QDPIO::cout << "Dumping memory map" << endl;
//       FILE *fp = fopen("memory.out", "wb");
       FILE *fp = stdout;
       fprintf(fp,"Dumping memory map\n");
       for( CI j = the_alignment_map.begin();
             j != the_alignment_map.end(); j++)
       {
         sum += j->second.bytes;
         fprintf(fp,"mem= 0x%x  bytes= %d  bytes/site= %d  line= %d  func= %s\n", j->first, 
                j->second.bytes, j->second.bytes/Layout::sitesOnNode(), 
                j->second.line, j->second.func.c_str());
         fprintf(fp,"\n");
       }
       fprintf(fp,"total bytes= %d\n", sum);
//       fclose(fp);
     }
     QMP_barrier();
  }

  // Setter
  void
  QDPQCDOCAllocator::pushFunc(const char* func, int line)
  {
    infostack.push(FuncInfo_t(func,line));
  }

  // Nuker
  void
  QDPQCDOCAllocator::popFunc()
  {
    if (infostack.empty())
    {
      QDPIO::cerr << __func__ << ": invalid pop" << endl;
      QDP_abort(1);
    }
  
    infostack.pop();
  }


  static const char* nowhere = "nowhere";

  // Init
  void
  QDPQCDOCAllocator::init()
  {
    infostack.push(FuncInfo_t(nowhere,0));
  }

#else
  // No memory debugging

  //! Dump the map
  void
  QDPQCDOCAllocator::dump()
  {
     if ( Layout::primaryNode() )
     {
       typedef MapT::const_iterator CI;
       QDPIO::cout << "Dumping memory map" << endl;
       for( CI j = the_alignment_map.begin();
             j != the_alignment_map.end(); j++)
       {
         printf("mem= 0x%x  bytes = %d\n", j->first, j->second);
       }
     }
  }

  // Setter
  void
  QDPQCDOCAllocator::pushFunc(const char* func, int line) {}

  // Nuker
  void
  QDPQCDOCAllocator::popFunc() {}

  // Init
  void
  QDPQCDOCAllocator::init() {}

#endif


} // namespace Allocator
} // namespace QDP
