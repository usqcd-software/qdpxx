/*! @file
 * @brief QCDOC memory allocator
 */

#include "qdp.h"
#include "qdp_config.h"
#include "qdp_default_allocator.h"
#include "qdp_pool_allocator.h"
#include <vector>
#include <map>
#include <new>
#include <cstdio>


#undef DEBUG_POOL_ALLOCATOR

namespace QDP {
namespace Allocator {

	struct MemInfo
	{
		size_t Size;
		unsigned char* Unaligned;
	};

	using MapT = std::map<void *,MemInfo>;

	MapT theAllocMap;


	QDPPoolAllocator::QDPPoolAllocator() : _PoolSize(0),
			_MyMem(nullptr), _LargePool(nullptr){}


	QDPPoolAllocator::~QDPPoolAllocator() {
		if ( _LargePool ) delete _LargePool;
		if ( _MyMem ) {
			delete [] _MyMem;
		}
		_LargePool = nullptr;
		_MyMem = nullptr;
		_PoolSize = 0;
	}

	void
	QDPPoolAllocator::init(size_t PoolSizeInMB)
	{

		QDPIO::cout << "Initializing TBB Pool Allocator" << std::endl;
		if( _MyMem != nullptr ) {
			QDPIO::cout << "Allocator Already Inited. Aborting" << std::endl;
			QDP_abort(1);
		}
		theAllocMap.clear();

		QDPIO::cout << std::flush;
		_PoolSize = PoolSizeInMB*1024*1024;

		QDPIO::cout << "Intializing TBB Fixed Pool Allocator: Allocating " << PoolSizeInMB << " MB"  << std::endl;

		_MyMem = new (std::nothrow) unsigned char [ _PoolSize ];
		if ( _MyMem == nullptr) {
			QDPIO::cout << "Unable to allocate " << _PoolSize <<" bytes" << std::endl;
			QDPIO::cout << "Aborting" <<std::endl;
			QDP_abort(1);
		}

		_LargePool = new (std::nothrow ) tbb::fixed_pool((void *)_MyMem, _PoolSize);
		if ( _LargePool == nullptr) {
				QDPIO::cout << "Unable to allocate fixed pool" << std::endl;
				QDPIO::cout << "Aborting" <<std::endl;
				QDP_abort(1);
		}
	}

	void*
	QDPPoolAllocator::allocate(size_t n_bytes,
				const MemoryPoolHint& mem_pool_hint=DEFAULT)
	{
	    size_t BytesToAlloc;
	    BytesToAlloc = n_bytes;
	    BytesToAlloc += QDP_ALIGNMENT_SIZE;


	    unsigned char* Unaligned = (unsigned char *)(_LargePool->malloc(BytesToAlloc));

	    unsigned char* Aligned = (unsigned char *)
	    				( ( (unsigned long)Unaligned + (QDP_ALIGNMENT_SIZE-1) ) & ~(QDP_ALIGNMENT_SIZE - 1));

#ifdef DEBUG_POOL_ALLOCATOR
	    QDPIO::cout << " Allocated: " << BytesToAlloc << " Bytes, Unaligend=" <<(unsigned long) Unaligned
	    			<< " Aligned=" << (unsigned long) Aligned << std::endl;
#endif
	    // Insert into the map
	    auto r = theAllocMap.insert(std::make_pair(Aligned, MemInfo{BytesToAlloc,Unaligned}));

	    // Check success of insertion.
	    if( ! r.second ) {
	      //QDPIO::cerr << "Failed to insert (unaligned,aligned) pair into map" << std::endl;
	      QDP_abort(1);
	    }

	    // Return the aligned pointer
	    return (void *)Aligned;
	  }

	void QDPPoolAllocator::free(void *mem)
	{
		try {
			// Look it up.
			auto d = theAllocMap[(unsigned char *)mem];
#ifdef DEBUG_POOL_ALLOCATOR
			QDPIO::cout << "PoolAlloc::free: Descriptor Found: Size="<< d.Size
						<< "  Unaligned =" << std::hex <<(unsigned long)d.Unaligned << std::endl;
#endif

			_LargePool->free(d.Unaligned);

#ifdef DEBUG_POOL_ALLOCATOR
			QDPIO::cout << "PoolAlloc::free: Erasing Addr from theAllocMap... ";
#endif

			// Delete _InUseMap entry
			theAllocMap.erase((unsigned char *)mem);

#ifdef DEBUG_POOL_ALLOCATOR
			QDPIO::cout << " ... done" <<std::endl;
#endif
		}
		catch(...) {
			//QDPIO::cout << "QDPPoolAllocate::free threw exception" << std::endl;
			QDP_abort(1);
		}
	}

	void QDPPoolAllocator::pushFunc(const char * func,int line) {}
	void QDPPoolAllocator::popFunc(void) {}
	void QDPPoolAllocator::dump(void)
	{
		if ( Layout::primaryNode() ) {

			QDPIO::cout << "Dumping memory map" << std::endl;
			for( auto j = theAllocMap.begin(); j != theAllocMap.end(); j++) {
				const unsigned long unaligned = (unsigned long)((*j).first);
				const MemInfo& mem = (*j).second;

				printf("mem= 0x%lx  unaligned= 0x%lx size=%d\n", (unsigned long)unaligned,
						(unsigned long)mem.Unaligned, mem.Size);
			}
		}
	}
} // namespace Allocator
} // namespace QDP
