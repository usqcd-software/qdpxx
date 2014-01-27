#ifndef INCL_QDP_POOL
#define INCL_QDP_POOL

namespace QDP
{
	namespace Pool
	{
		void *allocate(size_t n_bytes,const QDP::Allocator::MemoryPoolHint& mem_pool_hint);
		void free(void *ptr, size_t n_bytes);
	}		// namespace Pool
}		// namespace QDP
#endif