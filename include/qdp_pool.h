/*
	Pool to allocate lattice objects
*/
#ifndef INCL_QDP_POOL
#define INCL_QDP_POOL

#include "qdp.h"

#define ORIGINALPOOL
//#define TEMPLATEDPOOL

namespace QDP
{
	namespace Pool
	{
		class Pool
		{

#ifdef ORIGINALPOOL
		typedef list<void*> Memcontainer;
		//typedef vector<void*> Memcontainer;
		Memcontainer avail, used;
#else
			vector<void*> ptr;
			vector<bool> flag_used;
			int avail;
#endif

			public:
			Pool(const string name, size_t n_bytes);
			~Pool();
			void *allocate(size_t n_bytes,const QDP::Allocator::MemoryPoolHint& mem_pool_hint);
			void free(void *mem);

			string name;
			size_t n_bytes;
		};

		extern Pool pool[];
		extern int Npool;

		template <class T>int get_pool()
		{
			for (int ipool=0; ipool<Npool; ++ipool)
			{
				if (sizeof(T) == pool[ipool].n_bytes)
					return ipool;
			}
			return -1;
		}

		template <class T>void *allocate(size_t n_bytes,const QDP::Allocator::MemoryPoolHint& mem_pool_hint)
		{
			void *ptr;
			int VERBOSE=0;

			int ipool = get_pool<T>();

			if (ipool==-1)
			{
				ptr = QDP::Allocator::theQDPAllocator::Instance().allocate(n_bytes,mem_pool_hint);
				if (VERBOSE) cout << "allocate mem=" << ptr << " size=" << n_bytes << endl;
			}
			else
				ptr = pool[ipool].allocate(n_bytes, mem_pool_hint);

			return ptr;
		}

		template <class T>void free(void *ptr)
		{
			int ipool = get_pool<T>();

			if (ipool==-1)
			{
				QDP::Allocator::theQDPAllocator::Instance().free(ptr);
			}
			else
			{
				pool[ipool].free(ptr);
			}
		}

#ifdef TEMPLATEDPOOL
		#warning "USING TEMPLATED POOL"

		//LatticeFermion -> pool 0
		template<> void *allocate<PSpinVector< PColorVector< RComplex<REAL>, Nc>, Ns> >(size_t n_bytes,const QDP::Allocator::MemoryPoolHint& mem_pool_hint)
		{
			return pool[0].allocate(n_bytes, mem_pool_hint);
		}
		template <> void free<PSpinVector< PColorVector< RComplex<REAL>, Nc>, Ns> >(void *ptr)
		{
			pool[0].free(ptr);
		}

		// LatticeColorMatrix -> pool 1
		template<> void *allocate<PScalar< PColorMatrix< RComplex<REAL>, Nc> > >(size_t n_bytes,const QDP::Allocator::MemoryPoolHint& mem_pool_hint)
		{
			return pool[1].allocate(n_bytes, mem_pool_hint);
		}
		template <> void free<PScalar< PColorMatrix< RComplex<REAL>, Nc> > >(void *ptr)
		{
			pool[1].free(ptr);
		}
#endif

	}		// namespace Pool
}		// namespace QDP
#endif