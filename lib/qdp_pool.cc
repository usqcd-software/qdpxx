// Jacques Bloch - 24 Jan 2014
/*
	Pool to allocate lattice objects
*/

#include <vector>
#include "qdp.h"
#include "qdp_pool.h"
//#define LARGE_POOL

namespace QDP
{
	namespace Pool
	{
		static int VERBOSE=0;

#ifdef LARGEPOOL
		int Npool = 7;
		Pool pool[]=
		{
			Pool("Fermion", sizeof(Fermion)),
			Pool("ColorMatrix", sizeof(ColorMatrix)),
			Pool("HalfFermion", sizeof(HalfFermion)),
			Pool("Complex", sizeof(RComplex<REAL>)),
			Pool("Real", sizeof(REAL)),
			Pool("int", sizeof(int)),
			Pool("byte", sizeof(char))
		};
#else
		int Npool = 2;
		Pool pool[]=
		{
			Pool("Fermion", sizeof(Fermion)),
			Pool("ColorMatrix", sizeof(ColorMatrix)),
		};
#endif

#ifdef ORIGINALPOOL
		#warning "USING ORIGINAL POOL"

		// constructor
		Pool::Pool(const string name, size_t n_bytes) : name(name), n_bytes(n_bytes) {}

		// destructor
		Pool::~Pool()
		{
				QDPIO::cout << "pool " << name << " released: total=" << avail.size()+used.size() << " free=" << avail.size() << " used=" << used.size() << endl;
		}

		void *Pool::allocate(size_t n_bytes,const QDP::Allocator::MemoryPoolHint& mem_pool_hint)
		{
			if(avail.empty())
			{
				used.push_back(QDP::Allocator::theQDPAllocator::Instance().allocate(n_bytes,mem_pool_hint));
			}
			else
			{
					used.push_back(avail.back());
					avail.pop_back();
			}
			if (VERBOSE) cout << "pool " << name << " : total=" << avail.size()+used.size() << " free=" << avail.size() << " used=" << used.size() << endl;
			return used.back();
		}

		void Pool::free(void *ptr)
		{
			Memcontainer::iterator iter=find(used.begin(), used.end(),ptr);
			if (iter != used.end())
			{
				used.erase(iter);
				avail.push_back(ptr);
				if (VERBOSE) cout << "pool " << name << " : total=" << avail.size()+used.size() << " free=" << avail.size() << " used=" << used.size() << endl;
			}
			else
				QDPIO::cout << "Object is not allocated!\n";
		}
#else

		#warning "USING NEW POOL"

		// constructor
		Pool::Pool(const string name, size_t n_bytes) : avail(0), n_bytes(n_bytes), name(name) {}

		// destructor
		Pool::~Pool()
		{
				QDPIO::cout << "pool " << name << " released: total=" << ptr.size() << " free=" << avail << " used=" << ptr.size()-avail << endl;
		}

		void *Pool::allocate(size_t n_bytes,const QDP::Allocator::MemoryPoolHint& mem_pool_hint)
		{
			if(!avail)
			{
				ptr.push_back(QDP::Allocator::theQDPAllocator::Instance().allocate(n_bytes,mem_pool_hint));
				flag_used.push_back(true);
				if (VERBOSE) cout << "pool " << name << ": total=" << ptr.size() << " free=" << avail << " used=" << ptr.size()-avail << endl;
				return ptr.back();
			}
			else
			{
				// search for unused element
				vector<bool>::iterator iter=find(flag_used.begin(), flag_used.end(), false);
				*iter = true;
				--avail;
				if (VERBOSE) cout << "pool " << name << ": total=" << ptr.size() << " free=" << avail << " used=" << ptr.size()-avail << endl;
				return ptr[iter-flag_used.begin()];
			}
		}

		void Pool::free(void *mem)
		{
			vector<void*>::iterator iter=find(ptr.begin(), ptr.end(),mem);
			if (iter != ptr.end())
			{
				flag_used[iter-ptr.begin()] = false;
				++avail;
				if (VERBOSE) cout << "pool " << name << ": total=" << ptr.size() << " free=" << avail << " used=" << ptr.size()-avail << endl;
			}
			else
				QDPIO::cout << "Object is not allocated!\n";
		}
#endif

	}		// namespace Pool
}		// namespace QDP
