// Jacques Bloch - 24 Jan 2014
#include <vector>
#include "qdp.h"
#include "qdp_pool.h"

namespace QDP
{
	namespace Pool
	{
		static list<void*> avail[2], used[2];

		void *allocate(size_t n_bytes,const QDP::Allocator::MemoryPoolHint& mem_pool_hint)
		{
			bool usepool;
			int ipool;

			if (n_bytes == Layout::sitesOnNode()*Ns*Nc*2*sizeof(REAL))	// LatticeFermion
			{
				usepool=true;
				ipool=0;
			}
			else if (n_bytes == Layout::sitesOnNode()*Nc*Nc*2*sizeof(REAL))	// LatticeColorMatrix
			{
				usepool=true;
				ipool=1;
			}
			else
				usepool=false;

			if (!usepool)
			{
					return QDP::Allocator::theQDPAllocator::Instance().allocate(n_bytes,mem_pool_hint);
			}
			else
			{
				if(avail[ipool].empty())
				{
		used[ipool].push_back(QDP::Allocator::theQDPAllocator::Instance().allocate(n_bytes,mem_pool_hint));
				}
				else
				{
						used[ipool].push_back(avail[ipool].back());
						avail[ipool].pop_back();
				}
				cout << "pool " << ipool << " : total=" << avail[ipool].size()+used[ipool].size() << " free=" << avail[ipool].size() << " used=" << used[ipool].size() << endl;
				return used[ipool].back();
			}
		}

		void free(void *ptr, size_t n_bytes)
		{
			bool usepool;
			int ipool;

			if (n_bytes == Layout::sitesOnNode()*Ns*Nc*2*sizeof(REAL))	// LatticeFermion
			{
				usepool=true;
				ipool=0;
			}
			else if (n_bytes == Layout::sitesOnNode()*Nc*Nc*2*sizeof(REAL))	// LatticeColorMatrix
			{
				usepool=true;
				ipool=1;
			}
			else
				usepool=false;

			if (!usepool)
			{
				QDP::Allocator::theQDPAllocator::Instance().free(ptr);
			}
			else
			{
				list<void*>::iterator iter=find(used[ipool].begin(), used[ipool].end(),ptr);
				if (iter != used[ipool].end())
				{
					used[ipool].erase(iter);
					avail[ipool].push_back(ptr);
					cout << "pool " << ipool << " : total=" << avail[ipool].size()+used[ipool].size() << " free=" << avail[ipool].size() << " used=" << used[ipool].size() << endl;
				}
				else
					QDPIO::cout << "Object is not allocated!\n";
			}
		}
	}		// namespace Pool
}		// namespace QDP
