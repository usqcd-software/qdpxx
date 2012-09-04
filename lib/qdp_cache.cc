#include "qdp.h"

#include <map>
#include <list>
#include <functional>

#include <iostream>
#include <fstream>

namespace QDP
{


  struct QDPCache::Entry {
    int    Id;
    size_t size;
    int    flags;
    void*  hstPtr;  // NULL if not allocated
    void*  devPtr;  // NULL if not allocated
    int    lockCount;
    list<int>::iterator iterTrack;
  };


  QDPCache& QDPCache::Instance()
  {
    static QDPCache singleton;
    return singleton;
  }

  void QDPCache::beginNewLockSet() {
    prevLS = currLS;
    currLS = currLS == 0 ? 1 : 0;
  }

  void QDPCache::releasePrevLockSet() {
    if (prevLS < 0)
      return;

    while ( vecLockSet[prevLS].size() > 0 ) {

      Entry& e = vecEntry[ vecLockSet[prevLS].back() ];
      e.lockCount--;

#ifdef SANITY_CHECKS_CACHE
      // SANITY
      if (e.lockCount < 0)
	QDP_error_exit("cache releasePrevLockSet lockCount < 0");
#endif

      vecLockSet[prevLS].pop_back();
    }
  }

  void QDPCache::printLockSets() {
    for ( int ls = 0 ; ls < 2 ; ls++ ) {
      int n=0;
      if (ls==currLS)
	QDP_info("Lock set (current):");
      else
	QDP_info("Lock set (previous):");
      for (vector<int>::iterator i = vecLockSet[ls].begin() ; i != vecLockSet[ls].end() ; ++i ) {
	Entry& e = vecEntry[*i];
	bool inDel = find(lstDel.begin(),lstDel.end(),*i) != lstDel.end();
	QDP_info("%d: id=%u size=%u lockCount=%u  signed off=%u",n++,(unsigned)e.Id,(unsigned)e.size,(unsigned)e.lockCount,inDel );
      }
    }
  }


  bool QDPCache::allocate_device_static( void** ptr, size_t n_bytes ) {
    while (!CUDADevicePoolAllocator::Instance().allocate( ptr , n_bytes )) {
      if (!spill_lru()) {
	QDP_error_exit("cache allocate_device_static: can't spill LRU object");
      }
    }
    lstStatic.push_back(*ptr);
    return true;
  }

  void QDPCache::free_device_static( void* ptr ) {
    CUDADevicePoolAllocator::Instance().free( ptr );

    list<void*>::iterator i = find( lstStatic.begin() , lstStatic.end() , ptr );

    if (i == lstStatic.end())
      QDP_error_exit("Cache static free: Pointer not found in record");

    lstStatic.erase(i);
  }



  void QDPCache::backupOnHost() {
#ifdef SANITY_CHECKS_CACHE
    // SANITY
    if (vecLockSet[0].size() + vecLockSet[1].size() == 0)
      QDP_error_exit("cache backup: Nothing locked! Really what you intend to do?");
    if (vecLockSet[currLS].size() == 0)
      QDP_error_exit("cache backup: current lock set is empty! Really what you intend to do?");
#endif

    for ( int ls = 0 ; ls < 2 ; ls++ ) {
      for (vector<int>::iterator i = vecLockSet[ls].begin() ; i != vecLockSet[ls].end() ; ++i ) {

	Entry& e = vecEntry[*i];

#ifdef SANITY_CHECKS_CACHE
	// SANITY
	if(!e.devPtr) 
	  QDP_error_exit("cache backup: Object has no device pointer");
#endif
	// Don't backup object which registrated with its own host memory
	// since it might already be destructed, and its host memory
	if ( e.flags != 2 ) {

	  listBackup.push_back( new char[e.size] );

	  // For backup/restore we use sync transfers
#ifdef GPU_DEBUG_DEEP
	  QDP_debug_deep("backup: D2H %p <- %p , %u" , listBackup.back() , e.devPtr , (unsigned)e.size);
#endif
	  CudaMemcpy( listBackup.back() , e.devPtr , e.size );
	}
      }
    }
  }

  void QDPCache::restoreFromHost() {
#ifdef SANITY_CHECKS_CACHE
    // SANITY
    if (vecLockSet[0].size() + vecLockSet[1].size() == 0)
      QDP_error_exit("cache backup: Nothing locked! Really what you intend to do?");
    if (vecLockSet[currLS].size() == 0)
      QDP_error_exit("cache backup: current lock set is empty! Really what you intend to do?");
#endif

    for ( int ls = 0 ; ls < 2 ; ls++ ) {
      for (vector<int>::iterator i = vecLockSet[ls].begin() ; i != vecLockSet[ls].end() ; ++i ) {
	Entry& e = vecEntry[*i];

#ifdef SANITY_CHECKS_CACHE
	// SANITY
	if(!e.devPtr) 
	  QDP_error_exit("cache backup: Object has no device pointer");
	if(!e.hstPtr) 
	  QDP_error_exit("cache backup: Object has no host pointer");
#endif

	// Don't restore object which registrated with its own host memory
	// since it might already be destructed, and its host memory
	if ( e.flags != 2 ) {

	  // For backup/restore we use sync transfers
#ifdef GPU_DEBUG_DEEP
	  QDP_debug_deep("restore: H2D %p -> %p , %u", listBackup.front() , e.devPtr , (unsigned)e.size);
#endif
	  CudaMemcpy( e.devPtr , listBackup.front() , e.size );
	  delete[] listBackup.front();
	  listBackup.pop_front();
	}
      }
    }
  }

  void QDPCache::sayHi () {
    CUDADevicePoolAllocator::Instance().sayHi();
    CUDAHostPoolAllocator::Instance().sayHi();
  }



  bool QDPCache::onDevice(int id) const {

    const Entry& e = vecEntry[id];

#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("cache onDevice id=%u devPtr=%p size=%u",(unsigned)id,e.devPtr, (unsigned)e.size);
#endif

    return (e.devPtr != NULL);
  }

    
  void QDPCache::enlargeStack()
  {
    const int portion = 1024;
    QDP_info_primary("enlarging stack by %d entries",portion);
    vecEntry.resize( vecEntry.size() + portion );
    for ( int i = 0 ; i < portion ; i++ ) {
      stackFree.push( vecEntry.size()-i-1 );
    }
  }


  int QDPCache::registrate( size_t size, unsigned flags)
  {

#ifdef SANITY_CHECKS_CACHE
    if (size==0)
      QDP_error_exit("cache registrate ( size == 0 ),%u",flags);
#endif

    if (stackFree.size() == 0) {
      enlargeStack();
    }

    int Id = stackFree.top();
    Entry& e = vecEntry[ Id ];

#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("cache registrate size=%lu flags=%u Id=%u",(long)size,flags,(unsigned)Id );
#endif

#ifdef SANITY_CHECKS_CACHE
    // SANITY
    if (find(lstTracker.begin(),lstTracker.end(),Id) != lstTracker.end())
      QDP_error_exit("cache reg: already in tracker");
#endif
      
    e.Id        = Id;
    e.size      = size;
    e.flags     = flags;
    e.hstPtr    = NULL;
    e.devPtr    = NULL;
    e.lockCount = 0;
    e.iterTrack = lstTracker.insert( lstTracker.end() , Id );
      
    stackFree.pop();

#ifdef GPU_DEBUG_DEEP
    printTracker();
#endif

    return Id;
  }



  int QDPCache::registrateOwnHostMem( size_t size, void* ptr)
  {
    if (stackFree.size() == 0) {
      enlargeStack();
    }

    int Id = stackFree.top();
    Entry& e = vecEntry[ Id ];

#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("cache registrate ownHost size=%lu Id=%u",(long)size,(unsigned)Id );
#endif

#ifdef SANITY_CHECKS_CACHE
    // SANITY
    if (find(lstTracker.begin(),lstTracker.end(),Id) != lstTracker.end())
      QDP_error_exit("cache reg: already in tracker");
#endif

    e.Id        = Id;
    e.size      = size;
    e.flags     = 2;
    e.hstPtr    = ptr;
    e.devPtr    = NULL;
    e.lockCount = 0;
    e.iterTrack = lstTracker.insert( lstTracker.end() , Id );
      
    stackFree.pop();

#ifdef GPU_DEBUG_DEEP
    printTracker();
#endif

    return Id;
  }
    


  void QDPCache::signoff(int id) {
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("cache signoff id=%lu",(long)id );
#endif

#ifdef SANITY_CHECKS_CACHE
    // SANITY      
    if (find(lstTracker.begin(),lstTracker.end(),id) == lstTracker.end())
      QDP_error_exit("cache signoff: not in tracker");
    if (find(lstDel.begin(),lstDel.end(),id) != lstDel.end())
      QDP_error_exit("cache signoff: already in lstDel");
#endif

    lstDel.push_back( id );
    lstTracker.erase( vecEntry[id].iterTrack );

    deleteObjects();
#ifdef GPU_DEBUG_DEEP
    printLockSets();
#endif
  }


  void * QDPCache::getDevicePtr(int id) {
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("cache get device ptr id=%lu",(long)id );
#endif

    if (id < 0) return NULL;

#ifdef SANITY_CHECKS_CACHE
    // SANITY
    if (find(lstTracker.begin(),lstTracker.end(),id) == lstTracker.end())
      QDP_error_exit("cache getDevicePtr: id not in tracker");
    if (find(lstDel.begin(),lstDel.end(),id) != lstDel.end())
      QDP_error_exit("cache getDevicePtr: id already in lstDel!");
    if (id >= vecEntry.size())
      QDP_error_exit("cache getDevicePtr: out of range");
#endif

    Entry& e = vecEntry[id];

    assureDevice( e );

    lstTracker.splice( lstTracker.end(), lstTracker , e.iterTrack );

#ifdef GPU_DEBUG_DEEP
    printTracker();
    printLockSets();
#endif

    //CudaDeviceSynchronize();
    //CudaSyncTransferStream();

    return e.devPtr;
  }



  bool QDPCache::getHostPtr(void ** ptr , int id) {
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("cache get host ptr id=%lu",(long)id );
#endif

#ifdef SANITY_CHECKS_CACHE
    // SANITY
    if (find(lstTracker.begin(),lstTracker.end(),id) == lstTracker.end())
      QDP_error_exit("cache getDevicePtr: id not in tracker");
    if (find(lstDel.begin(),lstDel.end(),id) != lstDel.end())
      QDP_error_exit("cache getDevicePtr: id already in lstDel!");
    if (id >= vecEntry.size())
      QDP_error_exit("cache getDevicePtr: out of range");
#endif

    Entry& e = vecEntry[id];

    bool in_flight = assureHost( e );

    *ptr = e.hstPtr;
    return in_flight;
  }








  void QDPCache::freeHostMemory(Entry& e) {
    switch(e.flags) {
    case 0:
#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep("cache delete obj host memory flag=0");
#endif
      CUDAHostPoolAllocator::Instance().free( e.hstPtr );
      e.hstPtr=NULL;
      break;
    case 1:
#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep("cache delete obj host memory flag=1");
#endif
      QDP::Allocator::theQDPAllocator::Instance().free( e.hstPtr );
      e.hstPtr=NULL;
      break;
    case 2:
      // Do nothing, this object deallocates its own host memory
#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep("cache delete obj, no need to free host mem");
#endif
      break;
    default:
      QDP_error_exit("cache delete objects: unkown host memory allocator");
      break;
    }

  }

  void QDPCache::allocateHostMemory(Entry& e) {
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("cache: alloc host size=%lu",(unsigned long)e.size);
#endif
      
#ifdef SANITY_CHECKS_CACHE
    // SANITY
    if(e.hstPtr) 
      QDP_error_exit("cache allocateHostMemory: already allocated");
#endif

    switch(e.flags) {
    case 0:

      if ( !CUDAHostPoolAllocator::Instance().allocate( &e.hstPtr , e.size ) )
	QDP_error_exit("cache allocateHostMemory: host memory allocator flags=0 failed");

      break;
    case 1:

      try {
	e.hstPtr = (void*)QDP::Allocator::theQDPAllocator::Instance().allocate( e.size , QDP::Allocator::DEFAULT );
      }
      catch(std::bad_alloc) {
	QDP_error_exit("cache allocateHostMemory: host memory allocator flags=1 failed");
      }
      break;
    case 2:
      QDP_error_exit("cache allocateHostMemory: we should not be here");
      break;
    default:

      QDP_error_exit("cache allocateHostMemory: unkown host memory allocator");

      break;
    }

#ifdef SANITY_CHECKS_CACHE
    // SANITY
    if(!e.hstPtr)
      QDP_error_exit("cache allocateHostMemory: not allocated, but should");
#endif

  }


  void QDPCache::assureDevice(Entry& e) {

    if (!e.devPtr) {
      while (!CUDADevicePoolAllocator::Instance().allocate( &e.devPtr , e.size )) {
	if (!spill_lru()) {
	  QDP_info("Device pool:");
	  CUDADevicePoolAllocator::Instance().printListPool();
	  QDP_info("Host pool:");
	  CUDAHostPoolAllocator::Instance().printListPool();
	  printLockSets();
	  QDP_error_exit("cache assureDevice: can't spill LRU object. Out of GPU memory!");
	}
      }
      if (e.hstPtr) {
	CudaMemcpyAsync( e.devPtr , e.hstPtr , e.size );
	CudaSyncTransferStream();
	if (e.flags != 2)
	  freeHostMemory(e);
      }
    }

#ifdef SANITY_CHECKS_CACHE
    if (e.hstPtr) QDP_error_exit("assureDevice: We still have a host pointer");
#endif

    vecLockSet[currLS].push_back(e.Id);
    e.lockCount++;
  }


  bool QDPCache::assureHost(Entry& e) {
      
#ifdef SANITY_CHECKS_CACHE
    // SANITY
    if (find(lstDel.begin(),lstDel.end(),e.Id) != lstDel.end())
      QDP_error_exit("cache assureHost: e in lstDel");
    if (e.flags == 2)
      QDP_error_exit("cache assureHost: flags == 2");
#endif

    if (e.lockCount > 0) {
#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep("cache assure on host. obj in current calculation. will sync with kernel stream");
#endif
      CudaSyncKernelStream();
      releasePrevLockSet();
      beginNewLockSet();
#ifdef SANITY_CHECKS_CACHE
      if (e.lockCount > 0)
	QDP_error_exit("cache assureHost: e still locked!");
#endif	
    }

    // When it's an object which manages its own host memory
    // we can immediately free the device memory
    if (e.flags == 2) {
      if (e.devPtr) {
	CUDADevicePoolAllocator::Instance().free( e.devPtr );
	e.devPtr = NULL;
      }
    } else {
      if (!e.hstPtr) {
	allocateHostMemory(e);
	if (e.devPtr) {
	  CudaMemcpyAsync( e.hstPtr , e.devPtr , e.size );
	  CudaSyncTransferStream();
	  CUDADevicePoolAllocator::Instance().free( e.devPtr );
	  e.devPtr = NULL;
	}
      }
    }

    bool in_flight=false;
    return in_flight;
  }


  bool QDPCache::spill_lru() {
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("cache: spill lru");
#endif

    if (lstTracker.size() < 1)
      return false;

    list<int>::iterator it_key = lstTracker.begin();
      
    bool found=false;
    Entry* e;

    while ( !found  &&  it_key != lstTracker.end() ) {
      e = &vecEntry[ *it_key ];

      found = ( (e->lockCount == 0) && (e->devPtr != NULL) );
      if (!found)
	it_key++;
    }


    if (found) {
#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep("cache: spill_lru: not locked obj found, will spill now. size=%u devPtr=%p *it_key=%d id=%d",e->size,e->devPtr,*it_key,e->Id);
#endif
      assureHost( *e );
      return true;
    } else {
#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep("cache: spill_lru:  Its not possible to spill an object (all locked).");
#endif
      return false;
    }
      

  }




  void QDPCache::printTracker() {
#if 0
    QDP_debug_deep("Tracker: ---");
    int n=0;
    for ( ListKey::iterator i = lstTracker.begin() ; i != lstTracker.end() ; i++ ) {
      MapKeyValue::iterator it = mapReg.find( *i );
      Value e = (*it).second.first;
      QDP_debug_deep("%d: id=%u size=%u ",n++,(unsigned)*i,(unsigned)e->size );
    }
    QDP_debug_deep("------------");
#endif
  }


  void QDPCache::deleteObjects() {

    list<int>::iterator i = lstDel.begin();

    while ( i != lstDel.end() ) {

      Entry& e = vecEntry[ *i ];

#ifdef SANITY_CHECKS_CACHE
      // SANITY
      if (e.lockCount < 0)
	QDP_error_exit("cache deleteObj: lockCount < 0");
#endif

#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep("del objs: size=%u  lockcount=%u devPtr=%p hstPtr=%p flags=%d",(unsigned)e.size,(unsigned)e.lockCount,e.devPtr,e.hstPtr,e.flags);
#endif

      if (e.lockCount == 0) {

#ifdef SANITY_CHECKS_CACHE
	// SANITY
	if (find(vecLockSet[0].begin(),vecLockSet[0].end(),*i) != vecLockSet[0].end())
	  QDP_error_exit("cache deleteObj: obj in lock set 0");
	if (find(vecLockSet[1].begin(),vecLockSet[1].end(),*i) != vecLockSet[1].end())
	  QDP_error_exit("cache deleteObj: obj in lock set 1");
#endif

#ifdef GPU_DEBUG_DEEP
	QDP_debug_deep("cache delete obj size=%u",(unsigned)e.size);
#endif
	  
	if (e.devPtr)
	  CUDADevicePoolAllocator::Instance().free( e.devPtr );

	if (e.hstPtr)
	  freeHostMemory(e);

	stackFree.push( *i );
	lstDel.erase( i++ );

      } else {
	i++;
      }
    }
  }



  QDPCache::QDPCache() : currLS(0) , prevLS(-1), vecEntry(1024) {
#ifdef GPU_DEBUG_DEEP
    QDP_info_primary("Constructing cache ..");
    QDP_info_primary("cache: pushing %u elements into stack",(unsigned)vecEntry.size());
#endif
    for ( int i = vecEntry.size()-1 ; i >= 0 ; --i ) {
      stackFree.push(i);
    }
    vecLockSet[0].reserve(1024);
    vecLockSet[1].reserve(1024);
  }



  QDPCache::~QDPCache() {
  }





}


