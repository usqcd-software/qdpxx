#ifndef QDP_CACHE
#define QDP_CACHE

#include <iostream>
#include <map>
#include <vector>
#include <stack>
#include "string.h"
#include "math.h"

//#define SANITY_CHECKS_CACHE

using namespace std;

namespace QDP 
{
  class QDPJitArgs;


  class QDPCache
  {
    struct Entry;
  public:
    static QDPCache& Instance();

    void beginNewLockSet();
    void releasePrevLockSet();
    void printLockSets();
    bool allocate_device_static( void** ptr, size_t n_bytes );
    void free_device_static( void* ptr );
    void backupOnHost();
    void restoreFromHost();
    void sayHi();
    bool onDevice(int id) const;    
    void enlargeStack();
    int registrate( size_t size, unsigned flags);
    int registrateOwnHostMem( size_t size, void* ptr);
    void signoff(int id);
    void * getDevicePtr(int id);
    bool getHostPtr(void ** ptr , int id);
    void freeHostMemory(Entry& e);
    void allocateHostMemory(Entry& e);
    void assureDevice(Entry& e);
    bool assureHost(Entry& e);
    bool spill_lru();
    void printTracker();
    void deleteObjects();
    QDPCache();
    ~QDPCache();

  private:
    list<void *>        lstStatic;

    vector<Entry>       vecEntry;
    stack<int>          stackFree;

    list<int>           lstTracker;

    list<int>           lstDel;
    vector<int>         vecLockSet[2];   // with duplicate entries
    int                 currLS;
    int                 prevLS;
    list<char*>         listBackup;

  };



}


#endif

