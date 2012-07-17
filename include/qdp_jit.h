// -*- C++ -*-

/*! @file
 * @brief 
 *
 * 
 */

#ifndef QDP_JIT_H
#define QDP_JIT_H

#include<map>
#include "qdp_iface.h"
#include <dlfcn.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <string.h>
#include <cmath>

#define JIT_VERSION 2

namespace QDP {


  void xmlready(string& xml);


  class QDPJit
  {
  public:
    static QDPJit& Instance() {
      static QDPJit singleton;
      return singleton;
    }



    typedef int (*Kernel)(bool,UnionDevPtr*,kernel_geom_t *,void *); // sync args geom
    typedef char* Pretty;
    typedef int Version;
    struct SharedLibEntry {
      Kernel             kernel;
      Version            version;
      Pretty             pretty;
      void *             handle;
    };
    struct benchresult_t {
      int threadsPerBlock;
    };
    typedef map<string,SharedLibEntry> MapFunction;
    typedef list<void *> ListHandle;

    bool buildObject( const std::string& fname_dest , 
		      const std::string& fname_src , 
		      const std::string& path_qdp , 
		      const std::string& jit_options );

    void setKernelPath(const string& _path);
    void setQDPPath(const string& _path);
    bool operator()( const string& strId , const string& kernel_calc , 
		     UnionDevPtr* args ,
		     int numSites,
		     SharedLibEntry & sharedLibEntry,
		     MapVolumes*& mapVolumes );
    bool jitFixedGeom( string& strId , string& kernel_calc , 
		       UnionDevPtr* args ,
		       int numSites,
		       SharedLibEntry & sharedLibEntry,
		       int threads,int blocks ,int smemSize );
    void closeId(string& strId);
    void closeAllShared();
    void rmAllBuiltShared();
    void loadAllShared();

    string getJitOptions();
    
    void addJitOption(string& opt);

  private:

    std::string program(const std::string& strId,
			const std::string& kernel_calc);
    bool hasEnding (std::string const &fullString, std::string const &ending);
    int getdir(string dir, list<string> &files);
    SharedLibEntry loadShared(string filename);
    string buildFunction(const string & prefix,string pretty);
    kernel_geom_t getGeom(int numSites , int threadsPerBlock);

    bool benchmark(benchresult_t & benchresult,
		   const SharedLibEntry & entry , 
		   UnionDevPtr* args,
		   int numSites);

    MapFunction mapFunction;
    ListHandle listHandle;
    string kernelPath;
    string qdpPath;
    list<string> listBuiltShared;
    list<string> listJitOption;
  };

  //extern QDPJit<> theQDPJit;

}




namespace QDP 
{



}




#endif
