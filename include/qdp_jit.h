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

#ifdef QDP_USE_SOCKET
#include<tcpsocket.hh>
#endif


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

    QDPJit(): compileServerIPv6(false), haveCompileServer(false), compileServerPort(1500) {}

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
    bool buildObject_local( const std::string& fname_dest , 
			    const std::string& fname_src , 
			    const std::string& path_qdp , 
			    const std::string& jit_options );
    bool buildObject_cs( const std::string& fname_dest , 
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

    void setCompilerServerName(const char * name) {
      QDP_info_primary("Setting compile server name to %s",name);
      compileServerName = string(name);
      haveCompileServer=true;
    }
    void setCompilerServerPort(int port) {
      QDP_info_primary("Setting compile server port number to %d",port);
      compileServerPort = port;
    }
    void setCompilerServerIPv6(bool ipv6) {
      QDP_info_primary("Setting compile server AF %s",ipv6?"IPv6":"IPv4");
      compileServerIPv6 = ipv6;
    }

  private:
#ifdef QDP_USE_SOCKET
    void sendString(Network::TcpSocket& sock, const string& str);
    void recvAck(Network::TcpSocket& cClientSocket);
#endif

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

    bool haveCompileServer;
    string compileServerName;
    int compileServerPort;
    bool compileServerIPv6;
  };

  //extern QDPJit<> theQDPJit;

}




namespace QDP 
{



}




#endif
