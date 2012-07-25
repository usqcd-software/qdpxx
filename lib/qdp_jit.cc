#include "qdp.h"


namespace QDP {

  const char * QDPuni[5]={".ptr",".Int",".Bool",".IntPtr",".Size_t"};


  void QDPJit::setKernelPath(const string& _path) { 
    kernelPath = _path; 
  }

  void QDPJit::setQDPPath(const string& _path) { 
    qdpPath = _path; 
  }

  string QDPJit::getJitOptions() {
    ostringstream oss;
    for (list<string>::iterator i = listJitOption.begin() ; i != listJitOption.end() ; ++i )
      oss << *i << " ";
    return oss.str();
  }
    
  void QDPJit::addJitOption(string& opt) {
    QDP_info_primary("Adding option to JIT compiler: %s",opt.c_str());
    listJitOption.push_back(opt);
  }


    
  kernel_geom_t QDPJit::getGeom(int numSites , int threadsPerBlock) 
  {
    kernel_geom_t geom_host;

    geom_host.threads_per_block = threadsPerBlock;
    geom_host.Nblock_x = min( DeviceParams::Instance().getMaxGridX() , (int)std::ceil( (double)numSites / (double)geom_host.threads_per_block ) );
    geom_host.Nblock_y = (int)std::ceil(  (double)numSites / (double)(geom_host.Nblock_x * threadsPerBlock) );
    geom_host.smemSize = DeviceParams::Instance().getDefaultSMem();

    return geom_host;
  }


  
  bool QDPJit::benchmark(benchresult_t & benchresult,
			 const SharedLibEntry & entry,
			 UnionDevPtr* args,
			 int numSites)
  {
    kernel_geom_t geom_host;

    double bestTime;
    bool first=true;
    bool ret=false;
    benchresult.threadsPerBlock=-1;

    for ( int i = 1 ; i <= DeviceParams::Instance().getMaxBlockX() ; i*=2 ) {
      if ( i <= numSites ) {

	geom_host = getGeom(numSites , i);

#ifdef GPU_DEBUG_DEEP
	QDP_debug_deep("Grid_x_y = %d %d",geom_host.Nblock_x,geom_host.Nblock_y);
#endif

	StopWatch watch0;
	watch0.start();

	int result;
	//
	// Here we need sync (1st argument)
	// since kernel call is non-blocking
	result = entry.kernel(true,args,&geom_host,CudaGetKernelStream());

	watch0.stop();
      
	QDP_info("benchmark numThreads=%d result=%d time=%f" , i , result , watch0.getTimeInMicroseconds() );
      
	if ((result==0) && ((watch0.getTimeInMicroseconds() < bestTime) || first)) {
	  first=false;
	  bestTime=watch0.getTimeInMicroseconds();
	  benchresult.threadsPerBlock=i;
	  ret=true;
	}
      }
    }
    if (ret)
      QDP_info_primary("best numThreads=%d time=%f",benchresult.threadsPerBlock,bestTime);
    else
      QDP_info_primary("No bechmarking kernel call succeeded");

    return ret;
  }


  
  bool QDPJit::operator()( const string& strId , const string& kernel_calc , 
			   UnionDevPtr* args ,
			   int numSites ,
			   QDPJit::SharedLibEntry & sharedLibEntry ,
			   MapVolumes*& mapVolumes )
  {
#if 1
    CudaSyncTransferStream();
    CudaSyncKernelStream();
#else
    // not working
    //CudaSyncKernelStream();
    //CudaRecordAndWaitEvent();

    // not working
    CudaSyncKernelStream();
    CudaRecordAndWaitEvent();
#endif

    MapFunction::const_iterator call;

#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("JIT::op() kernel = %p",(void*)(sharedLibEntry.kernel) );
#endif

    //
    // Is the kernel already in place ?
    //
    if (!sharedLibEntry.kernel) {

      //
      // Do we have the kernel in the loaded shared libs ?
      //
      call = mapFunction.find(strId);
      if (call == mapFunction.end()) {
	std::string filename;

	//
	// Only primary node builds kernel
	//
	if (Layout::primaryNode()) {
	  QDP_info("building kernel for benchmarking...");
	  filename = buildFunction("",program(strId,kernel_calc));
	}

	QDPInternal::broadcast_str(filename);

	if (!Layout::primaryNode()) {
	  QDP_debug_deep("Received filename for compute kernel = %s" , filename.c_str());
	}

	// Do we need a filesystem sync here ?
	sharedLibEntry = loadShared(filename);
 
	// Sanity check
	for( MapFunction::iterator iter = mapFunction.begin() ; iter != mapFunction.end() ; ++iter ) 
	  if (iter->second.kernel == sharedLibEntry.kernel)
	    QDP_error_exit("Same memory address found for another cudp function (2). Giving up!\n");

	mapFunction.insert( MapFunction::value_type(strId,sharedLibEntry));

	call = mapFunction.find(strId);
	if (call == mapFunction.end()) {
	  QDP_error_exit("qdp_jit.h serious problems here");
	}
      } else {
	sharedLibEntry = (*call).second;
      }

    }

    mapVolumes = &JitTuning::Instance().getMapTuning()[strId];

    if (numSites > 0) {
      //
      // Do we have benchmark results for this kernel ?
      //
      if (mapVolumes->count(numSites) == 0) {
	//
	// Every node does benchmarking
	//
	benchresult_t benchresult;
	QDPCache::Instance().backupOnHost();
	if (!benchmark(benchresult,sharedLibEntry,args,numSites ))
	  QDP_error_exit("JIT: Benchmarking a kernel was not possible");
	QDPCache::Instance().restoreFromHost();

	QDP_info("Tuning DB: Inserting [volume=%d][threads=%d]",numSites,benchresult.threadsPerBlock);
	(*mapVolumes)[numSites] = benchresult.threadsPerBlock;
      }

      int result;
      kernel_geom_t kernel_geom;

#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep("mapVolumes[numSites]= = %d",(*mapVolumes)[numSites]);
#endif

      kernel_geom = getGeom( numSites , (*mapVolumes)[numSites] );
      while (result = sharedLibEntry.kernel( DeviceParams::Instance().getSyncDevice() , 
					     args , &kernel_geom, 
					     CudaGetKernelStream() ) ) {

	if (result != 2)
	  QDP_error_exit("Kernel launch problem, not out-of-memory %d: Giving up!\n",result);

	QDP_error_exit("kernel call: out of memory. ");
      }
    }

    //
    // Release the current lock set and start a new one
    //
    QDPCache::Instance().releasePrevLockSet();
    QDPCache::Instance().beginNewLockSet();

    return true;
  }



  
  bool QDPJit::jitFixedGeom( string& strId , string& kernel_calc , 
			     UnionDevPtr* args ,
			     int numSites,
			     QDPJit::SharedLibEntry & sharedLibEntry,
			     int threads, int blocks,int smemSize )
  {
    CudaSyncKernelStream();
    CudaRecordAndWaitEvent();

    MapFunction::const_iterator call;

#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("JIT:: fixedGeom() kernel = %p",(void*)(sharedLibEntry.kernel) );
#endif

    if (!sharedLibEntry.kernel) {
      call = mapFunction.find(strId);
      SharedLibEntry entry;
      if (call == mapFunction.end()) {

	std::string filename;

	//
	// Only primary node builds kernel
	//
	if (Layout::primaryNode()) {
	  QDP_info("building kernel with fixed thread geometry information ...");
	  filename = buildFunction("",program(strId,kernel_calc));
	}

	QDPInternal::broadcast_str(filename);

	if (!Layout::primaryNode()) {
	  QDP_debug_deep("Received filename for fixedGeom compute kernel = %s" , filename.c_str());
	}

	// Do we need a filesystem sync here ?
	entry = loadShared(filename);
 
	// Sanity check
	for( MapFunction::iterator iter = mapFunction.begin() ; iter != mapFunction.end() ; ++iter ) 
	  if (iter->second.kernel == entry.kernel)
	    QDP_error_exit("Same memory address found for another cudp function. Giving up!\n");
	mapFunction.insert( MapFunction::value_type(strId,entry));

	call = mapFunction.find(strId);
	if (call == mapFunction.end()) {
	  QDP_error_exit("qdp_jit.h serious problems here");
	}
      } 
      // 
      //sharedLibEntry = entry;
      sharedLibEntry = (*call).second;
      //kernel = (*call).second.kernel;
    }

    int result;
    kernel_geom_t kernel_geom;
    kernel_geom.threads_per_block=threads;
    kernel_geom.Nblock_x=blocks;
    kernel_geom.Nblock_y=1;
    kernel_geom.smemSize=smemSize;

    while (result = sharedLibEntry.kernel( DeviceParams::Instance().getSyncDevice() , 
					   args , &kernel_geom , 
					   CudaGetKernelStream() ) ) {
      if (result != 2)
	QDP_error_exit("Kernel launch problem, not out-of-memory %d: Giving up!\n",result);

      QDP_error_exit("kernel call: out of memory. " );

    }

    QDPCache::Instance().releasePrevLockSet();
    QDPCache::Instance().beginNewLockSet();

    return true;
  }



  bool QDPJit::buildObject( const std::string& fname_dest , 
			    const std::string& fname_src , 
			    const std::string& path_qdp , 
			    const std::string& jit_options )
  {
    std::ostringstream jit_command;

    jit_command << string(CUDA_DIR) << "/bin/nvcc " << jit_options;
    jit_command << " -arch=" << string(QDP_GPUARCH);
    jit_command << " -m64 --compiler-options -fPIC,-shared -link ";
    jit_command << fname_src << " -I" << path_qdp << "/include -o " << fname_dest;
    
    string gen = jit_command.str();
    QDP_info("%s",gen.c_str());

    char* cmd = new(nothrow) char[gen.size()+1];
    if (!cmd) QDP_error_exit("no memory for jitter command");
    strcpy(cmd,gen.c_str());

    int ret;

    if (ret=system(cmd)) {

      cout << "failed cmd = " << (void*)cmd << endl;

      QDP_info("Problems calling nvcc jitter:");

      if (WIFEXITED(ret)) {
	printf("exited, status=%d\n", WEXITSTATUS(ret));
      } else if (WIFSIGNALED(ret)) {
	printf("killed by signal %d\n", WTERMSIG(ret));
      } else if (WIFSTOPPED(ret)) {
	printf("stopped by signal %d\n", WSTOPSIG(ret));
      } else if (WIFCONTINUED(ret)) {
	printf("continued\n");
      } else {
	printf("not recognized\n");
      }

      cout << "Checking shell.. ";
      if(system(NULL))
	cout << "ok!\n";
      else
	cout << "nope!\n";

      if (WEXITSTATUS(ret) != 127)
	QDP_error_exit("This error is not the hw/kernel bug that we have a workaround. giving up");

      QDP_info("This error is the hw/kernel bug");

      list<char*> listPtr;
      int maxtry=1000;
      do{
	cout << maxtry << " failed cmd = " << (void*)cmd << endl;

	cout << "Problems calling nvcc jitter: ";

	if (WIFEXITED(ret)) {
	  printf("exited, status=%d\n", WEXITSTATUS(ret));
	} else if (WIFSIGNALED(ret)) {
	  printf("killed by signal %d\n", WTERMSIG(ret));
	} else if (WIFSTOPPED(ret)) {
	  printf("stopped by signal %d\n", WSTOPSIG(ret));
	} else if (WIFCONTINUED(ret)) {
	  printf("continued\n");
	} else {
	  printf("not recognized\n");
	}

	cout << "alloc " << gen.size()+1 << endl;
	char* tmp = new(nothrow) char[gen.size()+1];
	if (!tmp) QDP_error_exit("no memory for jitter command");
	strcpy(tmp,gen.c_str());
	listPtr.push_back( tmp );

	printf("%s\n",listPtr.back());
	  
      } while ((ret=system(listPtr.back())) && (--maxtry>0));

      int ii=0;
      while(listPtr.size()) {
	cout << "deleting " << ii++ << endl;
	delete[] listPtr.back();
	listPtr.pop_back();
      }

      if (ret)
	QDP_error_exit("Nvcc error\n");
      else
	cout << "system call kernel/hardware bug: workaround applied !" << endl;

    }
    delete[] cmd;

    return true;
  }


  
  std::string QDPJit::buildFunction(const string & prefix,string prg)
  {
    string tmp = kernelPath + "/" + prefix + "cudp_XXXXXX";

    char * temp = new char[tmp.size() + 1];
    std::copy(tmp.begin(), tmp.end(), temp);
    temp[tmp.size()] = '\0';

    mktemp(temp);
    if (!temp) {
      QDP_error_exit("error while creating temporary file /tmp/cudp_...\n");
    }

    string basename = string(temp);

    QDP_info( "building cudp function using temporary file: %s" , basename.c_str());
    string file_cu = basename + ".cu";
    string file_o  = basename + ".o";

    ofstream tmpfile;
    tmpfile.open( file_cu.c_str() );
    if (tmpfile.good()) {
      tmpfile << prg;
    }
    tmpfile.close();

    if (!buildObject( file_o , file_cu , qdpPath , getJitOptions() ))
       QDP_error_exit("QDPJit: buildObject error");

    if ((prefix.size()>0) || (!Layout::primaryNode())) {
      listBuiltShared.push_back( file_cu );
      listBuiltShared.push_back( file_o );
    }

    return file_o;
  }


  
   QDPJit::SharedLibEntry QDPJit::loadShared(std::string filename)
  {
    size_t total,free1;

    QDPJit::SharedLibEntry entry;

    void *handle;
    handle = dlopen( filename.c_str() ,  RTLD_LAZY);

    if (!handle) {
      QDP_error_exit("dlopen error: %s\n",dlerror());
    } else {
      QDP_info_primary("LSB shared object loaded successfully");
    }

    listHandle.push_back(handle);

    entry.handle = handle;

    {
      int (*fptr)(bool,UnionDevPtr*,kernel_geom_t *,void *);
      char *err;
      dlerror(); /* clear error code */
      *(void **)(&fptr) = dlsym(handle, "function_host");
      if ((err = dlerror()) != NULL) {
	QDP_info("dlerror = %s",err);
	QDP_error_exit("dlsym error, function");
      }
      entry.kernel = fptr;
    }

    {
      char *fptr;
      char *err;
      dlerror(); /* clear error code */
      *(void **)(&fptr) = dlsym(handle, "pretty");
      if ((err = dlerror()) != NULL) {
	QDP_info("dlerror = %s",err);
	QDP_error_exit("dlsym error, pretty");
      }
      entry.pretty = fptr;
    }

    {
      void * ptr;
      char *err;
      dlerror(); /* clear error code */
      ptr = dlsym(handle, "version");
      if ((err = dlerror()) != NULL) {
	QDP_info("dlerror = %s",err);
	QDP_error_exit("dlsym error, version");
      }
      entry.version = *(int*)ptr;
      if (entry.version != (int)JIT_VERSION)
	QDP_error_exit("Kernel found has version number %d, this QDP-JIT requires kernels with version number %d",entry.version,(int)JIT_VERSION);
    }

    QDP_debug_deep("symbol found");

    return entry;
  }

  
  void QDPJit::closeAllShared()
  {
    QDP_info_primary("QDPJit: closing %d opened shared objects." , listHandle.size() );
    for(ListHandle::iterator iter = listHandle.begin() ; iter != listHandle.end() ; ++iter ) 
      {
	dlclose(*iter);
      }
  }


  
  void QDPJit::rmAllBuiltShared()
  {
    QDP_info_primary( "QDPJit: removing %d built shared objects." , listBuiltShared.size() );
    for(list<string>::iterator iter = listBuiltShared.begin() ; iter != listBuiltShared.end() ; ++iter ) 
      {
	cout << "removing " << *iter << endl;
	std::remove((*iter).c_str());
      }
  }


  
  void QDPJit::closeId(string& strId)
  {
    MapFunction::const_iterator call;

    call = mapFunction.find(strId);
    if (call == mapFunction.end()) {
      QDP_error_exit("QDPJit::closeId: error");
    } else {
      QDP_info("dlclose Id = %s" , strId.c_str());
      dlclose( (*call).second.handle );
      mapFunction.erase(strId);
    }
  }

  // 
  // void QDPJit::closeEntry(SharedLibEntry& entry)
  // {
    
  //   cout << "cudp: closing " << listHandle.size() << " opened shared objects." << endl;
  //   for(ListHandle::iterator iter = listHandle.begin() ; iter != listHandle.end() ; ++iter ) 
  //     {
  // 	dlclose(*iter);
  //     }
  // }


  
  bool QDPJit::hasEnding (std::string const &fullString, std::string const &ending)
  {
    if (fullString.length() > ending.length()) {
      return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
      return false;
    }
  }

  
  int QDPJit::getdir(string dir, list<string> &files)
  {
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
      QDP_info("Error(%d) opening %s", errno, dir.c_str());
      return errno;
    }

    QDP_info_primary("JIT: Kernels found:\n");
    while ((dirp = readdir(dp)) != NULL) {
      if ((hasEnding(string(dirp->d_name),".o")) &&
	  (!strncmp(basename(string(dirp->d_name).c_str()),"cudp_",5) )) {
	string tmp = dir + "/" + string(dirp->d_name);
	QDP_info_primary("%s",tmp.c_str());
	files.push_back(dir+"/"+string(dirp->d_name));
      }
    }
    closedir(dp);
    return 0;
  }


  
  void QDPJit::loadAllShared()
  {
    list<std::string> filenames;
    if (!getdir( kernelPath , filenames )) {

      for(list<std::string>::iterator iter = filenames.begin() ; iter != filenames.end() ; ++iter ) {
	QDP_info_primary("loading %s",(*iter).c_str());
	SharedLibEntry entry = loadShared( *iter );
	mapFunction.insert( MapFunction::value_type(string(entry.pretty),entry));
      }
    }
  }


  
  std::string QDPJit::program( const std::string& strId , const std::string& kernel_calc )
  {
    std::ostringstream sprg;

    sprg << "#include \"qdp_device.h\"" << endl;
    sprg << "#include \"qdp_iface.h\"" << endl;
    sprg << "#include <iostream>" << endl;
    sprg << "using namespace QDP;" << endl;
    sprg << "using namespace std;" << endl;
    sprg << "extern \"C\" char pretty[]=\"" << strId << "\";" << endl;
    sprg << "extern \"C\" int version=" << (int)JIT_VERSION << ";" << endl;
    sprg << "__global__ void kernel(UnionDevPtr* args)" << endl;
    sprg << "{" << endl;
    sprg << kernel_calc << endl;
    sprg << "}"<< endl;
    sprg << "extern \"C\" int function_host(";
    sprg << "bool sync,";
    sprg << "UnionDevPtr* kernel_args,";
    sprg << "kernel_geom_t * kernel_geom,";
    sprg << "void * ptr_cudaStream";
    sprg << ")"<< endl;
    sprg << "{"<< endl;
#ifdef GPU_DEBUG_DEEP
    sprg << "    cout << \"function_host()\" << endl;"<< endl;
#endif
    sprg << endl;
    sprg << "    dim3  blocksPerGrid( kernel_geom->Nblock_x , kernel_geom->Nblock_y , 1 );"<< endl;
    sprg << "    dim3  threadsPerBlock( kernel_geom->threads_per_block , 1, 1);"<< endl;
    sprg << endl;
    sprg << "    if ( kernel_geom->Nblock_x != 0 && kernel_geom->Nblock_y != 0 && kernel_geom->threads_per_block !=0 )" << endl;
    sprg << "      kernel<<< blocksPerGrid , threadsPerBlock , kernel_geom->smemSize , *((cudaStream_t*)ptr_cudaStream) >>>( kernel_args );" << endl;
    sprg << "    if (sync)" << endl;
    sprg << "      cudaDeviceSynchronize();" << endl;
    sprg << "    cudaError_t kernel_call = cudaGetLastError();" << endl;
#ifdef GPU_DEBUG_DEEP
    sprg << "    cout << \"kernel launched with \" << kernel_geom->Nblock_x << \" \" << kernel_geom->Nblock_y << \" blocks and \" << kernel_geom->threads_per_block << \" threads each...\" << endl;" << endl;
    sprg << "    cout << \"kernel call:         \" << string(cudaGetErrorString(kernel_call)) << endl;" << endl;
#endif
    sprg << "    if (kernel_call != cudaSuccess) {\n";
    sprg << "      cout << \"kernel call:         \" << string(cudaGetErrorString(kernel_call)) << \" code = \" << (int)(kernel_call) << \" \" << cudaErrorMemoryAllocation << endl;\n";
    sprg << "      return((int)(kernel_call));\n";
    sprg << "    }\n";
    sprg << "    return(0);\n";
    sprg << "}\n";

    return sprg.str();
  }




  void xmlready(string& xml) {
    std::map<char, std::string> transformations;
    transformations['&']  = std::string("");
    transformations['\''] = std::string("");
    transformations['"']  = std::string("");
    transformations['>']  = std::string("");
    transformations['<']  = std::string("");

    std::string reserved_chars;
    for (std::map<char, std::string>::iterator ti = transformations.begin(); ti != transformations.end(); ti++)
      {
        reserved_chars += ti->first;
      }

    size_t pos = 0;
    while (std::string::npos != (pos = xml.find_first_of(reserved_chars, pos)))
      {
        xml.replace(pos, 1, transformations[xml[pos]]);
        pos++;
      }
  }


}
