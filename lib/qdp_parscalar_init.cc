
/*! @file
 * @brief Parscalar init routines
 * 
 * Initialization routines for parscalar implementation
 */

#include <stdlib.h>
#include <unistd.h>

#include "qdp.h"
#include "qmp.h"

#if defined(QDP_USE_QMT_THREADS)
#include <qmt.h>
#endif


namespace QDP {
	
	namespace ThreadReductions {
		REAL64* norm2_results;
		REAL64* innerProd_results;
	}

  //! Private flag for status
  static bool isInit = false;

#ifdef QDP_IS_QDPJIT	
  //! Public flag for using the GPU or not
  bool QDPuseGPU = false;

  void QDP_startGPU()
  {
    QDP_info("Start using the GPU");
    QDPuseGPU=true;
    
    CudaCreateStreams();
    CUDAHostPoolAllocator::Instance().registerMemory();
  }


  //! Set the GPU device
  void QDP_setGPU()
  {
    int deviceCount;
    CudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
      cout << "no CUDA devices found" << endl;
      exit(1);
    }
    
    int rank_QMP = QMP_get_node_number();
    int dev      = rank_QMP % deviceCount;
    
    cout << "QDP-JIT: Setting active Cuda device to " << dev << endl;
    CudaSetDevice( dev );
  }
#endif

	
	//! Turn on the machine
	void QDP_initialize(int *argc, char ***argv)
	{
		if (isInit)
		{
			QDPIO::cerr << "QDP already inited" << endl;
			QDP_abort(1);
		}

#ifdef QDP_IS_QDPJIT
		bool paramCC = false;
#endif
		
		//
		// Process command line
		//
		
		// Look for help
		bool help_flag = false;
		for (int i=0; i<*argc; i++) 
		{
			if (strcmp((*argv)[i], "-h")==0)
				help_flag = true;
		}
		
		bool setGeomP = false;
		multi1d<int> logical_geom(Nd);   // apriori logical geometry of the machine
		logical_geom = 0;
		
		bool setIOGeomP = false;
		multi1d<int> logical_iogeom(Nd); // apriori logical 	
		logical_iogeom = 0;
		
#ifdef USE_REMOTE_QIO
		int rtiP = 0;
#endif
		int QMP_verboseP = 0;
		const int maxlen = 256;
		char rtinode[maxlen];
		strncpy(rtinode, "your_local_food_store", maxlen);
		
		// Usage
		if (Layout::primaryNode())  {
			if (help_flag) 
			{
				fprintf(stderr,"Usage:    %s options\n",(*argv)[0]);
				fprintf(stderr,"options:\n");
				fprintf(stderr,"    -h        help\n");
				fprintf(stderr,"    -V        %%d [%d] verbose mode for QMP\n", 
						QMP_verboseP);
#if defined(QDP_USE_PROFILING)   
				fprintf(stderr,"    -p        %%d [%d] profile level\n", 
						getProfileLevel());
#endif
				
				// logical geometry info
				fprintf(stderr,"    -geom     %%d");
				for(int i=1; i < Nd; i++) 
					fprintf(stderr," %%d");
				
				fprintf(stderr," [-1");
				for(int i=1; i < Nd; i++) 
					fprintf(stderr,",-1");
				fprintf(stderr,"] logical machine geometry\n");
				
#ifdef USE_REMOTE_QIO
				fprintf(stderr,"    -cd       %%s [.] set working dir for QIO interface\n");
				fprintf(stderr,"    -rti      %%d [%d] use run-time interface\n", 
						rtiP);
				fprintf(stderr,"    -rtinode  %%s [%s] run-time interface fileserver node\n", 
						rtinode);
#endif
				
				QDP_abort(1);
			}
		}

		for (int i=1; i<*argc; i++) 
		{
			if (strcmp((*argv)[i], "-V")==0) 
			{
				QMP_verboseP = 1;
			}
#if defined(QDP_USE_PROFILING)   
			else if (strcmp((*argv)[i], "-p")==0) 
			{
				int lev;
				sscanf((*argv)[++i], "%d", &lev);
				setProgramProfileLevel(lev);
			}
#endif
#ifdef QDP_IS_QDPJIT
			else if (strcmp((*argv)[i], "-sm")==0) 
			  {
			    int uu;
			    sscanf((*argv)[++i], "%d", &uu);
			    paramCC=true;
			    DeviceParams::Instance().setCC(uu);
			  }
			else if (strcmp((*argv)[i], "-sync")==0) 
			  {
			    DeviceParams::Instance().setSyncDevice(true);
			  }
			else if (strcmp((*argv)[i], "-jitopt")==0) 
			  {
			    char buf[1024*2];
			    sscanf((*argv)[++i], "%s", buf);
			    string opt(buf);
			    QDPJit::Instance().addJitOption( opt );
			  }
			else if (strcmp((*argv)[i], "-poolsize")==0) 
			  {
			    float f;
			    char c;
			    sscanf((*argv)[++i],"%f%c",&f,&c);
			    double mul;
			    switch (tolower(c)) {
			    case 'k': 
			      mul=1024.; 
			      break;
			    case 'm': 
			      mul=1024.*1024; 
			      break;
			    case 'g': 
			      mul=1024.*1024*1024; 
			      break;
			    case 't':
			      mul=1024.*1024*1024*1024;
			      break;
			    case '\0':
			      break;
			    default:
			      QDP_error_exit("unknown multiplication factor");
			    }
			    size_t val = (size_t)((double)(f) * mul);
			    CUDADevicePoolAllocator::Instance().setPoolSize(val);
			  }
#endif
			else if (strcmp((*argv)[i], "-geom")==0) 
			{
				setGeomP = true;
				for(int j=0; j < Nd; j++) 
				{
					int uu;
					sscanf((*argv)[++i], "%d", &uu);
					logical_geom[j] = uu;
				}
			}
			else if (strcmp((*argv)[i], "-iogeom")==0) 
			{
				setIOGeomP = true;
				for(int j=0; j < Nd; j++) 
				{
					int uu;
					sscanf((*argv)[++i], "%d", &uu);
					logical_iogeom[j] = uu;
				}
			}
#ifdef USE_REMOTE_QIO
			else if (strcmp((*argv)[i], "-cd")==0) 
			{
				/* push the dir into the environment vars so qio.c can pick it up */
				setenv("QHOSTDIR", (*argv)[++i], 0);
			}
			else if (strcmp((*argv)[i], "-rti")==0) 
			{
				sscanf((*argv)[++i], "%d", &rtiP);
			}
			else if (strcmp((*argv)[i], "-rtinode")==0) 
			{
				int n = strlen((*argv)[++i]);
				if (n >= maxlen)
				{
					QDPIO::cerr << __func__ << ": rtinode name too long" << endl;
					QDP_abort(1);
				}
				sscanf((*argv)[i], "%s", rtinode);
			}
#endif
#if 0
			else 
			{
				QDPIO::cerr << __func__ << ": Unknown argument = " << (*argv)[i] << endl;
				QDP_abort(1);
			}
#endif
			
			if (i >= *argc) 
			{
				QDPIO::cerr << __func__ << ": missing argument at the end" << endl;
				QDP_abort(1);
			}
		}
		
#ifdef QDP_IS_QDPJIT
		if (!paramCC)
		  DeviceParams::Instance().setCC(20);
#endif
		
		QMP_verbose (QMP_verboseP);
		
#if QDP_DEBUG >= 1
		// Print command line args
		for (int i=0; i<*argc; i++) 
			QDP_info("QDP_init: arg[%d] = XX%sXX",i,(*argv)[i]);
#endif
		
#if QDP_DEBUG >= 1
		QDP_info("Now initialize QMP");
#endif
		
		if (QMP_is_initialized() == QMP_FALSE)
		{
			QMP_thread_level_t prv;
			if (QMP_init_msg_passing(argc, argv, QMP_THREAD_SINGLE, &prv) != QMP_SUCCESS)
			{
				QDPIO::cerr << __func__ << ": QMP_init_msg_passing failed" << endl;
				QDP_abort(1);
			}
		}
		
#if QDP_DEBUG >= 1
		QDP_info("QMP inititalized");
#endif
		
		if (setGeomP)
			if (QMP_declare_logical_topology(logical_geom.slice(), Nd) != QMP_SUCCESS)
			{
				QDPIO::cerr << __func__ << ": QMP_declare_logical_topology failed" << endl;
				QDP_abort(1);
			}
		
#if QDP_DEBUG >= 1
		QDP_info("Some layout init");
#endif


#ifdef QDP_IS_QDPJIT
		char * qdp_install = getenv("QDP_INSTALL");
		if (qdp_install) {
		  QDP_info("Using QDP-JIT installation in: %s", qdp_install );
		  QDPJit::Instance().setQDPPath( string( qdp_install ) );
		} else {
		  QDP_error_exit("QDP_INSTALL not set");
		}
		char * qdp_temp = getenv("QDP_TEMP");
		if (qdp_temp) {
		  QDP_info("kernel directory: %s" , qdp_temp );
		  QDPJit::Instance().setKernelPath( string( qdp_temp ) );
		  QDPJit::Instance().loadAllShared();
		} else {
		  QDP_error_exit("QDP_TEMP not set");
		}
		char * quda_rpath = getenv("QUDA_RESOURCE_PATH");
		if (quda_rpath) {
		  QDP_info("QDP-JIT DB: %s" , quda_rpath );
		  
		  JitTuning::Instance().setResourcePath( string(quda_rpath) + "/qdp-jit.xml" );
		  
		  ifstream check_file( JitTuning::Instance().getResourcePath().c_str() );
		  if (check_file.good())
		    JitTuning::Instance().load( JitTuning::Instance().getResourcePath() );
		  
		} else {
		  QDP_error_exit("QUDA_RESOURCE_PATH not set");
		}
#endif

		
		Layout::init();   // setup extremely basic functionality in Layout
		
		isInit = true;
		
#if QDP_DEBUG >= 1
		QDP_info("Init qio");
#endif
		// OK, I need to set up the IO geometry here...
		// I should make it part of layout...
		if( setIOGeomP ) { 
#if QDP_DEBUG >=1
			std::ostringstream outbuf;
			for(int mu=0; mu < Nd; mu++) { 
				outbuf << " " << logical_iogeom[mu];
			}
			
			QDP_info("Setting IO Geometry: %s\n", outbuf.str().c_str());
#endif
			
			Layout::setIONodeGrid(logical_iogeom);
			
		}
		//
		// add qmt inilisisation
		//
#ifdef QDP_USE_QMT_THREADS
		
		// Initialize threads
		if( Layout::primaryNode() ) { 
			cout << "QDP use qmt threading: Initializing threads..." ;
		} 
		int thread_status = qmt_init();
		
		if( thread_status == 0 ) { 
			if (  Layout::primaryNode() ) { 
				cout << "Success. We have " << qdpNumThreads() << " threads \n";
			} 
		}
		else { 
			cout << "Failure... qmt_init() returned " << thread_status << endl;
			QDP_abort(1);
		}
		
#else
#ifdef QDP_USE_OMP_THREADS
		
		if( Layout::primaryNode()) {
			cout << "QDP use OpenMP threading. We have " << qdpNumThreads() << " threads\n"; 
		}
		
#endif
#endif
		
		// Alloc space for reductions
		ThreadReductions::norm2_results = new REAL64 [ qdpNumThreads() ];
		if( ThreadReductions::norm2_results == 0x0 ) { 
			cout << "Failure... space for norm2 results failed "  << endl;
			QDP_abort(1);
		}
		
		ThreadReductions::innerProd_results = new REAL64 [ 2*qdpNumThreads() ];
		if( ThreadReductions::innerProd_results == 0x0 ) { 
			cout << "Failure... space for innerProd results failed "  << endl;
			QDP_abort(1);
		}
		
		
		// initialize the global streams
		QDPIO::cin.init(&std::cin);
		QDPIO::cout.init(&std::cout);
		QDPIO::cerr.init(&std::cerr);
		
		initProfile(__FILE__, __func__, __LINE__);
		
		QDPIO::cout << "Initialize done" << std::endl;
	}
	
	//! Is the machine initialized?
	bool QDP_isInitialized() {return isInit;}
	
	//! Turn off the machine
	void QDP_finalize()
	{
		if ( ! QDP_isInitialized() )
		{
			QDPIO::cerr << "QDP is not inited" << std::endl;
			QDP_abort(1);
		}

#ifdef QDP_IS_QDPJIT
		if( Layout::primaryNode() )
		  JitTuning::Instance().save( JitTuning::Instance().getResourcePath() );
		CUDAHostPoolAllocator::Instance().unregisterMemory();
		QDPJit::Instance().closeAllShared();
#endif
	
		//
		// finalise qmt
		//
		delete [] ThreadReductions::norm2_results;
		delete [] ThreadReductions::innerProd_results;
#if defined(QMT_USE_QMT_THREADS)
		// Finalize threads
		cout << "QDP use qmt threading: Finalizing threads" << endl;
		qmt_finalize();
#endif 
		
		printProfile();
		
		QMP_finalize_msg_passing();
		
		isInit = false;
	}
	
	//! Panic button
	void QDP_abort(int status)
	{
		QMP_abort(status); 
	}
	
	//! Resumes QDP communications
	void QDP_resume() {}
	
	//! Suspends QDP communications
	void QDP_suspend() {}
	
	
} // namespace QDP;
