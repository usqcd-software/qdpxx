// $Id: qdp_parscalarvec_init.cc,v 1.2 2003-09-03 01:24:19 edwards Exp $

/*! @file
 * @brief Parscalarvec init routines
 * 
 * Initialization routines for parscalarvec implementation
 */

#include <stdlib.h>
#include <unistd.h>

#include "qdp.h"
#include "qmp.h"

QDP_BEGIN_NAMESPACE(QDP);

//! Private flag for status
static bool isInit = false;

//! Turn on the machine
void QDP_initialize(int *argc, char ***argv)
{
  if (isInit)
    QDP_error_exit("QDP already inited");

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

  int rtiP = 0;
  int QMP_verboseP = 0;
  const int maxlen = 256;
  char rtinode[maxlen];
  strncpy(rtinode, "your_local_food_store", maxlen);

  // Usage
  if (Layout::primaryNode()) 
    if (help_flag) 
    {
      fprintf(stderr,"Usage:    %s options\n",(*argv)[0]);
      fprintf(stderr,"options:\n");
      fprintf(stderr,"    -h                 help\n");
      fprintf(stderr,"    -V        %%d [%d] verbose mode for QMP\n", 
	      QMP_verboseP);
#ifdef USE_REMOTE_QIO
      fprintf(stderr,"    -cd       %%s [.] set working dir for QIO interface\n");
      fprintf(stderr,"    -rti      %%d [%d] use run-time interface\n", 
	      rtiP);
      fprintf(stderr,"    -rtinode  %%s [%s] run-time interface fileserver node\n", 
	      rtinode);
#endif
      
      exit(1);
    }

  for (int i=1; i<*argc; i++) 
  {
    cerr << "arg = " << (*argv)[i] << endl;

    if (strcmp((*argv)[i], "-V")==0) 
    {
      cerr << "Use verbose mode" << endl;
      QMP_verboseP = 1;
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
	QDP_error_exit("rtinode name too long");
      sscanf((*argv)[i], "%s", rtinode);
    }
#endif
#if 0
    else 
    {
      QDP_error_exit("Unknown argument %s\n", (*argv)[i]);
    }
#endif
 
    if (i >= *argc) 
    {
      QDP_error_exit("missing argument at the end");
    }
  }


  QMP_verbose (QMP_verboseP);

  QDP_info("Now initialize QMP");

  if (QMP_init_msg_passing(argc, argv, QMP_SMP_ONE_ADDRESS) != QMP_SUCCESS)
    QDP_error_exit("QDP_initialize failed");

  QDP_info("Some layout init");

  Layout::init();   // setup extremely basic functionality in Layout

  isInit = true;

  QDP_info("Init qio");

  // initialize remote file service (QIO)
  bool use_qio = (rtiP != 0) ? true : false;
  QDPUtil::RemoteFileInit(rtinode, use_qio);

  QDP_info("Initialize done");
}

//! Is the machine initialized?
bool QDP_isInitialized() {return isInit;}

//! Turn off the machine
void QDP_finalize()
{
  if ( ! QDP_isInitialized() )
    QDP_error_exit("QDP is not inited");

  // shutdown remote file service (QIO)
  QDPUtil::RemoteFileShutdown();

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


QDP_END_NAMESPACE();
