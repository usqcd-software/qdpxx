// $Id: qdp_scalar_init.cc,v 1.5 2003-10-09 18:27:40 edwards Exp $

/*! @file
 * @brief Scalar init routines
 * 
 * Initialization routines for scalar implementation
 */


#include "qdp.h"

QDP_BEGIN_NAMESPACE(QDP);


//! Private flag for status
static bool isInit = false;

//! Turn on the machine
void QDP_initialize(int *argc, char ***argv) 
{
  Layout::init();   // setup extremely basic functionality in Layout

  isInit = true;

  // initialize remote file service (QIO)
  QDPUtil::RemoteFileInit("qcdi01", false);

  // initialize the global streams
  QDPIO::cin.init(&std::cin);
  QDPIO::cout.init(&std::cout);
  QDPIO::cerr.init(&std::cerr);
}

//! Is the machine initialized?
bool QDP_isInitialized() {return isInit;}

//! Turn off the machine
void QDP_finalize()
{
  // shutdown remote file service (QIO)
  QDPUtil::RemoteFileShutdown();

  isInit = false;
}

//! Panic button
void QDP_abort(int status)
{
  QDP_finalize(); 
  exit(status);
}

//! Resumes QDP communications
void QDP_resume() {}

//! Suspends QDP communications
void QDP_suspend() {}


QDP_END_NAMESPACE();
