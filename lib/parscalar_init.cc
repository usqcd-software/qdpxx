// $Id: parscalar_init.cc,v 1.1 2003-01-14 04:46:26 edwards Exp $

/*! @file
 * @brief Parscalar init routines
 * 
 * Initialization routines for parscalar implementation
 */


#include "qdp.h"
#include "QMP.h"

QDP_BEGIN_NAMESPACE(QDP);

//! Private flag for status
static bool isInit = false;

//! Turn on the machine
void QDP_initialize(int *argc, char ***argv)
{
  if (isInit)
    QDP_error_exit("QDP already inited");

  QMP_verbose (QMP_TRUE);

  if (QMP_init_msg_passing(argc, argv, QMP_SMP_ONE_ADDRESS) != QMP_SUCCESS)
    QDP_error_exit("QDP_initialize failed");

  isInit = true;
}

//! Is the machine initialized?
bool QDP_isInitialized() {return isInit;}

//! Turn off the machine
void QDP_finalize()
{
  if ( ! QDP_isInitialized() )
    QDP_error_exit("QDP is not inited");

  QMP_finalize_msg_passing();
  isInit = false;
}

//! Panic button
void QDP_abort(int status)
{
  QDP_finalize(); 
  exit(status);
}

QDP_END_NAMESPACE();
