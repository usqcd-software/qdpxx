// $Id: qdp_scalar_init.cc,v 1.1 2003-05-22 20:06:29 edwards Exp $

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
}

//! Is the machine initialized?
bool QDP_isInitialized() {return isInit;}

//! Turn off the machine
void QDP_finalize() {isInit = false;}

//! Panic button
void QDP_abort(int status) {QDP_finalize(); exit(status);}



QDP_END_NAMESPACE();
