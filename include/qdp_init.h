// -*- C++ -*-

/*! \file
 * \brief Routines for top level QDP management
 *
 * Fundamental routines for turning on/off and inserting/extracting
 * variables.
 */

// Info/error routines
namespace QDP {

#ifdef QDP_IS_QDPJIT
  extern bool QDPuseGPU;
  void QDP_setGPU();
  void QDP_startGPU();
#endif

//! Turn on the machine
void QDP_initialize (int *argc, char ***argv);

//! Is the machine initialized?
bool QDP_isInitialized ();

//! Turn off the machine
void QDP_finalize ();

//! Panic button
void QDP_abort (int status);

//! Simple information display routine
int  QDP_info (const char* format, ...);

#ifdef QDP_IS_QDPJIT
//! Simple information display routine
int  QDP_info_primary (const char* format, ...);

//! Simple debug display routine
int  QDP_debug (const char* format, ...);

//! Simple deep debug display routine
int  QDP_debug_deep (const char* format, ...);
#endif

//! Simple error display routine
int  QDP_error (const char* format, ...);

//! Simple error display and abort routine
void QDP_error_exit (const char *format, ...);

//! Resumes QDP communications
void QDP_resume();

//! Suspends QDP communications
void QDP_suspend();

}

