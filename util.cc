// $Id: util.cc,v 1.1 2002-11-04 04:32:25 edwards Exp $
//
/*! 
 * @file
 * @brief Utility routines like info and error reporting
 */

#include <cstdarg>
#include <unistd.h>

#include "qdp.h"

QDP_BEGIN_NAMESPACE(QDP);

/**
 * Simple information display routine
 */
int
QDP_info (const char* format, ...)
{
  va_list argp;
  char    info[128], hostname[256];
  char    buffer[1024];

  /* get machine host name */
  gethostname (hostname, sizeof (hostname) - 1);
  int rank = Layout::nodeNumber();
  int size = Layout::numNodes();
  sprintf (info, "QDP m%d,n%d@%s info: ", 
	   rank, size, hostname);

  va_start (argp, format);
  int status = vsprintf (buffer, format, argp);
  va_end (argp);

  fprintf (stderr, "%s %s\n", info, buffer);
  return status;
}


/**
 * Simple error display routine
 */
int
QDP_error (const char* format, ...)
{
  va_list argp;
  char    info[128], hostname[256];
  char    buffer[1024];

  /* get machine host name */
  gethostname (hostname, sizeof (hostname) - 1);
  int rank = Layout::nodeNumber();
  int size = Layout::numNodes();
  sprintf (info, "QDP m%d,n%d@%s error: ", rank, size, hostname);
	   
  va_start (argp, format);
  int status = vsprintf (buffer, format, argp);
  va_end (argp);

  fprintf (stderr, "%s %s\n", info, buffer);
  return status;
}

/**
 * Simple error display and abort routine
 */
void
QDP_error_exit (const char* format, ...)
{
  va_list argp;
  char    info[128], hostname[256];
  char    buffer[1024];

  /* get machine host name */
  gethostname (hostname, sizeof (hostname) - 1);
  int rank = Layout::nodeNumber();
  int size = Layout::numNodes();
  sprintf (info, "QDP m%d,n%d@%s fatal error: ", rank, size, hostname);
	   
  va_start (argp, format);
  vsprintf (buffer, format, argp);
  va_end (argp);

  fprintf (stderr, "%s %s\n", info, buffer);

  Layout::abort(1);  // Layout abort knows how to shutdown the machine
}

#if 0
/**
 * Enable or Disable verbose mode.
 */
void
QDP_verbose (bool verbose)
{
  QDP_global_m.verbose = verbose;
}
#endif

QDP_END_NAMESPACE();
