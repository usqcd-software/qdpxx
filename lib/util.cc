// $Id: util.cc,v 1.2 2002-12-14 01:11:01 edwards Exp $
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

//-----------------------------------------------------------------------------
//! Unique-ify a list
/*! Given a list of ints, collapse it to a list of the unique ints */
multi1d<int>
uniquify_list(const multi1d<int>& ll)
{
  multi1d<int> d(ll.size());

  // Enter the first element as unique to prime the search
  int ipos = 0;
  int num = 0;
  int prev_node;
  
  d[num++] = prev_node = ll[ipos++];

  // Find the unique source nodes
  while (ipos < ll.size())
  {
    for(; ipos < ll.size(); ++ipos)
      if (ll[ipos] != prev_node)
	break;

    int this_node = ll[ipos];

    // Has this node occured before?
    bool found = false;
    for(int i=0; i < num; ++i)
      if (d[i] == prev_node)
      {
	found = true;
	break;
      }

    // If this is the first time this value has occurred, enter it
    if (! found)
      d[num++] = this_node;

    ipos++;
  }

  // Copy into a compact size array
  multi1d<int> dd(num);
  for(int i=0; i < num; ++i)
    dd[i] = d[i];

  return dd;
}


QDP_END_NAMESPACE();
