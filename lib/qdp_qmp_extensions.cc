#include "qmp.h"
#include "qdp_qmp_extensions.h"
#include <stdlib.h>
#include <stdio.h>

extern "C" { 
#ifndef HAVE_QMP_ABORT
void QMP_abort(int status)
{
	exit(status);
}
#endif

#ifndef HAVE_QMP_ERROR_EXIT
#include <stdarg.h>

void QMP_error_exit (const char* format, ...)
{
  va_list argp;
  int i;
  int status;

  char loc_info[256];
  char buffer[1024];

  QMP_u32_t ndims = QMP_get_logical_number_of_dimensions();
  QMP_u32_t node_nr = QMP_get_node_number();
  const QMP_u32_t*  coords  = QMP_get_logical_coordinates();

  sprintf(loc_info, "Node=%d Logical Coords:", node_nr);
  for(i=0; i < ndims; i++) {
    sprintf(loc_info, " %d", coords[i]);
  }


  va_start(argp, format);
  status = vsprintf(buffer, format, argp);
  va_end(argp);

  fprintf(stderr, "%s: FATAL ERROR: %s\n", loc_info, buffer);
  fflush(stdout);

  QMP_abort(QMP_ERROR);
}
#endif

#ifndef HAVE_QMP_VERBOSE
void   QMP_verbose(QMP_bool_t verbose) 
{
   fprintf(stderr, "QMP_verbose does nothing for me or you\n");
}
#endif

};
 
