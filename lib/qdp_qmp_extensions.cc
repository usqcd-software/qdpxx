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

#ifndef HAVE_QMP_ROUTE
#include <string.h>

/* My own implementation of QMP_route */
QMP_status_t QMP_route(void* buffer, QMP_u32_t count,
		       QMP_u32_t src, QMP_u32_t dest)
{


  QMP_u32_t* l_src_coords;      /* Coordinates of the source */
  QMP_u32_t* l_dst_coords;      /* Coordinates of the destination */
  int* l_disp_vec;              /* Displacement vector dst_coords - src_coords */
  
  QMP_msgmem_t sendbufmm, recvbufmm; /* Message memory handles */

  void *sendbuf;                   /* These are the actual comms buffers */
  void *recvbuf;                 

  QMP_s32_t i,j;                   /* Loop Counters. Use for directions too */
  QMP_u32_t me;                    /* My node */

  int n_hops;                      /* Number of hops */
  QMP_s32_t  direction_sign;       /* Direction of hops */

  QMP_msghandle_t send_handle;     /* A handle for sending */
  QMP_msghandle_t recv_handle;     /* A handle for receiving */

  QMP_status_t err;                /* Error status */
  
  /* The number of dimensions in our "grid" */
  QMP_u32_t ndim;

  QMP_u32_t bufsize;

  QMP_bool_t log_top_declP;

  /* Check to see if the logical topology is declared or not */
  log_top_declP = QMP_logical_topology_is_declared();
  
  if ( log_top_declP == QMP_FALSE ) { 
    fprintf(stderr, "QMP_route: QMP logical topology MUST be declared\n");
    fprintf(stderr, "It appears not to be\n");
    return QMP_TOPOLOGY_EXISTS;
  }

  /* Topology is declared */
  /* Get its details */
  ndim = QMP_get_logical_number_of_dimensions();

  /* Get my node number -- use it to see whether I am source or dest */
  me = QMP_get_node_number();

  /* Must free these later I think */
  /* Allocate space for the coordinates */
  l_src_coords =(QMP_u32_t *)QMP_get_logical_coordinates_from(src);
  if( l_src_coords == (QMP_u32_t *)NULL ) { 
    fprintf(stderr, "QMP_route: QMP_get_logical_coordinates_from failed\n");
    return QMP_NOMEM_ERR;
  }

  l_dst_coords = (QMP_u32_t *)QMP_get_logical_coordinates_from(dest);
  if( l_dst_coords == (QMP_u32_t *)NULL ) { 
    fprintf(stderr, "QMP_route: QMP_get_logical_coordinates_from failed\n");
    return QMP_NOMEM_ERR;
  }

  /* Will definitely have to free this */
  l_disp_vec = (int *)malloc(sizeof(int)*ndim);
  if( l_disp_vec == (int *)NULL ) {
    fprintf(stderr, "QMP_route: Unable to allocate displacement array\n");
    return QMP_NOMEM_ERR;
  }

  /* Compute the displacement */
  for(i=0; i < ndim; i++) {
    l_disp_vec[i] = l_dst_coords[i] - l_src_coords[i];
  }

  /* Don't need these anymore */
  /* QMP_get_logical_coordinates_from ought to have used malloc to get these */
  free(l_src_coords);
  free(l_dst_coords);

  /* Pad the buffers so that their lengths are always divisible by 8 */
  /* This is a funky QCDOC-ism -- maybe */
  bufsize = count;
  if( count % 8 != 0 ) { 
    bufsize += (8 - (count % 8));
  }

  /* Will have to free these with QMP_free_aligned_memory */
  sendbuf = QMP_allocate_aligned_memory(bufsize);
  if( sendbuf == NULL ) { 
    fprintf(stderr, "Unable to allocate sendbuf in QMP_route\n");
    return QMP_NOMEM_ERR;
  }

  recvbuf = QMP_allocate_aligned_memory(bufsize);
  if( recvbuf == NULL ) { 
    fprintf(stderr ,"Unable to allocate recvbuf in QMP_route\n");
    return QMP_NOMEM_ERR;
  }

  /* To start with -- the first thing I have to do, is to copy
     the message into my sendbuf if I am the sender. Otherwise 
     I really don't care what junk is in there. */

  if( me == src ) {
    memcpy( sendbuf, buffer, count);
  }
  else {
    /* I don't care what my buffer contains if I am not the source
       but it may be nice to set it to zero so I don't send complete 
       garbage */

    memset( sendbuf, 0, count);
  }
  /*   
       Now Roll around
  */

  /* Declare the message memories */
  sendbufmm = QMP_declare_msgmem(sendbuf, bufsize);
  recvbufmm = QMP_declare_msgmem(recvbuf, bufsize);

  /* For each dimension do */
  for(i=0; i < ndim; i++) { 
    
    /* If the displacement in this direction is nonzero */
    if( l_disp_vec[i] != 0 ) {    

      /* Get the number of hops */
      n_hops = abs(l_disp_vec[i]);

      /* Get the direction */
      direction_sign = ( l_disp_vec[i] > 0 ?  1 : -1 );

      /* Declare relative sends in that direction */
      /* Do N Hops , in the direction. */
      /* I can re-use the handles for this direction */

      /* Create a receive handle in -direction sign */
      recv_handle = QMP_declare_receive_relative(recvbufmm, 
						 i,
						 -direction_sign,
						 0);

      if( recv_handle == NULL) { 
	fprintf(stderr, "QMP_declare_receive_relative returned NULL\n");
	return QMP_BAD_MESSAGE;
      }

      /* Create a send handle in direction sign */
      send_handle = QMP_declare_send_relative(sendbufmm, 
					      i, 
					      direction_sign, 
					      0);

      if( send_handle == NULL ) { 
	fprintf(stderr, "QMP_declare_send_relative returned NULL\n");
	return QMP_BAD_MESSAGE;
      }
	
      /* Do the hops */
      for(j=0; j < n_hops; j++) { 
	/* Start receiving */
	err = QMP_start(recv_handle);
	if(err != QMP_SUCCESS ) { 
	  QMP_error_exit("QMP_start() failed on receive in QMP_route\n");
	}

	/* Start sending */
	err = QMP_start(send_handle);
	if(err != QMP_SUCCESS ) { 
	  fprintf(stderr, "QMP_start() failed on receive in QMP_route\n");
	  return QMP_ERROR;
	}
	
	/* Wait for send to complete */
	err = QMP_wait(send_handle);
	if( err != QMP_SUCCESS ) { 
	  fprintf(stderr, "QMP_wait() failed on send in QMP_route\n");
	  return QMP_ERROR;
	}

	/* Wait for receive to complete */
	err = QMP_wait(recv_handle);
	if( err != QMP_SUCCESS ) { 
	  fprintf(stderr, "QMP_wait() recv on send in QMP_route\n");
	  return QMP_ERROR;
	}

	/* Copy the contents of my recvbuf into my sendbuf, 
	   ready for the next hop  -- In theory I could 
	   pointer swap here, but this is clean if slow */

	memcpy(sendbuf, recvbuf, count);

      }  /* Data is now in sendbuf */

      /* We have now done n_hops shifts. We need to change 
	 direction, so I free the message handles */
      QMP_free_msghandle(send_handle);
      QMP_free_msghandle(recv_handle);

    }

    /* Next direction */
  }
  
  /* We have now rolled around all the dimensions */
  /* The data is in the send buffer */
  /* Take it out and put it in "buffer" on the destination node only */
  if( me == dest ) { 
    memcpy(buffer, sendbuf, count);
  }
  else {
    memset(buffer, 0, count);
  }


  /* We can now free a whole  bunch of stuff */
  QMP_free_msgmem(sendbufmm);
  QMP_free_msgmem(recvbufmm);
  QMP_free_aligned_memory(sendbuf);
  QMP_free_aligned_memory(recvbuf);

  /* Alloced with malloc */
  free(l_disp_vec);

  return(QMP_SUCCESS);
}

#endif

#ifndef HAVE_QMP_VERBOSE
void   QMP_verbose(QMP_bool_t verbose) 
{
   fprintf(stderr, "QMP_verbose does nothing for me or you\n");
}
#endif

};
 
