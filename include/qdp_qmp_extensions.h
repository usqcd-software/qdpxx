#ifndef QDP_QMP_EXTENSIONS_H
#define QDP_QMP_EXTENSIONS_H

#include "qdp_config.h"

#ifndef HAVE_QMP_ABORT
extern "C" { 
	void QMP_abort(int);
};
#endif

#ifndef HAVE_QMP_ERROR_EXIT
extern "C" { 
	void QMP_error_exit (const char* format, ...);
};
#endif

#ifndef HAVE_QMP_VERBOSE
extern "C" { 
 void   QMP_verbose(QMP_bool_t verbose);
};

#endif	
#endif
