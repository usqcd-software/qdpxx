// -*- C++ -*-
// $Id: qdp_db.h,v 1.3 2008-08-04 17:40:37 edwards Exp $
/*! @file
 * @brief Support for ffdb-lite - a wrapper over Berkeley DB
 */

#ifndef QDP_DB_H
#define QDP_DB_H

#ifdef BUILD_FFDB_LITE

// Berkeley DB is used, so define a real set of classes
#include "qdp_db_imp.h"

#else

// Berkeley DB is NOT used, so define stubs of classes
#include "qdp_db_stub.h"
#endif

#endif
