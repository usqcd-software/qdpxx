#ifndef QDP_STRNLEN_H
#define QDP_STRNLEN_H

#include "qdp_config.h"

#ifndef HAVE_STRNLEN
#include <stdlib.h>
#include <sys/types.h>

extern "C" { 
  size_t strnlen(const char *s, size_t maxlen);
};
#else
#include <string.h>
#endif

#endif
