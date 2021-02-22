#ifndef QDP_CONFIG_H
#define QDP_CONFIG_H


/* Undef the unwanted from the environment -- eg the compiler command line */
#undef PACKAGE
#undef PACKAGE_BUGREPORT
#undef PACKAGE_NAME
#undef PACKAGE_STRING
#undef PACKAGE_TARNAME
#undef PACKAGE_VERSION
#undef VERSION

/* Include the stuff generated by autoconf */
#include "qdp_config_internal.h"

/* Prefix everything with QDP_ -- these are the only two referenced */
static const char* const QDP_PACKAGE_STRING(PACKAGE_STRING);
static const char* const QDP_PACKAGE_VERSION(PACKAGE_VERSION);

/* Undef the unwanted */
#undef PACKAGE
#undef PACKAGE_BUGREPORT
#undef PACKAGE_NAME
#undef PACKAGE_STRING
#undef PACKAGE_TARNAME
#undef PACKAGE_VERSION
#undef VERSION

#endif
