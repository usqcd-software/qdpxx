/**
 * This is a hand tweaked header file for nvcc to compile qdp on cuda devices
 */
/* Scalar architecture */
#define ARCH_SCALAR 1

/* Double precision */
#define BASE_PRECISION 32

/* Name of package */
#define PACKAGE "qdp--"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "edwards@jlab.org"

/* Define to the full name of this package. */
#define PACKAGE_NAME "qdp++"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "qdp++ 1.38.3"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "qdp--"

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION "1.38.3"

/* Number of color components */
#define QDP_NC 3

/* Number of dimensions */
#define QDP_ND 4

/* Number of spin components */
#define QDP_NS 4

/* Use checkerboarded layout */
#define QDP_USE_CB2_LAYOUT 1

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Version number of package */
#define VERSION "1.38.3"

#ifndef PETE_DEVICE
#define PETE_DEVICE __device__
#endif
