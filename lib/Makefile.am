# Have to add top_builddir/include to pick up made qdp_config.h

INCFLAGS = -I@top_srcdir@/include -I@top_builddir@/include \
           -I@top_srcdir@/other_libs/qio/include \
           -I@top_srcdir@/other_libs/qio/other_libs/c-dime/include

## GTF: Using @QMP_CFLAGS@ instead of @QMP_CXXFLAGS@ here since QMP
## is written in C, not C++
AM_CXXFLAGS = $(INCFLAGS) @CXXFLAGS@ @QMP_CFLAGS@

VPATH = @top_srcdir@
VPATH += @top_srcdir@/lib
VPATH += @top_srcdir@/include
VPATH += @top_srcdir@/other_libs/qio/include

lib_LIBRARIES = libqdp.a

HDRS = defs.h forward.h globalfuncs.h inner.h io.h layout.h multi.h \
       newops.h outer.h outersubtype.h params.h primcolormat.h \
       primcolorvec.h primgamma.h primitive.h primmatrix.h \
       primscalar.h primseed.h primspinmat.h primspinvec.h primvector.h \
       qdp_util.h qdpexpr.h qdp.h qdp_config.h QDPOperators.h qdpsubtype.h \
       qdptype.h random.h reality.h simpleword.h map.h \
       specializations.h subset.h traits.h word.h qdpio.h   qio.h

## GTF: Major bug fix here.  Had sources which were conditionally included
## in the library based on variable substitution from autoconf.  Basically,
## it meant that automake had no chance of making sure all the sources got
## put into the distribution by 'make dist' because it only saw those sources
## that autoconf told it about based on the chosen architecture.  Using
## automake conditionals fixes the problem, same as was used in QMP.
## Only tedious part will be in the future if we have many choices
## for arch's then a conditional for each one has to be added.

GENERIC_SRCS = map.cc subset.cc random.cc qdp.cc layout.cc io.cc iogauge.cc \
               byteorder.cc util.cc readszin.cc  qdpio.cc

SCALAR_SRCS = scalar_init.cc scalar_layout.cc scalar_specific.cc \
              ../include/scalar_specific.h

PARSCALAR_SRCS = parscalar_init.cc parscalar_layout.cc parscalar_specific.cc \
                 ../include/parscalar_specific.h

if ARCH_SCALAR
  libqdp_a_SOURCES = $(GENERIC_SRCS) $(SCALAR_SRCS)
  EXTRA_libqdp_a_SOURCES = $(PARSCALAR_SRCS)
endif

if ARCH_PARSCALAR
  libqdp_a_SOURCES = $(GENERIC_SRCS) $(PARSCALAR_SRCS)
  EXTRA_libqdp_a_SOURCES = $(SCALAR_SRCS)
endif

# Need this header as part of libqdp_a_sources
nodist_libqdp_a_SOURCES = $(HDRS)
