# Have to add top_builddir/include to pick up made qdp_config.h

INCFLAGS = -I@top_srcdir@/include -I@top_builddir@/include \
           -I@top_srcdir@/other_libs/nml \
           -I@top_srcdir@/other_libs/qio/include \
           -I@top_srcdir@/other_libs/qio/other_libs/c-dime/include \
           -I@top_srcdir@/other_libs/xpath_reader/include

## GTF: Using @QMP_CFLAGS@ instead of @QMP_CXXFLAGS@ here since QMP
## is written in C, not C++
AM_CXXFLAGS = $(INCFLAGS) @CXXFLAGS@ @LIBXML2_CXXFLAGS@ @QMP_CFLAGS@

VPATH = @top_srcdir@
VPATH += @top_srcdir@/lib
VPATH += @top_srcdir@/include
VPATH += @top_srcdir@/other_libs/qio/include

lib_LIBRARIES = libqdp.a

HDRS = qdp_defs.h qdp_forward.h qdp_globalfuncs.h qdp_inner.h qdp_io.h qdp_layout.h qdp_multi.h \
       qdp_newops.h qdp_outer.h qdp_outersubtype.h qdp_params.h qdp_primcolormat.h \
       qdp_primcolorvec.h qdp_primgamma.h qdp_primitive.h qdp_primmatrix.h \
       qdp_primscalar.h qdp_primseed.h qdp_primspinmat.h qdp_primspinvec.h qdp_primvector.h \
       qdp_util.h qdp_qdpexpr.h qdp.h qdp_config.h QDPOperators.h qdp_qdpsubtype.h \
       qdp_qdptype.h qdp_random.h qdp_reality.h qdp_simpleword.h qdp_map.h \
       qdp_specializations.h qdp_subset.h qdp_traits.h qdp_word.h qdp_qdpio.h \
       qio.h qdp_xmlio.h xml_simplewriter.h qdp_init.h

## GTF: Major bug fix here.  Had sources which were conditionally included
## in the library based on variable substitution from autoconf.  Basically,
## it meant that automake had no chance of making sure all the sources got
## put into the distribution by 'make dist' because it only saw those sources
## that autoconf told it about based on the chosen architecture.  Using
## automake conditionals fixes the problem, same as was used in QMP.
## Only tedious part will be in the future if we have many choices
## for arch's then a conditional for each one has to be added.

libqdp_a_SOURCES = qdp_map.cc qdp_subset.cc qdp_random.cc qdp.cc qdp_layout.cc qdp_io.cc qdp_iogauge.cc \
               qdp_byteorder.cc qdp_util.cc qdp_qdpio.cc qdp_xmlio.cc   qdp_filebuf.cc

# Scalar
if ARCH_SCALAR
libqdp_a_SOURCES += qdp_scalar_init.cc qdp_scalar_layout.cc qdp_scalar_specific.cc \
              qdp_scalarsite_specific.cc \
              ../include/qdp_scalar_specific.h
endif

# Parallel-scalar
if ARCH_PARSCALAR
libqdp_a_SOURCES += qdp_parscalar_init.cc qdp_parscalar_layout.cc qdp_parscalar_specific.cc \
                 qdp_scalarsite_specific.cc \
                 ../include/qdp_parscalar_specific.h
endif

# Optimized code using sse extensions
if QDP_USE_SCALAR_SSE
libqdp_a_SOURCES += qdp_scalarsite_sse.cc ../include/qdp_scalarsite_sse.h
endif


# Scalar with Vector extensions
if ARCH_SCALARVEC
libqdp_a_SOURCES += qdp_scalarvec_init.cc qdp_scalarvec_layout.cc qdp_scalarvec_specific.cc \
                 qdp_scalarvecsite_specific.cc \
                 ../include/qdp_scalarvec_specific.h
endif

# Parallel scalar with Vector extensions
if ARCH_PARSCALARVEC
libqdp_a_SOURCES += qdp_parscalarvec_init.cc qdp_parscalarvec_layout.cc qdp_parscalarvec_specific.cc \
                    qdp_scalarvecsite_specific.cc \
                    ../include/qdp_parscalarvec_specific.h
endif

# Optimized code using sse extensions
if QDP_USE_SCALARVEC_SSE
libqdp_a_SOURCES += qdp_scalarvecsite_sse.cc ../include/qdp_scalarvecsite_sse.h
endif


# Need this header as part of libqdp_a_sources
nodist_libqdp_a_SOURCES = $(HDRS)
