# Have to add top_builddir/include to pick up made qdp_config.h

INCFLAGS=-I@top_srcdir@/include -I@top_builddir@/include

AM_CXXFLAGS = $(INCFLAGS) @CXXFLAGS@ @QMP_CXXFLAGS@

VPATH = @top_srcdir@
VPATH += @top_srcdir@/lib
VPATH += @top_srcdir@/include

lib_LIBRARIES = libqdp.a

HDRS = defs.h forward.h globalfuncs.h inner.h io.h layout.h multi.h \
       newops.h outer.h outersubtype.h params.h primcolormat.h \
       primcolorvec.h primgamma.h primitive.h primmatrix.h \
       primscalar.h primseed.h primspinmat.h primspinvec.h primvector.h \
       proto.h qdpexpr.h qdp.h qdp_config.h QDPOperators.h qdpsubtype.h \
       qdptype.h random.h reality.h simpleword.h \
       specializations.h subset.h traits.h word.h

ARCH_SRCS = @ARCH_SRCS@
ARCH_HDRS = @ARCH_HDRS@
ARCH_OBJS = $(ARCH_SRCS:.cc=.o)

libqdp_a_SOURCES = subset.cc random.cc qdp.cc layout.cc io.cc \
	iogauge.cc byteorder.cc util.cc readszin.cc

# Need this header as part of libqdp_a_sources
nodist_libqdp_a_SOURCES = $(HDRS)

libqdp_a_LIBADD = $(ARCH_OBJS)
libqdp_a_DEPENDENCIES = $(ARCH_HDRS) $(ARCH_OBJS) $(ARCH_SRCS)
