# Have to add top_builddir/include to pick up made qdp_config_internal.h

INCFLAGS = -I@top_srcdir@/include -I@top_builddir@/include \
           -I@top_srcdir@/other_libs/qio/include \
           -I@top_builddir@/other_libs/qio/include \
           -I@top_srcdir@/other_libs/qio/other_libs/c-lime/include \
           -I@top_builddir@/other_libs/qio/other_libs/c-lime/include \
           -I@top_srcdir@/other_libs/xpath_reader/include


if BUILD_LIBINTRIN
INCFLAGS += -I@top_srcdir@/other_libs/@LIBINTRIN_DIR@/include -I@top_builddir@/other_libs/@LIBINTRIN_DIR@/include
endif

AM_CXXFLAGS = $(INCFLAGS) @BAGEL_QDP_CXXFLAGS@ @LIBXML2_CXXFLAGS@ @QMP_CFLAGS@

lib_LIBRARIES = libqdp.a

## Base set of sources and conditionally included sources
libqdp_a_SOURCES = qdp_map.cc qdp_subset.cc qdp_random.cc qdp.cc \
	qdp_layout.cc qdp_io.cc qdp_iogauge.cc qdp_byteorder.cc qdp_util.cc \
	qdp_qdpio.cc qdp_filebuf.cc qdp_stdio.cc \
        qdp_profile.cc qdp_strnlen.cc qdp_crc32.cc \
        qdp_stopwatch.cc qdp_qio_strings.cc \
        qdp_tree_rep.cc qdp_tree_io.cc \
        qdp_xml_io.cc qdp_xml_imp.cc \
        qdp_aff_io.cc qdp_aff_imp.cc

# This has been decomissioned. RGE is not aware of it being used.
# qdp_binx.cc   

# Scalar	
if ARCH_SCALAR
libqdp_a_SOURCES += qdp_scalar_init.cc qdp_scalar_layout.cc \
	qdp_scalar_specific.cc qdp_scalarsite_specific.cc
endif

# Parallel-scalar
if ARCH_PARSCALAR
libqdp_a_SOURCES += qdp_parscalar_init.cc qdp_parscalar_layout.cc \
	qdp_parscalar_specific.cc \
	qdp_scalarsite_specific.cc
endif

# Optimized code using sse extensions
if QDP_USE_SCALAR_SSE
libqdp_a_SOURCES += qdp_scalarsite_sse.cc
endif


# Scalar with Vector extensions
if ARCH_SCALARVEC
libqdp_a_SOURCES += qdp_scalarvec_init.cc qdp_scalarvec_layout.cc \
	qdp_scalarvec_specific.cc qdp_scalarvecsite_specific.cc
endif

# Parallel scalar with Vector extensions
if ARCH_PARSCALARVEC
libqdp_a_SOURCES += qdp_parscalarvec_init.cc qdp_parscalarvec_layout.cc \
	qdp_parscalarvec_specific.cc qdp_scalarvecsite_specific.cc
endif

# Optimized code using sse extensions
if QDP_USE_SCALARVEC_SSE
libqdp_a_SOURCES += qdp_scalarvecsite_sse.cc
endif

if BUILD_DEFAULT_ALLOCATOR
libqdp_a_SOURCES += qdp_default_allocator.cc
endif 

if QDP_USE_QCDOC
libqdp_a_SOURCES += qdp_qcdoc_allocator.cc
endif
