#
# Boiler Plate stuff: Autoconf Substitutions etc
#
## Should only need @QMP_CFLAGS@, @QMP_LDFLAGS@ and @QMP_LIBS@
## n.b. Only defined for some parallel architectures
INCFLAGS = -I@top_srcdir@/include -I@top_builddir@/include \
           -I@top_srcdir@/other_libs/qio/include \
	   -I@top_builddir@/other_libs/qio/include \
           -I@top_srcdir@/other_libs/qio/other_libs/c-lime/include \
           -I@top_srcdir@/other_libs/xpath_reader/include

AM_CXXFLAGS = $(INCFLAGS)  @LIBXML2_CXXFLAGS@ @QMP_CFLAGS@
AM_LDFLAGS = -L@top_builddir@/lib \
             -L@top_builddir@/other_libs/qio/lib \
             -L@top_builddir@/other_libs/qio/other_libs/c-lime/lib \
             -L@top_builddir@/other_libs/xpath_reader/lib \
             @LDFLAGS@ @QMP_LDFLAGS@
LDADD = -lqdp -lXPathReader -lxmlWriter -lqio -llime @LIBXML2_LIBS@ @QMP_LIBS@ @LIBS@

#
# Local Headers
#
HDRS=examples.h 

#
# The programs to build
# 
check_PROGRAMS = t_skeleton t_io t_mesplq \
      t_xml t_remote t_entry t_nersc t_shift t_exotic t_basic t_qio \
      t_cugauge

EXTRA_PROGRAMS  = t_qio_factory t_binx


if BUILD_WILSON_EXAMPLES
check_PROGRAMS += t_dslashm t_formfac t_spectrum t_qdp t_linalg 
EXTRA_PROGRAMS += t_foo t_blas t_cblas t_blas_g5 t_blas_g5_2 t_blas_g5_3
endif


if BUILD_STAGGERED_EXAMPLES
check_PROGRAMS += 
endif


#
# The program and its dependencies
t_skeleton_SOURCES = t_skeleton.cc
t_remote_SOURCES = t_remote.cc $(HDRS)
t_foo_SOURCES = t_foo.cc $(HDRS)
t_xml_SOURCES = t_xml.cc $(HDRS)
t_binx_SOURCES = t_binx.cc $(HDRS)
t_qio_SOURCES = t_qio.cc $(HDRS)
t_qio_DEPENDENCIES = build_lib rebuild_other_libs
t_qio_factory_SOURCES = t_qio_factory.cc $(HDRS)
t_qio_factory_DEPENDENCIES = build_lib rebuild_other_libs

t_basic_SOURCES = t_basic.cc
t_shift_SOURCES = t_shift.cc $(HDRS)
t_io_SOURCES = t_io.cc $(HDRS)
t_entry_SOURCES = t_entry.cc $(HDRS)
t_nersc_SOURCES = t_nersc.cc mesplq.cc reunit.cc $(HDRS)
t_dslashm_SOURCES = t_dslashm.cc dslashm_w.cc $(HDRS)
t_mesplq_SOURCES = t_mesplq.cc mesplq.cc reunit.cc $(HDRS)
t_exotic_SOURCES = t_exotic.cc rgauge.cc taproj.cc expm12.cc reunit.cc $(HDRS)
t_formfac_SOURCES = t_formfac.cc formfac_w.cc $(HDRS)
t_spectrum_SOURCES = t_spectrum.cc baryon_w.cc mesons_w.cc mesplq.cc $(HDRS)
t_qdp_SOURCES =  t_qdp.cc formfac_w.cc dslashm_w.cc baryon_w.cc \
	 mesons_w.cc mesplq.cc reunit.cc $(HDRS)
t_linalg_SOURCES =  t_linalg.cc linalg1.cc linalg.h

t_blas_SOURCES = t_blas.cc blas1.cc blas1.h
t_cblas_SOURCES= t_cblas.cc cblas1.cc cblas1.h 

t_blas_g5_SOURCES = t_blas_g5.cc $(HDRS)
t_blas_g5_2_SOURCES = t_blas_g5_2.cc $(HDRS)
t_blas_g5_3_SOURCES = t_blas_g5_3.cc $(HDRS)

t_cugauge_SOURCES = t_cugauge.cc mesplq.cc $(HDRS) 

# build lib is a target that goes tot he build dir of the library and 
# does a make to make sure all those dependencies are OK. In order
# for it to be done every time, we have to make it a 'phony' target
.PHONY: build_lib
build_lib:
	cd @top_builddir@/lib ; $(MAKE)

.PHONY: rebuild_other_libs
rebuild_other_libs:
	cd @top_builddir@/other_libs ; $(MAKE)
