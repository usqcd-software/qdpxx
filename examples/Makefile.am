#
# Boiler Plate stuff: Autoconf Substitutions etc
#
## Should only need @QMP_CFLAGS@, @QMP_LDFLAGS@ and @QMP_LIBS@
## n.b. Only defined for some parallel architectures
INCFLAGS = -I@top_srcdir@/include -I@top_builddir@/include \
           -I@top_srcdir@/other_libs/nml \
           -I@top_srcdir@/other_libs/qio/include \
           -I@top_srcdir@/other_libs/qio/other_libs/c-dime/include \
           -I@top_srcdir@/other_libs/xpath_reader/include

AM_CXXFLAGS = $(INCFLAGS) @CXXFLAGS@ @LIBXML2_CXXFLAGS@ @QMP_CFLAGS@
AM_LDFLAGS = -L@top_builddir@/lib \
             -L@top_builddir@/other_libs/nml \
             -L@top_builddir@/other_libs/qio/lib \
             -L@top_builddir@/other_libs/qio/other_libs/c-dime/lib \
             -L@top_builddir@/other_libs/xpath_reader/lib \
             @LDFLAGS@ @QMP_LDFLAGS@
LDADD = -lqdp -lnml -lXPathReader -lxmlWriter -lqio -ldime @LIBXML2_LIBS@ @QMP_LIBS@ @LIBS@

#
# Local Headers
#
HDRS=examples.h 

#
# VPATH
#
VPATH=@top_srcdir@
VPATH+=@top_srcdir@/examples

#
# The programs to build
# 
check_PROGRAMS= t_skeleton t_dslashm t_formfac t_io t_mesplq t_qdp t_spectrum t_foo \
      t_nml t_xml t_qio t_remote t_entry t_linalg t_nersc

#
# The program and its dependencies
t_skeleton_SOURCES = t_skeleton.cc
t_remote_SOURCES = t_remote.cc $(HDRS)
t_foo_SOURCES = t_foo.cc $(HDRS)
t_nml_SOURCES = t_nml.cc $(HDRS)
t_xml_SOURCES = t_xml.cc $(HDRS)
t_qio_SOURCES = t_qio.cc $(HDRS)
t_io_SOURCES = t_io.cc $(HDRS)
t_entry_SOURCES = t_entry.cc $(HDRS)
t_nersc_SOURCES = t_nersc.cc mesplq.cc $(HDRS)
t_dslashm_SOURCES = t_dslashm.cc dslashm_w.cc $(HDRS)
t_mesplq_SOURCES = t_mesplq.cc mesplq.cc reunit.cc $(HDRS)
t_formfac_SOURCES = t_formfac.cc formfac_w.cc $(HDRS)
t_spectrum_SOURCES = t_spectrum.cc formfac_w.cc dslashm_w.cc \
          baryon_w.cc mesons_w.cc mesplq.cc reunit.cc $(HDRS)
t_qdp_SOURCES =  t_qdp.cc formfac_w.cc dslashm_w.cc baryon_w.cc \
	 mesons_w.cc mesplq.cc reunit.cc $(HDRS)
t_linalg_SOURCES =  t_linalg.cc linalg1.cc linalg.h

# build lib is a target that goes tot he build dir of the library and 
# does a make to make sure all those dependencies are OK. In order
# for it to be done every time, we have to make it a 'phony' target
.PHONY: build_lib
build_lib:
	cd @top_builddir@/lib ; $(MAKE)

