#
# Boiler Plate stuff: Autoconf Substitutions etc
#
INCFLAGS=-I@top_srcdir@/include -I@top_builddir@/include
AM_CXXFLAGS = $(INCFLAGS) @QMP_CXXFLAGS@ @CXXFLAGS@
AM_LDFLAGS = -L@top_builddir@/lib @QMP_LDFLAGS@ @QMP_COMMS_LDFLAGS@ @LDFLAGS@
LDADD = @LIBS@ -lqdp @QMP_LIBS@ @QMP_COMMS_LIBS@ @LIBS@

SUBDIRS = propagator

#
# Local Headers
#
HDRS=tests.h 

#
# VPATH
#
VPATH=@top_srcdir@
VPATH+=@top_srcdir@/examples

#
# The programs to build
# 
check_PROGRAMS=t_dslashm t_formfac t_io t_mesplq t_qdp t_spectrum t_foo

#
# The program and its dependencies
t_foo_SOURCES = t_foo.cc $(HDRS)
t_io_SOURCES = t_io.cc $(HDRS)
t_dslashm_SOURCES = t_dslashm.cc dslashm_w.cc $(HDRS)
t_mesplq_SOURCES = t_mesplq.cc mesplq.cc reunit.cc $(HDRS)
t_formfac_SOURCES = t_formfac.cc formfac_w.cc $(HDRS)
t_spectrum_SOURCES = t_spectrum.cc formfac_w.cc dslashm_w.cc \
          baryon_w.cc mesons_w.cc mesplq.cc reunit.cc $(HDRS)
t_qdp_SOURCES =  t_qdp.cc formfac_w.cc dslashm_w.cc baryon_w.cc \
	 mesons_w.cc mesplq.cc reunit.cc $(HDRS)

# build lib is a target that goes tot he build dir of the library and 
# does a make to make sure all those dependencies are OK. In order
# for it to be done every time, we have to make it a 'phony' target
.PHONY: build_lib
build_lib:
	cd @top_builddir@/lib ; $(MAKE)








