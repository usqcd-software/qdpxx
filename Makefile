# $Id: Makefile,v 1.8 2002-10-28 03:08:44 edwards Exp $
#
# Makefile for C++ QDP code
#

include ./Makefile.cfg

PETE = ./PETE/Tools
#CXXINC = -I./PETE
CXXFLAGS += -I./PETE
MAKEEXPR = $(PETE)/MakeOperators

libname = qdp.a

sources = subset.cc random.cc qdp.cc layout.cc io.cc iogauge.cc byteorder.cc
headers = qdp.h word.h inner.h reality.h outer.h qdptype.h qdpexpr.h \
	defs.h specializations.h subset.h params.h multi.h random.h layout.h proto.h \
	io.h QDPOperators.h globalfuncs.h \
	primitive.h primscalar.h primmatrix.h primvector.h primseed.h primcolormat.h \
	primcolorvec.h primgamma.h primspinmat.h primspinvec.h

ifeq ($(ARCH),SCALAR)
sources += scalar_specific.cc
headers += scalar_specific.h 
else
ifeq ($(ARCH),PARSCALAR)
CXXFLAGS += -I../qmp
CXXLIBS += -L../qmp -lqmp

sources += parscalar_specific.cc
headers += parscalar_specific.h 
endif
endif

obj :=  $(sources:%.cc=%.o)

.PHONY: all clean

all: $(libname)

foo: foo.o $(libname) $(headers)
	$(CXX) $(CXXINC) $(CXXFLAGS) $< -o $@  $(libname) $(CXXLIBS)

QDPOperators.h: QDPClasses.in QDPOps.in
	$(MAKEEXPR) --classes $< --operators QDPOps.in --pete-ops --op-tags --guard QDPOPS_H > $@

$(libname): $(libname)($(obj)) $(headers)

%.o : %.cc $(headers) 
	$(CXX) $(CXXINC) $(CXXFLAGS) -c $< -o $@

.SUFFIXES:
.SUFFIXES: .h .cc .o

clean::
	$(RM) $(obj) $(libname) foo.o foo.exe foo


