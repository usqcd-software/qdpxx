# $Id: Makefile,v 1.4 2002-10-01 16:24:41 edwards Exp $
#
# Makefile for C++ QDP code
#

CXX = g++
CXXFLAGS = -g -Wno-deprecated 
#CXXFLAGS = -O3
#CXXFLAGS = -O4 -fomit-frame-pointer -felide-constructors
#CXXFLAGS = -O4 -fomit-frame-pointer
#CXXFLAGS =  -Wno-deprecated -ftemplate-depth-80 -O3 -fomit-frame-pointer -ffast-math \
#	-funsafe-math-optimizations -Winline -felide-constructors -fargument-noalias-global \
#	-msse -fprefetch-loop-arrays -finline-limit=2000 

# -finline-functions 

PETE = ./PETE/Tools
#CXXINC = -I./PETE
CXXFLAGS += -I./PETE
MAKEEXPR = $(PETE)/MakeOperators

libname = qdp.a

sources = subset.cc random.cc qdp.cc layout.cc io.cc
headers = qdp.h word.h inner.h reality.h outer.h qdptype.h qdpexpr.h \
	defs.h specializations.h subset.h params.h multi.h random.h layout.h proto.h \
	scalar_specific.h io.h QDPOperators.h globalfuncs.h \
	primitive.h primscalar.h primmatrix.h primvector.h primseed.h primcolormat.h \
	primcolorvec.h primgamma.h primspinmat.h primspinvec.h
obj :=  $(sources:%.cc=%.o)

.PHONY: all clean

all: $(libname)

foo: foo.o $(libname) $(headers)
	$(CXX) $(CXXINC) $(CXXFLAGS) $< -o $@  $(libname)

QDPOperators.h: QDPClasses.in QDPOps.in
	$(MAKEEXPR) --classes $< --operators QDPOps.in --pete-ops --op-tags --guard QDPOPS_H > $@

$(libname): $(libname)($(obj)) $(headers)

%.o : %.cc $(headers) 
	$(CXX) $(CXXINC) $(CXXFLAGS) -c $< -o $@

.SUFFIXES:
.SUFFIXES: .h .cc .o

clean::
	$(RM) $(obj) $(libname) foo.o foo.exe foo


