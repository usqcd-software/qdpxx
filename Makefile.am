#
# Recurse down into lib (to build the library)
#                   examples (build examples)
#                   includes (to install includes)
#
SUBDIRS = lib examples include

#
# User explicitly has to make docs -- you may not want them
#
doc:
	cd docs ; $(MAKE)

doc-install: 
	cd docs ; $(MAKE) install

doc-uninstall:
	cd docs ; $(MAKE) uninstall

doc-clean:
	cd docs ; $(MAKE) clean

doc-distclean:
	cd docs ; $(MAKE) distclean

#
# Some things that we want in the distribution tarball 
#

#
# All the sources that are somehow arch-specific
# All the headers get into the dist by default
# 
ARCH_SOURCES = lib/scalar_specific.cc \
               lib/parscalar_specific.cc

EXTRA_DIST= $(ARCH_SOURCES) PETE/QDPClasses.in PETE/QDPOps.in  PETE/Tools  examples/BuildingBlocks docs
