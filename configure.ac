AC_INIT(qdp++, 0.18, edwards@jlab.org)

#
# Options that may or may not be required. There is probably a 
# Nice autoconf-y way of doing this but this is just me being 
# Dumb
#
# At the outset none of them are required
#
ac_required_mpich_home=0;
ac_required_gm_home=0;
ac_required_qmp=0;


############################################################
#
# --with-xxxx and --enable-xxxx switches 
#
############################################################

#
# --enable-parallel-arch argument
#
AC_ARG_ENABLE(parallel-arch,
	      AC_HELP_STRING([--enable-parallel-arch=<arch>],
 	        [ Build QDP++ for parallel architecture <arch> ]),
	      ac_parallel_arch=$enableval, 
	      ac_parallel_arch=scalar)

#
# --with-qmp-home argument
AC_ARG_WITH(qmp-home,
	    AC_HELP_STRING([--with-qmp-home=<dir> ], 
	       [ Specify the QMP installation directory is <dir> ] ),
	    ac_qmp_home=$withval)
	    
#
# --with-qmp-comms-type argument 
#
AC_ARG_WITH(qmp-comms-type, 
            AC_HELP_STRING([--with-qmp-comms-type=<type> ],
	      [ Build QMP over MPICH for CH4 target ]),
	    ac_qmp_comms_type=$withval)


#
# --with-mpich-home Argument
#
AC_ARG_WITH(mpich-home,
	    AC_HELP_STRING([--with-mpich-home],
	      [ Location of the MPICH installation ]),
	    ac_mpich_home=$withval)

#
# --with-gm-home argument
#
AC_ARG_WITH(gm-home,
	    AC_HELP_STRING([--with-gm-home], 
	      [ Location of the GMP installation ]),
	    ac_gm_home=$withval)

#
# --enable-qmp-cfg
#
# this sets the value in the Makefile.
#
#
AC_ARG_ENABLE(qmp-cfg,
	     AC_HELP_STRING([--enable-qmp-cfg=cfgvalue],
               [ Sets Makefile option CFG, for internal debugging. (default is RELEASE)]),
            ac_qmp_cfg=$enableval, ac_qmp_cfg=RELEASE)


#############################################################
# Check the parallel architecture
#############################################################
case $ac_parallel_arch in 
  scalar) 
    AC_MSG_NOTICE([ Building QDP++ for scalar architecture])
    ac_required_qmp=0
    ;;
  parscalar)
    AC_MSG_NOTICE([ Building QDP++ for parscalar architecture ])
    ac_required_qmp=1
    ;;
  *)
    AC_MSG_ERROR([ Unsupported Parallel Architecture ])
    ;;
esac

############################################################
# If QMP is required make sure all the relevant required
# stuff is also given
############################################################

if test ${ac_required_qmp} -eq 1 ;
then
  if test "X${ac_qmp_comms_type}X" = "XX";
  then 
    AC_MSG_ERROR([ You need to specify --with-qmp-comms-type to build over QMP ] )
  else 
    case ${ac_qmp_comms_type} in
      mpich)
	# Generic MPICH target -- need mpich-home 
	ac_required_mpich_home=1
	ac_required_gm_home=0
	;;
      mpich-gm)
        # MPICH-GM target -- need mpich-home and gm-home
	ac_required_mpich_home=1
	ac_required_gm_home=1
	;;
      gm)
	# Pure GM target -- need gm-home
	ac_required_mpich_home=0
	ac_required_gm_home=1
	;;
      *)
	# Unsupported target.
	AC_MSG_NOTICE([Unknown value for --with-comms-type])
	AC_MSG_NOTICE([ Allowed values are: ])
	AC_MSG_NOTICE([    mpich      -- Compile over MPICH-CH4 ])
	AC_MSG_NOTICE([    mpich-gm   -- Compile over MPICH-GM  ])
	AC_MSG_NOTICE([    gm         -- Compile over GM ])
	AC_MSG_ERROR([ Please rerun configure specifying --with-comms-type])
	;;
    esac

    ############################################################
    #
    # Check that the required directories have been specified
    #
    ############################################################
 
    if test "X${ac_qmp_home}X" = "XX";
    then
      AC_MSG_ERROR([ You need to specify --with-qmp-home for this build ])
    fi

    if test "X${ac_qmp_home}X" = "XyesX";
    then 
      AC_MSG_ERROR([ You need to specify a value for --with-qmp-home for this build] )
    fi

    #
    # Check that MPICH-Home is specified 	
    if test ${ac_required_mpich_home} -eq 1 ;
    then
      if test "X${ac_mpich_home}X" = "XX" ; 
      then
        AC_MSG_ERROR([ You need to specify --with-mpich-home for this build ]) 
      fi

      if test "X${ac_mpich_home}X" = "XyesX" ; 
      then 
        AC_MSG_ERROR([ you need to specify a  value for --with-mpich-home ])
      fi
    fi

    #
    # Check that GM-Home is specified 
    if test ${ac_required_gm_home} -eq 1 ;
    then
      if test "X${ac_gm_home}X" = "XX" ;
      then
        AC_MSG_ERROR([ You need to specify --with-gm-home for this build ] )
      fi
	   
      if test "X${ac_gh_home}X" = "XyesX" ; 
      then 
        AC_MSG_ERROR([ You need to specify a value fo --with-gm-home ])
      fi 
    fi
  fi 
fi
	
############################################################
# Check that qmp_cfg is reasonable
############################################################
case ${ac_qmp_cfg} in
  RELEASE)
    ;;
  DEBUG)
    ;;
  DEV)
    ;;
  MEMORY-CHECK)
    ;;
  *)
    AC_MSG_ERROR([ Shouldnt have got here]) 
    ;;
esac

##############################################################
##############################################################
#
# Now have all the options... Do some configuration 
#
##############################################################
##############################################################

## Hunt for a C compiler -- favouring gcc
AC_PROG_CC(gcc cc)

## Hunt for a C++ compiler -- favouring g++
AC_PROG_CXX(g++ cxx CC)

LIBNAME=qdp

SRCS="subset.cc random.cc qdp.cc layout.cc io.cc iogauge.cc byteorder.cc\
  util.cc readszin.cc"
HDRS="qdp.h word.h inner.h reality.h outer.h qdptype.h qdpexpr.h \
  defs.h specializations.h subset.h params.h multi.h random.h layout.h \
  proto.h io.h QDPOperators.h globalfuncs.h primitive.h primscalar.h \
  primmatrix.h primvector.h primseed.h primcolormat.h \
  primcolorvec.h primgamma.h primspinmat.h primspinvec.h"

CXXFLAGS=${CXXFLAGS}" -Wall -g"
CXXFLAGS=${CXXFLAGS}" -I\$(TOPDIR) -I\$(TOPDIR)/include"

case ${ac_parallel_arch} in
  scalar) 
    AC_DEFINE(ARCH_SCALAR)
    SRCS=${SRCS}" scalar_specific.cc"
    HDRS=${HDRS}" scalar_specific.h"
    ;;
  parscalar)
    AC_DEFINE(ARCH_PARSCALAR)
    SRCS=${SRCS}" parscalar_specific.cc"
    HDRS=${HDRS}" parscalar_specific.h"
    CXXFLAGS=${CXXFLAGS}" -I${ac_qmp_home}/include"
    LDFLAGS=${LDFLAGS}" -L${ac_qmp_home}/lib"
    LIBS=${LIBS}" -lqmp"

    case ${ac_qmp_comms_type} in
      mpich)
        LDFLAGS=${LDFLAGS}" -L${ac_mpich_home}/lib"
        LIBS=${LIBS}" -lpmpich -lmpich -lpmpich"
        ;;
      mpich-gm)
        LDFLAGS=${LDFLAGS}" -L${ac_mpich_home}/lib"
        LDFLAGS=${LDFLAGS}" -L${ac_gm_home}/binary/lib"
        LIBS=${LIBS}" -lmpich -lgm"
        ;;
      gm) 
        LDFALGS=${LDFLAGS}" -L${ac_gm_home}/binary/lib"
        LIBS=${LIBS}" -lgm"
        ;;
      *)
        AC_MSG_ERROR([ Shouldnt be here ])
	;;
     esac
     ;;
  *)
    AC_MSG_ERROR([ Shouldnt be here 2 ] );
    ;;
esac

LDFLAGS="-L\$(TOPDIR)/lib "${LDFLAGS}
LIBS="-l${LIBNAME} "${LIBS}

MY_VPATH="\$(TOPDIR)/include \
          \$(TOPDIR)/src \
	  \$(TOPDIR)/lib \
	  \$(TOPDIR)"

AC_SUBST(LIBNAME)
AC_SUBST(SRCS)
AC_SUBST(HDRS)
AC_SUBST(MY_VPATH)

AC_CONFIG_FILES(Makefile)
AC_CONFIG_FILES(docs/Makefile)
AC_CONFIG_FILES(include/Makefile)
AC_CONFIG_FILES(lib/Makefile)
AC_CONFIG_FILES(tests/Makefile)
AC_CONFIG_FILES(tests/BuildingBlocks/Makefile)

AC_OUTPUT
