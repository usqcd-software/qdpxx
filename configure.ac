AC_INIT(qdp,0.31,edwards@jlab.org)

AC_CONFIG_AUX_DIR(config)

dnl Initialise automake
AM_INIT_AUTOMAKE
dnl
dnl
dnl --with-xxxx and --enable-xxxx switches 
dnl

dnl --enable-Nc
AC_ARG_ENABLE(Nc,
              AC_HELP_STRING([--enable-Nc=N ],
		[ Set Number of Colours, ie N for SU(N) default is SU(3) ]),
              ac_Nc=$enableval,
              ac_Nc=3)

dnl --enable-Nd
AC_ARG_ENABLE(Nd,
              AC_HELP_STRING([--enable-Nd=N ],
                [ Set No of Spacetime Dimensions, default is Nd=4 ]),
              ac_Nd=$enableval,
              ac_Nd=4)


dnl --enable-Ns
AC_ARG_ENABLE(Ns,
              AC_HELP_STRING([--enable-Ns=<No of Spin Components],
                [ Set No of Spin Components, Default is Ns=4 ]),
              ac_Ns=$enableval,
              ac_Ns=4)


dnl --enable-parallel-arch argument
AC_ARG_ENABLE(parallel-arch,
	      AC_HELP_STRING([--enable-parallel-arch=<arch>],
 	        [ Build QDP++ for parallel architecture <arch> ]),
	      ac_parallel_arch=$enableval, 
	      ac_parallel_arch=scalar)


dnl Tell the user about Nd
AC_MSG_NOTICE([ Configuring QDP++ for Nd = ${ac_Nd} ])

dnl Tell the user about Nc
AC_MSG_NOTICE([ Configuring QDP++ for Nc = ${ac_Nc} ])

dnl Tell the user about Ns
AC_MSG_NOTICE([ Configuring QDP for Ns = ${ac_Ns} ])

dnl Balint Joo 13/12/2002 
dnl Options for finding out where MPI/GM/QMP is
dnl folowing George Flemings QMP mods

dnl Balint Joo 3/1/2003
dnl Option for trying to get the config info from QMP compilation
AC_ARG_WITH(qmp,
  AC_HELP_STRING([--with-qmp=DIR],
    [Build QDP++ on top of QMP, where QMP is installed in DIR]),
    [ QMP_HOME="$with_qmp"
      AC_MSG_CHECKING([if $QMP_HOME/config/qmp_build_env.sh exists])
      if test -f $QMP_HOME/config/qmp_build_env.sh ;
      then
	AC_MSG_RESULT([yes])
	AC_MSG_NOTICE([ Sourcing qmp_build_env.sh in $QMP_HOME/config ])
	source $QMP_HOME/config/qmp_build_env.sh
      else 
	AC_MSG_RESULT([no])
	AC_MSG_NOTICE([ configuration may still work if you set the relevant ])
	AC_MSG_NOTICE([ environment variables elsewhere ])
      fi]
)

AC_ARG_WITH(qmp-comms-type,
  AC_HELP_STRING([--with-qmp-comms-type=TYPE],
    [Build QMP on top of MPI or GM.  If MPI over GM is desired, choose MPI.
      Default is MPI.]),
  [case "${with_qmp_comms_type}" in
    MPI) ;;
    mpi) with_qmp_comms_type=MPI ;;
    GM)  ;;
    gm) with_qmp_comms_type=GM ;;
    *) AC_MSG_ERROR([bad value "${with_qmp_comms_type}"
      for --with-qmp-comms-type]) ;;
    esac],
  [with_qmp_comms_type=MPI])

AC_ARG_WITH( qmp-comms-cxxflags,
  AC_HELP_STRING([--with-qmp-comms-cxxflags=QMP_COMMS_CFLAGS],
    [To pass optional include path to comms header files via compiler flag
      -I]),
  [QMP_COMMS_CXXFLAGS="$with_qmp_comms_cxxflags"] )

AC_SUBST(QMP_COMMS_CXXFLAGS)

AC_ARG_WITH( qmp-comms-ldflags,
  AC_HELP_STRING([--with-qmp-comms-ldflags=QMP_COMMS_LDFLAGS],
    [To pass optional library search path to comms libraries via compiler
      flag -L]),
  [QMP_COMMS_LDFLAGS="$with_qmp_comms_ldflags"])

AC_SUBST(QMP_COMMS_LDFLAGS)

AC_ARG_WITH( qmp-comms-libs,
  AC_HELP_STRING([--with-qmp-comms-libs=QMP_COMMS_LIBS],
    [To pass optional list of comms libraries to linker via compiler flag -l]),
  [QMP_COMMS_LIBS="$with_qmp_comms_libs"])

AC_SUBST(QMP_COMMS_LIBS)


dnl Support for memory debugging with DMALLOC
AM_WITH_DMALLOC

dnl now the qmp-flags
AC_ARG_WITH( qmp-cxxflags,
  AC_HELP_STRING([--with-qmp-cxxflags=QMP_CXXFLAGS],
    [To pass optional include path to QMP header files via compiler flag
      -I]),
  [QMP_CXXFLAGS="$with_qmp_cxxflags"] )
AC_SUBST(QMP_CXXFLAGS)

AC_ARG_WITH( qmp-ldflags, 
  AC_HELP_STRING([--with-qmp-ldflags=QMP_LDFLAGS],
    [To pass optional link path to QMP libraries via compiler flag -L]),
    [QMP_LDFLAGS="$with_qmp_ldflags"])

AC_SUBST(QMP_LDFLAGS)

AC_ARG_WITH( qmp-libs,
  AC_HELP_STRING([-with-qmp-libs=QMP_LIBS],
    [To pass optional libraries to QMP libraries via compiler flag -l]),
    [QMP_LIBS="$with_qmp_libs"])

AC_SUBST(QMP_LIBS)

AC_LANG_CPLUSPLUS
AC_PROG_CXX(g++ cxx CC)
AC_PROG_RANLIB

case ${ac_parallel_arch} in
  scalar)
    AC_MSG_NOTICE([ Scalar Build! Not checking for QMP ])
    ;;
  parscalar)
    AC_MSG_NOTICE([ Parscalar build! Checking for QMP ])

#   This check is redundant, the QMP_LINK_CHECK Below is sufficient
#   Balint Joo, 03/01/2003
#
#    dnl check comms harness and qmp builds depending on with_qmp_comms_type
#    case ${with_qmp_comms_type} in 
#      MPI) 
#        dnl check MPI
#	AC_MSG_NOTICE([ QMP Comms harness is MPI ])
#        PAC_MPI_LINK_CXX_FUNC( ${QMP_COMMS_CXXFLAGS}, ${QMP_COMMS_LDFLAGS}, dnl
#          ${QMP_COMMS_LIBS}, , , [mpi_link_ok=yes], [mpi_link_ok=no])
#        AC_MSG_CHECKING([if we can compile/link of a simple MPI program])
#        if test "X${mpi_link_ok}X" = "XyesX" ; then
#          AC_MSG_RESULT(yes)
#        else
#          AC_MSG_RESULT(no)
#          AC_MSG_ERROR([Cannot compile/link a basic MPI C++ program!
#          Check QMP_COMMS_CXXFLAGS, QMP_COMMS_LDFLAGS, QMP_COMMS_LIBS.])
#        fi
#     	;;
#      GM)
#	AC_MSG_NOTICE([ QMP Comms harness is GM ])
#        ;;
#      *)
#        AC_MSG_ERROR([ Unsupported QMP Comms harness ])
#        ;;
#    esac
 
   dnl now check QMP works
    PAC_QMP_LINK_CXX_FUNC( ${QMP_CXXFLAGS}, ${QMP_LDFLAGS}, ${QMP_LIBS}, dnl
                       ${QMP_COMMS_CXXFLAGS}, ${QMP_COMMS_LDFLAGS}, dnl
                       ${QMP_COMMS_LIBS}, , , [qmp_link_ok=yes], dnl
	               [qmp_link_ok=no])
    AC_MSG_CHECKING([if we can compile/link of a simple QMP program])
    if test "X${qmp_link_ok}X" = "XyesX" ; then
      AC_MSG_RESULT(yes)
    else
      AC_MSG_RESULT(no)
      AC_MSG_ERROR([Cannot compile/link a basic QMP C++ program!
      Check QMP_CXXFLAGS, QMP_LDFLAGS, QMP_LIBS, QMP_COMMS_CXXFLAGS, QMP_COMMS_LDFLAGS, QMP_COMMS_LIBS.])
    fi
    ;;
  *)
    AC_MSG_ERROR([ Unknown value for --enable-parallel-arch ])
    ;;
esac

dnl
dnl
dnl Now have all the options... Do some configuration 
dnl
dnl


## Depending on what Parallel Architecture is, do some configuring
case ${ac_parallel_arch} in
  scalar) 
    ARCHDEF_SUBSTITUTION="ARCH_SCALAR" 
    ARCH_SRCS="scalar_init.cc scalar_layout.cc scalar_specific.cc"
    ARCH_HDRS="scalar_specific.h"
    AC_SUBST(ARCH_SRCS)
    AC_SUBST(ARCH_HDRS)
    ;;
  parscalar)
    ARCHDEF_SUBSTITUTION="ARCH_PARSCALAR"
    ARCH_SRCS="parscalar_init.cc parscalar_layout.cc parscalar_specific.cc"
    ARCH_HDRDS=" parscalar_specific.h"
    AC_SUBST(ARCH_SRCS)
    AC_SUBST(ARCH_HDRS)
    ;;
esac

AC_SUBST(ARCHDEF_SUBSTITUTION)


#######################
# Define Nc, Nd and Ns
#######################

AC_SUBST(ac_Nc)
AC_SUBST(ac_Nd)
AC_SUBST(ac_Ns)


##################################
# Check for programs to build docs
##################################
AC_CHECK_PROG(LATEX, latex, latex, true)
AC_CHECK_PROG(DVIPS, dvips, dvips, true)
AC_CHECK_PROG(DVIPDF, dvipdf, dvipdf, true)
AC_CHECK_PROG(DOXYGEN, doxygen, doxygen, true)

#######################
# Produce output
#######################

#
# Set up the doxygen file
# Substitute the right directories
#
AC_CONFIG_FILES(docs/qdpdoc)

#
# Generate makefiles
#
AC_CONFIG_FILES(Makefile)
AC_CONFIG_FILES(qdp_build_env.sh)
AC_CONFIG_FILES(lib/Makefile)
AC_CONFIG_FILES(include/Makefile)
AC_CONFIG_FILES(include/PETE/Makefile)
AC_CONFIG_FILES(include/qdp_config.h)
AC_CONFIG_FILES(docs/Makefile)
AC_CONFIG_FILES(examples/Makefile)
AC_CONFIG_FILES(examples/BuildingBlocks/Makefile)
AC_CONFIG_FILES(examples/propagator/Makefile)
#
# Finish
#
AC_OUTPUT


