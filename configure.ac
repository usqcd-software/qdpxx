AC_INIT(qdp,0.32-rc1,edwards@jlab.org)

AC_CONFIG_AUX_DIR(config)

dnl Initialise automake
AM_INIT_AUTOMAKE
dnl
dnl
dnl --with-xxxx and --enable-xxxx switches 
dnl

dnl --enable-Nd
AC_ARG_ENABLE(Nd,
  AC_HELP_STRING([--enable-Nd=N ],
    [Set No of Spacetime Dimensions, default is Nd=4 ]),
  AC_SUBST(ac_Nd, [$enableval]),
  AC_SUBST(ac_Nd, 4)
)

dnl Tell the user about Nd
AC_MSG_NOTICE([Configuring QDP++ for Nd = ${ac_Nd}])


dnl --enable-Nc
AC_ARG_ENABLE(Nc,
  AC_HELP_STRING([--enable-Nc=N ],
    [Set Number of Colours, ie N for SU(N) default is SU(3)]),
  AC_SUBST(ac_Nc, [$enableval]),
  AC_SUBST(ac_Nc, 3)
)

dnl Tell the user about Nc
AC_MSG_NOTICE([Configuring QDP++ for Nc = ${ac_Nc}])


dnl --enable-Ns
AC_ARG_ENABLE(Ns,
  AC_HELP_STRING([--enable-Ns=N],
    [Set No of Spin Components, Default is Ns=4]),
  AC_SUBST(ac_Ns, [$enableval]),
  AC_SUBST(ac_Ns, 4)
)

dnl Tell the user about Ns
AC_MSG_NOTICE([Configuring QDP++ for Ns = ${ac_Ns}])


dnl --enable-parallel-arch argument
AC_ARG_ENABLE(parallel-arch,
  AC_HELP_STRING([--enable-parallel-arch=<arch>],
    [Build QDP++ for parallel architecture <arch>]),
  AC_SUBST(PARALLEL_ARCH, [$enableval]),
  AC_SUBST(PARALLEL_ARCH, [scalar])
)


dnl George T. Fleming 03/03/2003
dnl This is a total reworking of Balint Joo's implementation
dnl of the qmp_build_env.sh script.  First of all, since the plan seems
dnl to have changed from having QMP live in a subdirectory of QDP++
dnl to having QMP configured, built and installed separately, we only
dnl need to know the flags necessary to compile and link user apps
dnl against QMP and not the flags necessary to compile QMP itself.
dnl
dnl Also, QMP is not needed for ac_parallel_arch=scalar, so don't bother
dnl looking for it in that case.
AC_ARG_WITH(qmp,
  AC_HELP_STRING(
    [--with-qmp=DIR],
    [Build QDP++ on top of QMP, where QMP is installed in DIR.]
  ),
  [QMP_HOME="${with_qmp}"]
)

case ${PARALLEL_ARCH} in
  scalar)
    AC_SUBST(ARCHDEF_SUBSTITUTION, ARCH_SCALAR)
    AC_MSG_NOTICE([ Scalar Build! Not checking for QMP ])
    AC_SUBST(QMP_CFLAGS)
    AC_SUBST(QMP_LDFLAGS)
    AC_SUBST(QMP_LIBS)
    ;;
  parscalar)
    AC_SUBST(ARCHDEF_SUBSTITUTION, ARCH_PARSCALAR)
    AC_MSG_NOTICE([ Parscalar build! Checking for QMP ])

    if test "X${QMP_HOME}X" = "XX" ; then
      AC_PATH_PROG([QMP_CONFIG], [qmp-config], [])
    else
      AC_PATH_PROG([QMP_CONFIG], [qmp-config], [], [${QMP_HOME}/bin:${PATH}])
    fi

    if test "X${QMP_CONFIG}X" != "XX" ; then
      AC_MSG_NOTICE([Found QMP configuration program ${QMP_CONFIG}])
      AC_SUBST(QMP_CFLAGS,  "`${QMP_CONFIG} --cflags`")
      AC_MSG_NOTICE([QMP compile flags: ${QMP_CFLAGS}])
      AC_SUBST(QMP_LDFLAGS, "`${QMP_CONFIG} --ldflags`")
      AC_MSG_NOTICE([QMP linking flags: ${QMP_LDFLAGS}])
      AC_SUBST(QMP_LIBS,    "`${QMP_CONFIG} --libs`")
      AC_MSG_NOTICE([QMP libraries flags: ${QMP_LIBS}])
    else 
      AC_MSG_WARN([QMP configuration program qmp-config not found.])
      AC_MSG_WARN([Set environment variables QMP_CFLAGS QMP_LDFAGS QMP_LIBS
        before configure])
    fi
    ;;
  *)
    AC_MSG_ERROR([ Unknown value for --enable-parallel-arch ])
    ;;
esac

dnl verify we can compile and link against QMP, if needed
PAC_QMP_LINK_CXX_FUNC(
  [${QMP_CFLAGS}],
  [${QMP_LDFLAGS}],
  [${QMP_LIBS}],
  ,
  ,
  [qmp_link_ok=yes],
  [qmp_link_ok=no]
)

AC_MSG_CHECKING([if we can compile/link of a simple QMP C++ program])
case ${PARALLEL_ARCH} in
  parscalar)
    if test "X${qmp_link_ok}X" = "XyesX" ; then
      AC_MSG_RESULT(yes)
    else
      AC_MSG_RESULT(no)
      AC_MSG_ERROR([Cannot compile/link a basic QMP C++ program!
        Check QMP_CFLAGS, QMP_LDFLAGS, QMP_LIBS.])
    fi
    ;;
  *)
    AC_MSG_RESULT(ignored)
    ;;
esac

dnl Support for memory debugging with DMALLOC
AM_WITH_DMALLOC

AC_LANG_CPLUSPLUS
AC_PROG_CXX(g++ cxx CC)
AC_PROG_RANLIB

dnl
dnl
dnl Now have all the options... Do some configuration 
dnl
dnl

## Depending on what Parallel Architecture is, do some configuring
dnl GTF:
dnl   This way of assigning sources to variables for variable substition
dnl   in automake will totally break 'make dist' since automake won't
dnl   know to include the sources for all possible arches.  Correct way
dnl   is to handle it as a conditional in Makefile.am
dnl
dnl case ${PARALLEL_ARCH} in
dnl   scalar) 
dnl     ARCH_SRCS="scalar_init.cc scalar_layout.cc scalar_specific.cc"
dnl     ARCH_HDRS="scalar_specific.h"
dnl     AC_SUBST(ARCH_SRCS)
dnl     AC_SUBST(ARCH_HDRS)
dnl     ;;
dnl   parscalar)
dnl     ARCH_SRCS="parscalar_init.cc parscalar_layout.cc parscalar_specific.cc"
dnl     ARCH_HDRDS=" parscalar_specific.h"
dnl     AC_SUBST(ARCH_SRCS)
dnl     AC_SUBST(ARCH_HDRS)
dnl     ;;
dnl esac

AM_CONDITIONAL(ARCH_SCALAR,    [test "X${PARALLEL_ARCH}X" = "XscalarX"])
AM_CONDITIONAL(ARCH_PARSCALAR, [test "X${PARALLEL_ARCH}X" = "XparscalarX"])

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
AC_CONFIG_FILES(qdp++-config)
AC_CONFIG_FILES(lib/Makefile)
AC_CONFIG_FILES(include/Makefile)
AC_CONFIG_FILES(include/PETE/Makefile)
AC_CONFIG_FILES(include/qdp_config.h)
AC_CONFIG_FILES(docs/Makefile)
AC_CONFIG_FILES(examples/Makefile)
#
# Finish
#
AC_OUTPUT

