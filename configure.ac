AC_INIT(qdp,1.5,edwards@jlab.org)

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

dnl --enable-sse
AC_ARG_ENABLE(sse,
  AC_HELP_STRING([--enable-sse],
    [Optimize code with Intel SSE instructions]),
  AC_SUBST(ac_sse,1),
  AC_SUBST(ac_sse,0)
)

dnl --enable-sse2
AC_ARG_ENABLE(sse2,
  AC_HELP_STRING([--enable-sse2],
    [Optimize code with Intel SSE2 instructions]),
  [AC_SUBST(ac_sse,1)
   AC_SUBST(ac_sse2,1)],
  AC_SUBST(ac_sse2,0)
)


dnl Tell the user about SSE
if test "X${ac_sse}X" = "X1X" ; then
  AC_MSG_NOTICE([Configuring QDP++ to use Streaming SIMD Extensions (SSE)])
fi

dnl Tell the user about SSE2
if test "X${ac_sse2}X" = "X1X" ; then
  AC_MSG_NOTICE([Configuring QDP++ to use Streaming SIMD Extensions (SSE2)])
fi


AC_ARG_ENABLE(precision, 
  AC_HELP_STRING( [ --enable precision=single|double ],
                  [ Sets the base precision to be single or double precision (default is single) ] ),
  [ ac_precision=${enableval} ],
  [ ac_precision="single" ] )

case ${ac_precision} in 
single)
	AC_MSG_NOTICE( [Configuring for single precision ] )
	AC_SUBST(ac_header_precision, [SINGLE_PRECISION] )
	;;
double)
	AC_MSG_NOTICE( [Configuring for double precision ] )
	AC_SUBST(ac_header_precision, [DOUBLE_PRECISION] )
	;;
*)
	AC_MSG_ERROR( [Unsupported value of precision. Check --enable-precision ] )
	;;
esac

dnl George T. Fleming 03/03/2003
dnl This is a total reworking of Balint Joo's implementation
dnl of the qmp_build_env.sh script.  First of all, since the plan seems
dnl to have changed from having QMP live in a subdirectory of QDP++
dnl to having QMP configured, built and installed separately, we only
dnl need to know the flags necessary to compile and link user apps
dnl against QMP and not the flags necessary to compile QMP itself.
dnl
dnl Also, QMP is not needed for ac_parallel_arch={scalar,scalarvec}, so don't bother
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
    AC_SUBST(ARCH_SITE, "scalarsite")
    AC_MSG_NOTICE([ Scalar Build! Not checking for QMP ])
    AC_SUBST(QMP_CFLAGS)
    AC_SUBST(QMP_LDFLAGS)
    AC_SUBST(QMP_LIBS)
    ;;
  scalarvec)
    AC_SUBST(ARCHDEF_SUBSTITUTION, ARCH_SCALARVEC)
    AC_SUBST(ARCH_SITE, "scalarvecsite")
    AC_MSG_NOTICE([ Scalar with Vector extensions Build! Not checking for QMP ])
    AC_SUBST(QMP_CFLAGS)
    AC_SUBST(QMP_LDFLAGS)
    AC_SUBST(QMP_LIBS)
    ;;
  parscalar)
    AC_SUBST(ARCHDEF_SUBSTITUTION, ARCH_PARSCALAR)
    AC_SUBST(ARCH_SITE, "scalarsite")
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
  parscalarvec)
    AC_SUBST(ARCHDEF_SUBSTITUTION, ARCH_PARSCALARVEC)
    AC_SUBST(ARCH_SITE, "scalarvecsite")
    AC_MSG_NOTICE([ Parscalar with Vector extensions build! Checking for QMP ])

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
  parscalarvec)
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


dnl
dnl
dnl now libxml2
dnl
dnl
AC_ARG_WITH(libxml2,
  AC_HELP_STRING(
    [--with-libxml2=DIR],
    [Build on top of libxml2 where libxml2 is installed in DIR]
  ),
  [LIBXML2_HOME="$with_libxml2"]
)

echo ${LIBXML2_HOME}

if test "X${LIBXML2_HOME}X" = "XX" ; then
  AC_PATH_PROG(LIBXML2_CONFIG, [xml2-config], [])
else
  AC_PATH_PROG(LIBXML2_CONFIG, [xml2-config], [], [${LIBXML2_HOME}/bin:${PATH}])
fi

if test "X${LIBXML2_CONFIG}X" = "XX" ; then
  AC_MSG_ERROR([libxml2 configuration program xml2-config not found.])
fi

AC_MSG_NOTICE([Found libxml2 configuration program ${LIBXML2XX_CONFIG}])
AC_SUBST(LIBXML2_CXXFLAGS, "`${LIBXML2_CONFIG} --cflags`")
AC_MSG_NOTICE([libxml2 compile flags: ${LIBXML2_CXXFLAGS}])
AC_SUBST(LIBXML2_LIBS,     "`${LIBXML2_CONFIG} --libs`")
AC_MSG_NOTICE([libxml2 libraries flags: ${LIBXML2_LIBS}])

dnl This function is defined in acinclude.m4
dnl it tries to try to compile the program 
AC_MSG_CHECKING([if we can compile/link a simple libxml2 program])
PAC_LIBXML2_LINK_CXX_FUNC(
  ${LIBXML2_CXXFLAGS},
  ${LIBXML2_LIBS},
  ,
  ,
  [libxml2_link_ok=yes],
  [libxml2_link_ok=no]
)

if test "X${libxml2_link_ok}X" = "XyesX" ; then 
  AC_MSG_RESULT(yes)
else 
  AC_MSG_RESULT(no)
  AC_MSG_ERROR([Cannot compile/link a program with libxml2.
    Use --with-libxml2=<dir> to select a working version.])
fi 



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

AM_CONDITIONAL(ARCH_SCALAR,       [test "X${PARALLEL_ARCH}X" = "XscalarX"])
AM_CONDITIONAL(ARCH_PARSCALAR,    [test "X${PARALLEL_ARCH}X" = "XparscalarX"])
AM_CONDITIONAL(ARCH_SCALARVEC,    [test "X${PARALLEL_ARCH}X" = "XscalarvecX"])
AM_CONDITIONAL(ARCH_PARSCALARVEC, [test "X${PARALLEL_ARCH}X" = "XparscalarvecX"])

AM_CONDITIONAL(QDP_USE_SCALAR_SSE, [test "X${ac_sse}X" = "X1X" -a "X${ARCH_SITE}X" = "XscalarsiteX"])
AM_CONDITIONAL(QDP_USE_SCALAR_SSE2, [test "X${ac_sse2}X" = "X1X" -a "X${ARCH_SITE}X" = "XscalarsiteX"])

AM_CONDITIONAL(QDP_USE_SCALARVEC_SSE, [test "X${ac_sse}X" = "X1X" -a "X${ARCH_SITE}X" = "XscalarvecsiteX"])
AM_CONDITIONAL(QDP_USE_SCALARVEC_SSE2, [test "X${ac_sse2}X" = "X1X" -a "X${ARCH_SITE}X" = "XscalarvecsiteX"])

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
# Configure subdir packages
#
AC_CONFIG_SUBDIRS(other_libs/nml)
AC_CONFIG_SUBDIRS(other_libs/qio)
AC_CONFIG_SUBDIRS(other_libs/xpath_reader)

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
AC_CONFIG_FILES(other_libs/Makefile)
AC_CONFIG_FILES(include/Makefile)
AC_CONFIG_FILES(include/qdp_config.h)
AC_CONFIG_FILES(docs/Makefile)
AC_CONFIG_FILES(examples/Makefile)
#
# Finish
#
AC_OUTPUT

