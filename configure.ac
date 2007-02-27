AC_INIT([qdp++],[1.22.0],[edwards@jlab.org])

AC_CONFIG_AUX_DIR(config)

AC_LANG([C++])
AC_PROG_CXX(g++ cxx CC)
AC_PROG_RANLIB
AC_CHECK_TOOL(AR, ar, [ar])

dnl Initialise automake
AM_INIT_AUTOMAKE

dnl defines...
AC_CONFIG_HEADERS([include/qdp_config_internal.h])

dnl
dnl
dnl --with-xxxx and --enable-xxxx switches 
dnl

dnl --enable-Nd
AC_ARG_ENABLE(Nd,
  AC_HELP_STRING([--enable-Nd=N ],
    [Set No of Spacetime Dimensions, default is Nd=4 ]),
  [ ac_Nd=${enableval} ],
  [ ac_Nd=4 ]
)
    
AC_DEFINE_UNQUOTED(QDP_ND, ${ac_Nd}, [Number of dimensions])        dnl config.h file
AC_SUBST(CONFIG_ND, [${ac_Nd}] )  dnl qdp++_config

dnl Tell the user about Nd
AC_MSG_NOTICE([Configuring QDP++ for Nd = ${ac_Nd}])


dnl --enable-Nc
AC_ARG_ENABLE(Nc,
  AC_HELP_STRING([--enable-Nc=N ],
    [Set Number of Colours, ie N for SU(N) default is SU(3)]),
  [ac_Nc=${enableval}],
  [ac_Nc=3]
)

AC_DEFINE_UNQUOTED(QDP_NC, ${ac_Nc}, [Number of color components])
AC_SUBST(CONFIG_NC, [${ac_Nc}] )

dnl Tell the user about Nc
AC_MSG_NOTICE([Configuring QDP++ for Nc = ${ac_Nc}])


dnl --enable-Ns
AC_ARG_ENABLE(Ns,
  AC_HELP_STRING([--enable-Ns=N],
    [Set No of Spin Components, Default is Ns=4]),

  [ac_Ns=${enableval}],
  [ac_Ns=4]
)

AC_DEFINE_UNQUOTED(QDP_NS, ${ac_Ns}, [Number of spin components])
AC_SUBST(CONFIG_NS, [${ac_Ns}] )
dnl Tell the user about Ns
AC_MSG_NOTICE([Configuring QDP++ for Ns = ${ac_Ns}])


dnl --enable-profiling
AC_ARG_ENABLE(profiling,
  AC_HELP_STRING([--enable-profiling],
    [Enable profiling. Diagnostic use only.]),
  [ac_profile=1],
  [ac_profile=0]
)

dnl --enable-parallel-arch argument
AC_ARG_ENABLE(parallel-arch,
  AC_HELP_STRING([--enable-parallel-arch=<arch>],
    [Build QDP++ for parallel architecture <arch>]),
  [PARALLEL_ARCH=${enableval}],
  AC_MSG_ERROR([Require architecture to be set])
)

dnl --enable-memory-debugging
AC_ARG_ENABLE(memory-debugging,
   AC_HELP_STRING([--enable-memory-debugging], 
     [Enable memory debugging code]),
     [ac_memdebug_enabled=1
      AC_MSG_NOTICE([Enabling memory debugging code])],
     [ac_memdebug_enabled=0]
)


dnl --enable-sse
AC_ARG_ENABLE(sse,
  AC_HELP_STRING([--enable-sse],
    [Optimize code with Intel SSE instructions]),
  [ac_sse=1],
  [ac_sse=0]
)

dnl --enable-sse2
AC_ARG_ENABLE(sse2,
  AC_HELP_STRING([--enable-sse2],
    [Optimize code with Intel SSE2 instructions]),
  [ac_sse=1
   ac_sse2=1],
  [ac_sse2=0]
)

dnl --enable-qcdoc
AC_ARG_ENABLE(qcdoc,
   AC_HELP_STRING([--enable-qcdoc], 
     [Enable QCDOC Optimisations]),
     [ac_qcdoc_enabled=1
      AC_MSG_NOTICE([Enabling QCDOC Specific Optimisations])],
     [ac_qcdoc_enabled=0]
)


if test ${ac_sse} -eq 1; then
  if test "X${GXX}X" = "XyesX"; then 
    AC_MSG_NOTICE([Enabling -msse flag for g++])
    CXXFLAGS="-msse "$CXXFLAGS
  
    AC_DEFINE_UNQUOTED(QDP_USE_SSE, ${ac_sse}, [Enable SSE instructions])
    AC_SUBST(CONFIG_SSE,[${ac_sse}])
    AC_MSG_NOTICE([Configuring QDP++ with SSE extentions enabled]);
  else 
    AC_MSG_NOTICE([WARNING: Non GNU Compiler. SSE Only supported for GCC-3.2.2 or greater due to feature creep. Sorry])
  fi
fi



if test ${ac_sse2} -eq 1; then
  if test "X${GXX}X" = "XyesX"; then 
    AC_MSG_NOTICE([Enabling -msse2 flag for g++])
    CXXFLAGS="-msse2 "$CXXFLAGS    
    AC_DEFINE_UNQUOTED(QDP_USE_SSE2, ${ac_sse2}, [Enable SSE2 instructions])
    AC_SUBST(CONFIG_SSE2,[${ac_sse2}])
    AC_MSG_NOTICE([Configuring QDP++ with SSE2 extentions enabled]);
  else 
    AC_MSG_NOTICE([WARNING: Non GNU Compiler. SSE2 Only supported for GCC-3.2.2 or greater due to feature creep. Sorry])
  fi
fi

dnl if QCDOC is enabled change some settings
if test ${ac_qcdoc_enabled} -eq 1; then 
   AC_DEFINE_UNQUOTED(QDP_USE_QCDOC, ${ac_qcdoc_enabled}, [Enable QCDOC opts])
fi

dnl Check if profiling is enabled
if test ${ac_profile} -eq 1; then 
   AC_DEFINE_UNQUOTED(QDP_USE_PROFILING, ${ac_profile}, [Enable profiling])
   AC_MSG_NOTICE([Enable profiling])
fi

dnl Check if memory debugging is enabled
if test ${ac_memdebug_enabled} -eq 1; then 
   AC_DEFINE_UNQUOTED(QDP_DEBUG_MEMORY, ${ac_memdebug_1}, [Enable memory debugging])
   AC_MSG_NOTICE([Enable memory debugging])
fi

dnl Precision munging
AC_ARG_ENABLE(precision,
  AC_HELP_STRING([--enable-precision=single|double],
                 [Sets the base precision to be single or double precision (default is single) ]),
  [ac_precision=${enableval}],
  [ac_precision="single"])

case ${ac_precision} in 
single)
	AC_MSG_NOTICE( [Configuring QDP++ for single precision ] )
	AC_DEFINE(BASE_PRECISION,[32], [Single precision])
	AC_SUBST(CONFIG_PRECISION, [32])
	;;
double)
	AC_MSG_NOTICE( [Configuring for double precision ] )
	AC_DEFINE(BASE_PRECISION, [64], [Double precision] )
	AC_SUBST(CONFIG_PRECISION, [64])
	;;
*)
	AC_MSG_ERROR( [Unsupported value of precision. Check --enable-precision ] )
	;;
esac

AC_ARG_ENABLE(layout,
  AC_HELP_STRING([--enable-layout=lexico|cb2|cb32],
                 [Sets the layout. lexico=lexicographic, cb2=even odd checkerboard, cb32=hypercubic checkerboard. (default is cb2)]),
  [ ac_layout=${enableval} ],
  [ ac_layout="cb2" ]
)

case "${ac_layout}" in
lexico|lexicographic)
	AC_SUBST(CONFIG_LAYOUT,[lexicographic])
	AC_DEFINE(QDP_USE_LEXICO_LAYOUT, [1], [Use lexicographic layout])
	;;
cb2|checkerboard)
	AC_SUBST(CONFIG_LAYOUT,[cb2])
	AC_DEFINE(QDP_USE_CB2_LAYOUT, [1], [Use checkerboarded layout])
	;;
cb32|hyper|hypercube)
	AC_SUBST(CONFIG_LAYOUT,[cb32])
	AC_DEFINE(QDP_USE_CB32_LAYOUT, [1], [Use hypercube checkerboard layout])
	;;
*)
	AC_MSG_ERROR([Unsupported Layout. Check --enable-layout])
	;;
esac
AC_MSG_NOTICE([Configuring QDP++ with ${CONFIG_LAYOUT} layout])

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
    AC_DEFINE(ARCH_SCALAR, [1], [Scalar architecture])
    AC_SUBST(ARCH_SITE, "scalarsite")
    AC_MSG_NOTICE([ Scalar Build! Not checking for QMP ])
    AC_SUBST(QMP_HOME)
    AC_SUBST(QMP_CFLAGS)
    AC_SUBST(QMP_LDFLAGS)
    AC_SUBST(QMP_LIBS)
    ;;
  scalarvec)
    AC_DEFINE(ARCH_SCALARVEC, [1], [Scalar with vector extensions architecture])
    AC_SUBST(ARCH_SITE, "scalarvecsite")
    AC_MSG_NOTICE([ Scalar with Vector extensions Build! Not checking for QMP ])
    AC_SUBST(QMP_HOME)
    AC_SUBST(QMP_CFLAGS)
    AC_SUBST(QMP_LDFLAGS)
    AC_SUBST(QMP_LIBS)
    ;;
  parscalar)
    AC_DEFINE(ARCH_PARSCALAR, [1], [Parallel-scalar architecture])
    AC_SUBST(ARCH_SITE, "scalarsite")
    AC_MSG_NOTICE([ Parscalar build! Checking for QMP ])

    if test "X${QMP_HOME}X" = "XX" ; then
      AC_PATH_PROG([QMP_CONFIG], [qmp-config], [])
    else
      AC_PATH_PROG([QMP_CONFIG], [qmp-config], [], [${QMP_HOME}/bin:${PATH}])
    fi

    if test "X${QMP_CONFIG}X" != "XX" ; then
      AC_MSG_NOTICE([Found QMP configuration program ${QMP_CONFIG}])
      AC_SUBST(QMP_HOME,    "${QMP_HOME}")
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
    AC_DEFINE(ARCH_PARSCALARVEC, [1], [Parallel-scalar with vector extensions layout])
    AC_SUBST(ARCH_SITE, "scalarvecsite")
    AC_MSG_NOTICE([ Parscalar with Vector extensions build! Checking for QMP ])

    if test "X${QMP_HOME}X" = "XX" ; then
      AC_PATH_PROG([QMP_CONFIG], [qmp-config], [])
    else
      AC_PATH_PROG([QMP_CONFIG], [qmp-config], [], [${QMP_HOME}/bin:${PATH}])
    fi

    if test "X${QMP_CONFIG}X" != "XX" ; then
      AC_MSG_NOTICE([Found QMP configuration program ${QMP_CONFIG}])
      AC_SUBST(QMP_HOME,    "${QMP_HOME}")
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
AC_SUBST(CONFIG_PARALLEL_ARCH, [${PARALLEL_ARCH}])

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
  echo Freddie
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


AC_ARG_WITH(bagel-qdp,
  AC_HELP_STRING(
    [--with-bagel-qdp=DIR],
    [Build QDP++ using bagel_qdp where bagel_qdp is in DIR.]
  ),
  [BAGEL_QDP_DIR=${with_bagel_qdp}
   BAGEL_QDP_ENABLED="yes"],
  [BAGEL_QDP_DIR=""
   BAGEL_QDP_ENABLED="no" ]
)

if test "X${BAGEL_QDP_ENABLED}X" = "XyesX";
then 
     AC_MSG_NOTICE([Using BAGEL QDP])
     AC_DEFINE(QDP_USE_BAGEL_QDP, [1], [Use Bagel QDP library for BLAS])
     dnl get flags
     if test "X${BAGEL_QDP_DIR}X" = "XX" ; 
     then
       AC_PATH_PROG([BAGEL_QDP_CONFIG], [bagel-qdp-config], [])
     else
       AC_PATH_PROG([BAGEL_QDP_CONFIG], [bagel-qdp-config], [], [${BAGEL_QDP_DIR}/bin:${PATH}])
     fi     	

     if test "X${BAGEL_QDP_CONFIG}X" = "XX"; 
     then
       AC_MSG_ERROR([bagel-qdp-config not found])
     fi
     AC_MSG_NOTICE([Found bagel_qdp configuration program ${BAGEL_QDP_CONFIG}])
     AC_SUBST(BAGEL_QDP_CXXFLAGS, "`${BAGEL_QDP_CONFIG} --cxxflags`")
     AC_MSG_NOTICE([bagel_qdp compile flags: ${BAGEL_QDP_CXXFLAGS}])
     AC_SUBST(BAGEL_QDP_LDFLAGS, "`${BAGEL_QDP_CONFIG} --ldflags`")
     AC_MSG_NOTICE([bagel_qdp link flags : ${BAGEL_QDP_LDFLAGS}])
     AC_SUBST(BAGEL_QDP_LIBS,     "`${BAGEL_QDP_CONFIG} --libs`")
     AC_MSG_NOTICE([bagel_qdp libraries flags: ${BAGEL_QDP_LIBS}])

     AC_MSG_CHECKING([if we can compile/link a simple bagel_qdp program])
     PAC_BAGEL_QDP_LINK_CXX_FUNC(
       ${BAGEL_QDP_CXXFLAGS},
       ${BAGEL_QDP_LDFLAGS},
       ${BAGEL_QDP_LIBS},
       ,
       ,
       [bagelqdp_link_ok=yes],
       [bagelqdp_link_ok=no]
     )

     if test "X${bagelqdp_link_ok}X" = "XyesX" ; then 
        AC_MSG_RESULT(yes)
     else 
         AC_MSG_RESULT(no)
         AC_MSG_ERROR([Cannot compile/link a program with bagel_qdp. Use --with-bagel-qdp=<dir> to select a working version.])
     fi 
else
     AC_MSG_NOTICE([not using bagel_qdp])
     AC_DEFINE(QDP_USE_BAGEL_QDP, [0], [Donot use Bagel QDP library for BLAS])
     AC_SUBST(BAGEL_QDP_CXXFLAGS, "")
     AC_SUBST(BAGEL_QDP_LDFLAGS, "")
     AC_SUBST(BAGEL_QDP_LIBS,     "")
fi
      
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

dnl conditional for using Peters assembler
AM_CONDITIONAL(QDP_USE_QCDOC, [test "X${ac_qcdoc_enabled}X" = "X1X" -a "X${ARCH_SITE}X" = "XscalarsiteX"])

AM_CONDITIONAL(QDP_USE_SCALARVEC_SSE, [test "X${ac_sse}X" = "X1X" -a "X${ARCH_SITE}X" = "XscalarvecsiteX"])
AM_CONDITIONAL(QDP_USE_SCALARVEC_SSE2, [test "X${ac_sse2}X" = "X1X" -a "X${ARCH_SITE}X" = "XscalarvecsiteX"])

#
# For the examples directory, conditionally compile some files.
# These conditionals are *only* used there.
#
AM_CONDITIONAL(BUILD_STAGGERED_EXAMPLES,
  [test "X${ac_Ns}X" = "X1X"])
AM_CONDITIONAL(BUILD_WILSON_EXAMPLES,
  [test "X${ac_Ns}X" != "X1X"])

AM_CONDITIONAL(BUILD_DEFAULT_ALLOCATOR,
  [test "X${ac_qcdoc_enabled}X" = "X0X"])

##################################
# Check for programs to build docs
##################################
AC_CHECK_PROG(LATEX, latex, latex, true)
AC_CHECK_PROG(DVIPS, dvips, dvips, true)
AC_CHECK_PROG(DVIPDF, dvipdf, dvipdf, true)
AC_CHECK_PROG(DOXYGEN, doxygen, doxygen, true)

##################################
# Check for various functions
##################################
AC_CHECK_FUNCS(gethostname)
AC_CHECK_FUNCS(strnlen)

case ${PARALLEL_ARCH} in 
parscalar|parscalarvec)
        QMP_BKUP_CXXFLAGS="${CXXFLAGS}"
        QMP_BKUP_LDFLAGS="${LDFLAGS}"
 	QMP_BKUP_LIBS="${LIBS}"
        AC_LANG_SAVE
        CXXFLAGS="${CXXFLAGS} ${QMP_CFLAGS}"
	LDFLAGS="${LDFLAGS} ${QMP_LDFLAGS}"
	LIBS="${LIBS} ${QMP_LIBS}"
	AC_CHECK_FUNCS(QMP_abort QMP_error_exit QMP_verbose)
	CXXFLAGS="${QMP_BKUP_CXXFLAGS}"
	LDFLAGS="${QMP_BKUP_LDFLAGS}"
	LIBS="${LIBS}"
	AC_LANG_RESTORE
	;;
esac

#######################
# Produce output
#######################

#
# Configure subdir packages
#
AC_CONFIG_SUBDIRS(other_libs/qio)
AC_CONFIG_SUBDIRS(other_libs/xpath_reader)

#
# Set up the doxygen file
# Substitute the right directories
#
# AC_CONFIG_FILES(docs/qdpdoc)

#
# Generate makefiles
#
AC_CONFIG_FILES(Makefile)
AC_CONFIG_FILES(qdp++-config)
AC_CONFIG_FILES(lib/Makefile)
AC_CONFIG_FILES(other_libs/Makefile)
AC_CONFIG_FILES(include/Makefile)
#AC_CONFIG_FILES(docs/Makefile)
AC_CONFIG_FILES(examples/Makefile)
#
# Finish
#
AC_OUTPUT

