Quick installations instructions for QDP
========================================

To build QDP, it is recommended you make a subdirectory for the build and keep
the build tree separate from the source tree. E.g., say to build the “scalar”
version (single node workstation) of QDP++:

    $ cd qdp++
    $ mkdir scalar-build
    $ cd scalar-build
    $ ../configure --prefix=<path to your favorite installation directory>  \
                   --enable-parallel-arch=scalar
    $ make

The include files, libraries and docs are installed in the path specified by
`prefix`.

Extended examples of installation
=================================

See `INSTALL_EXAMPLES`

---

Status
======

Scalar and Parscalar library compiles, examples compile and link and run.

Software Prerequisites
======================

This implementation has been developed under `gcc-3.2`, and Intel C++ (32bit)
V7.0. To compile this code you will need a compiler that supports the C++ 3.0
standard --- e.g., `gcc-3.x`.  Also needed: `gmake` and `libxml2`.  To be able
to play with the build system you'll also need `autoconf-2.5x` and
`automake-1.7.x` and `gm4`.

Installation
============

See the `./INSTALL` file.

Compile Time Parameters
=======================

The macros `NC`, `ND` and `NS` are defined in `params.h`, but in
`qdp_config.h`. `params.h` includes `qdp_config.h` and sets `Nc=NC`, `ND=Nd`
and `NS=Ns` from the defined values `NC`, `ND` and `NS` which are defined in
`qdp_config.h`.

The file `qdp_config.h` is created by the configure script from the
template `qdp_config.h.in` and the resulting file is placed in the build
directory `include/qdp_config.h`. Apart from defining `NC`, `ND` and `NS` it
also defines the relevant `ARCH_[PAR]SCALAR` which defines the type of
parallel architecture. Hence `qdp.h` also includes `qdp_config.h`.

Usage
=====

The installation creates and installs a library --- depending on the
environment it is called `libqdp.a`. In the installation directory there is an
`include` and a `lib` subdirectory where the relevant files are found. The user
uses compiles their code with the appropriate `include` and link flags pointing
to the installation directories. E.g., assuming configuration with
`--prefix=/usr/local/qdp++/scalar`:

    $ g++ -I/usr/local/qdp++/scalar/include -L/usr/local/qdp++/scalar/lib  \
      myfile.cc  -lqdp

If QMP is used there also needs to be links to its installation directories as
well.

Autoconf & Automake
===================

The build system has been converted to work with autoconf/automake. If you just
want to build/install QDP++ you shouldn't need to care about this.

If you are intending to develop the system you need to know how
autoconf/automake work.

Short summary:

Autoconf is a programming language based on shell (`sh`) script and `m4` macros
that allow you to write configure scripts. You edit the `configure.ac` script
with your editor and run `autotoconf` to turn it into a `configure` script.

The `configure` script allows configuration in several ways. It can substitute
values in files, it can define compiler `#define`s amongst other things.
Generally it maps template `Makefile.in` files to `Makefile` files (that get
used by your build)

Automake is a system to allow you to write `Makefile.in` files for `autoconf.`
Generally Automake takes a set of `Makefile.am` files and produces
`Makefile.in` files from it. These then get turned into `Makefile` files by the
`configure` script.

If you add a file to say a directory (say a `foo.cc` in `lib/`) and you want it
to be compiled into the library, you must do the following:

1. drop the `foo.cc` file into `lib/`
2. Edit lib/Makefile.am and add the source file at the end of the right `SOURCES`
   or `LIBADD` primary
3. Go to the toplevel directory and type `autoreconf`

You should find that if things went well, that when you next type `./configure`
everything should be taken care of.

If things go badly and `autoreconf` doesn't work, because
say there is a version mismatch with `automake` then remove
the `aclocal` file from the toplevel directory.

then type: 

    $ aclocal
    $ autoconf
    $ automake --add-missing --copy 

`aclocal` creates an `aclocal.m4` file containing macros for automake
`autoconf` and `automake` need to be run. Automake needs a few files
present (like `COPYING`, `NEWS`, `AUTHORS`, `ChangeLog` etc). If you 
don't have these the `--add-missing` `--copy` should get them for you.

`$ autoreconf` should work thereafter.

More pointers: 
--------------

If you add a header file to `include/` also add it to `Makefile.am` to the
`include_HEADERS` primary (so that it gets installed when you type `make
install`) 

in `lib/` you can make files depend on headers by adding them to the `HDRS`
macro. Note that `HDRS` in `lib/Makefile.am` is a part of
`nodist_libqdp_a_SOURCE` as those headers are also put into the distro by the
`Makefile.am`

If you just drop a file that you don't want to use yet, but want the file there
for later use, add it to the `EXTRA_DIST` line in to the `Makefile.am` in the
top source directory.

Rolling Distribution Tarballs
-----------------------------

Once you have configured QDP++, you should be able to create a distribution
tarball (`qdp-ver.tar.gz` --- where `ver` is version number specified in the
first line of `configure.ac`) by typing

    make dist

You can then put this distro on a web page or mail it to friends.

The rules of what goes into a distribution are fairly poorly specified. More or
less anything in a `_SOURCES` or `_HEADERS` will be put in, as well as all the
`Makefile.in`s and `Makefile.am`s and the `configure.ac` etc.

For things that are not in these (say `QDPClasses.in`) you have to add extra
instructins for the file to be included in a distribution tarball (see the
`EXTRA_DIST` Automake variable --- for things that need to be added in as
extra, and the `nodist_` primary prefix --- for things that are sources but
shouldn't be added (eg `qdp_config.h` shouldn't be added because it should
always be recreated at configure time)

Further Information
-------------------

- [The Autoconf Manual](http://www.gnu.org/manual/autoconf-2.53/autoconf.html)
- [The Automake Manual](http://www.gnu.org/manual/automake-1.6.1/automake.html)
- [GNU Autoconf, Automake and Libtool](http://sources.redhat.com/autobook)
- [Learning Autoconf and Automake](http://www.amath.washington.edu/~lf/tutorials/autoconf)

Software versions used in building the build system
---------------------------------------------------

    autoconf-2.56
    automake-1.7.1

You will also probably want a version of `gm4` installed (the above use it) and
`gmake` too for its `VPATH` support.
