#pragma once

/**
  \file

  Conditionally print diagnostic messages during compilation.

  In the various USQCD projects, the `#warning` directive has been used to
  signal the various compile time options to the user. This can clutter the
  output and hide real warnings. Therefore this macro will enable toggling those
  warnings.

  With help from http://stackoverflow.com/a/43796429/653152.
  */

#ifdef QDPXX_EMIT_MESSAGES
#define QDPXX_MESSAGE_I(s) _Pragma(#s)
#define QDPXX_MESSAGE(s) QDPXX_MESSAGE_I(message(s))
#else
#define QDPXX_MESSAGE(s)
#endif
