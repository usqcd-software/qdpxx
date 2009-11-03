// -*- C++ -*-
// $Id: qdp_filebuf.cc,v 1.10 2008-06-27 13:31:22 bjoo Exp $

/*! @file
 * @brief Remote file support
 *
 * The routines here are only used in support of remote file IO
 * If the compile time flag USE_REMOTE_QIO is not defined, these
 * classes function as conventional streams writing/reading from
 * local (on node) files.
 */

#include <iostream>
#include <string>

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <new>
#include <cstring>
#include <cstdio>

#include "qdp_filebuf.h"

namespace QDP { 
extern void QDP_error_exit(const char *format, ...);
};
	
namespace QDPUtil
{

typedef std::char_traits<char>::int_type int_type;


//-------------------------------------------------------------
static bool remoteFileUse = false;

#if defined(USE_REMOTE_QIO)
extern "C"
{
  int qio_init(const char *node, int rtiP0);
  void qio_shutdown();
  FILE *qfopen(const char *pathname, const char *type);
}

void RemoteFileInit(const char *remote_node, bool useP)
{
  qio_init(remote_node, (useP)? 1 : 0);
  remoteFileUse = useP;
}

void RemoteFileShutdown()
{
  qio_shutdown();
}

FILE* RemoteFileOpen(const char* filename, const char *mode)
{
  return (remoteFileUse) ? qfopen(filename, mode) : fopen(filename, mode);
}

#else  // ! defined(USE_REMOTE_QIO)

void RemoteFileInit(const char *remote_node, bool useP) {}

void RemoteFileShutdown() {}

FILE* RemoteFileOpen(const char *path, const char *mode)
{
  return fopen(path, mode);
}
#endif


//-------------------------------------------------------------
// Remote output file streambuf
RemoteOutputFileBuf::RemoteOutputFileBuf()
{
  f = NULL;
  iop = false;         // set file status
}

void RemoteOutputFileBuf::open(const char *p, std::ios_base::openmode mode)
{
  if (is_open())
  {
    std::cerr << "Buf already open: error opening" << p << std::endl;
    return;
  }

  // Map C++ modes to C modes
  std::string cmode("w");

  if ((mode & std::ios_base::binary) != 0)
    cmode.append("b");

  if (((mode & std::ios_base::ate) != 0) || ((mode & std::ios_base::app) != 0))
    cmode.append("a");

  // Open the file
  if ((f = RemoteFileOpen(p, cmode.c_str())) != NULL)
    iop = true;

  if (! iop)
  {
    std::cerr << "Serial: open: error opening file" << p << std::endl;
    return;
  }
}

bool RemoteOutputFileBuf::is_open() 
{
  return iop;
}

void RemoteOutputFileBuf::close()
{
  if (is_open())
  {
    fclose(f);
    iop = false;
  }
}

RemoteOutputFileBuf::~RemoteOutputFileBuf() {close();}

// write one character
int_type 
RemoteOutputFileBuf::overflow (int_type c) 
{
  if (c != EOF)
  {
    char z = c;
    if (fwrite (&z, 1, 1, f) != 1) 
      return EOF;
  }
  return c;
}

// write multiple characters
std::streamsize 
RemoteOutputFileBuf::xsputn (const char* s,
			     std::streamsize num) 
{
  return fwrite(s,1,num,f);
}



//-------------------------------------------------------------------------
// Remote input file streambuf
//
RemoteInputFileBuf::RemoteInputFileBuf()
{
  f   = NULL;
  iop = false;        // set file status

  bufferSize = 50;    // size of the data buffer
  buffer = new(std::nothrow) char[bufferSize];  // data buffer
  if( buffer == 0x0 ){
    QDP::QDP_error_exit("Unable to new buffer in qdp_filebuf\n");
  }

  setg (buffer+4,     // beginning of putback area
	buffer+4,     // read position
	buffer+4);    // end position
}

void RemoteInputFileBuf::open(const char *p, std::ios_base::openmode mode)
{
  if (is_open()) 
    std::cerr << "Buf already open: error opening" << p << std::endl;

  // Map C++ modes to C modes
  std::string cmode("r");

  if ((mode & std::ios_base::binary) != 0)
    cmode.append("b");

  if (((mode & std::ios_base::ate) != 0) || ((mode & std::ios_base::app) != 0))
    cmode.append("a");

  // Open the file
  if ((f = RemoteFileOpen(p, cmode.c_str())) != NULL)
    iop = true;

  if (! iop)
    std::cerr << "Serial: open: error opening file  " << p << std::endl;
}

bool RemoteInputFileBuf::is_open() {return iop;}

void RemoteInputFileBuf::close()
{
  if (iop)
  {
    fclose(f);
    iop = false;
  }
}

RemoteInputFileBuf::~RemoteInputFileBuf()
{
  close();
  delete[] buffer;
}

int_type RemoteInputFileBuf::underflow()
{
  // is read position before end of buffer?
  if (gptr() < egptr()) {
    return traits_type::to_int_type(*gptr());
  }

  /* process size of putback area
   * - use number of characters read
   * - but at most four
   */
  int numPutback = gptr() - eback();
  if (numPutback > 4) {
    numPutback = 4;
  }

  /* copy up to four characters previously read into
   * the putback buffer (area of first four characters)
   */
  std::memmove (buffer+(4-numPutback), gptr()-numPutback,
		numPutback);

  // read new characters
  int num;
  num = fread (buffer+4, sizeof(char), bufferSize-4, f);
  if (num <= 0) 
  {
    // ERROR or EOF
    return EOF;
  }

  // reset buffer pointers
  setg (buffer+(4-numPutback),   // beginning of putback area
	buffer+4,                // read position
	buffer+4+num);           // end of buffer

  // return next character
  return traits_type::to_int_type(*gptr());
}




//----------------------------------------------------------------
void RemoteOutputFileStream::open(const char *f, std::ios_base::openmode mode)
{
  if (ib->is_open()) 
    std::cerr << "Stream already open: error opening: " << f << std::endl;

  ib->open(f,mode);
}

bool RemoteOutputFileStream::is_open()
{
  return ib->is_open();
}

void RemoteOutputFileStream::close()
{
  if (ib->is_open())
    ib->close();
}



//----------------------------------------------------------------
void RemoteInputFileStream::open(const char *f, std::ios_base::openmode mode)
{
  if (ib->is_open()) 
    std::cerr << "Stream already open: error opening: " << f << std::endl;

  ib->open(f,mode);
}

bool RemoteInputFileStream::is_open()
{
  return ib->is_open();
}

void RemoteInputFileStream::close()
{
  if (ib->is_open())
    ib->close();
}

} // namespace QDPUtil
