// -*- C++ -*-
// $Id: qdp_filebuf.cc,v 1.1 2003-06-06 02:39:30 edwards Exp $

/*! @file
 * @brief Remote file support
 *
 * The routines here are only used in support of remote file IO
 */

// #include <cstdio>
#include <iostream>

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>


#include "qdp_filebuf.h"

extern "C"
{
  FILE *qfopen(const char *pathname, const char *type);
}


namespace QDPUtil
{

typedef std::char_traits<char>::int_type int_type;

//-----------------------------------------
// Remote output file streambuf
RemoteOutputFileBuf::RemoteOutputFileBuf()
{
  f = NULL;
  iop = false;         // set file status
}

void RemoteOutputFileBuf::open(const char *p)
{
  if (is_open())
  {
    std::cerr << "Buf already open: error opening" << p << std::endl;
    return;
  }

  if ((f = qfopen(p, "wb")) != NULL)
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
  iop = false;        // set file status

  init();
}

void RemoteInputFileBuf::init()
{
  setg (buffer+4,     // beginning of putback area
	buffer+4,     // read position
	buffer+4);    // end position
}

void RemoteInputFileBuf::open(const char *p)
{
  if (is_open()) 
    std::cerr << "Buf already open: error opening" << p << std::endl;

  if ((f = qfopen(p,"rb")) != NULL)
    iop = true;

  if (f == NULL)
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

RemoteInputFileBuf::~RemoteInputFileBuf() {close();}

int_type RemoteInputFileBuf::underflow()
{
  std::cerr << "enter underflow" << std::endl;

  // is read position before end of buffer?
  if (gptr() < egptr()) {
    return traits_type::to_int_type(*gptr());
  }

  std::cerr << "here a" << std::endl;

  /* process size of putback area
   * - use number of characters read
   * - but at most four
   */
  int numPutback = gptr() - eback();
  if (numPutback > 4) {
    numPutback = 4;
  }

  std::cerr << "here b" << std::endl;

  /* copy up to four characters previously read into
   * the putback buffer (area of first four characters)
   */
  std::memmove (buffer+(4-numPutback), gptr()-numPutback,
		numPutback);

  // read new characters
  int num;
  num = fread (buffer+4, sizeof(char), bufferSize-4, f);
  std::cerr << "read " << num << "  chars" << std::endl;
  if (num <= 0) 
  {
    // ERROR or EOF
    return EOF;
  }

  std::cerr << "reset" << std::endl;

  // reset buffer pointers
  setg (buffer+(4-numPutback),   // beginning of putback area
	buffer+4,                // read position
	buffer+4+num);           // end of buffer

  std::cerr << "return " << *gptr() << std::endl;

  // return next character
  return traits_type::to_int_type(*gptr());
}




//----------------------------------------------------------------
void RemoteOutputFileStream::open(const char *f)
{
  if (ib->is_open()) 
    std::cerr << "Stream already open: error opening: " << f << std::endl;

  ib->open(f);
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
void RemoteInputFileStream::open(const char *f)
{
  if (ib->is_open()) 
    std::cerr << "Stream already open: error opening: " << f << std::endl;

  ib->open(f);
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
