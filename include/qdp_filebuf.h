// -*- C++ -*-
// $Id: qdp_filebuf.h,v 1.1 2003-06-06 02:39:30 edwards Exp $

/*! @file
 * @brief IO support
 */

#include <ostream>

#include <streambuf>

namespace QDPUtil
{

/*! @defgroup io IO
 *
 * File input and output operations on QDP types
 *
 * @{
 */

//--------------------------------------------------------------------------------
//! RemoteOutputFileBuf class
/*! Use the qdaemon/qio facility for remote opening of files
 *  The returned file descriptor is wrapped inside the streambuf
 */
class RemoteOutputFileBuf : public std::streambuf 
{
public:
  //! Constructors are always empty
  RemoteOutputFileBuf();

  //! open a remote file
  void open(const char *filename);
  
  //! close a remote file
  void close();
  
  //! test if file is open
  bool is_open();
 
  //! destructor
  ~RemoteOutputFileBuf();

protected:
  virtual int_type overflow (int_type c);

  virtual std::streamsize xsputn (const char* s, 
				  std::streamsize num);

private:
  //! Use C-stdio
  FILE* f;
  bool iop;
};



//--------------------------------------------------------------------------------
//! RemoteInputFileBuf class
/*! Use the qdaemon/qio facility for remote opening of files
 *  The returned file descriptor is wrapped inside the streambuf
 */
class RemoteInputFileBuf : public std::streambuf 
{
public:
  //! partial constructor
  RemoteInputFileBuf();

  //! open a file
  void open(const char *p);
  
  //! close a file
  void close();
  
  //! test if file is open
  bool is_open();
 
  //! destructor
  ~RemoteInputFileBuf();

protected:
  //! initialize internally
  void init ();

  //! insert new characters into the buffer
  virtual int_type underflow ();

private:
  /* data buffer:
   * - at most, four characters in putback area plus
   * - at most, six characters in ordinary read buffer
   */
  static const int bufferSize = 256;   // size of the data buffer
  char buffer[bufferSize];             // data buffer

private:
  //! Use C-stdio
  FILE* f;
  bool iop;
};





//--------------------------------------------------------------------------------
//! RemoteOutputFileStream class
/*! Stream class using the RemoteOutputFileBuf facility
 */
class RemoteOutputFileStream : public std::ostream
{
private:
  RemoteOutputFileBuf*  ib;

public:
  //! Constructor
  RemoteOutputFileStream() :
    std::ostream(ib = new RemoteOutputFileBuf()) {}

  //! Construct from a file
  RemoteOutputFileStream(const char* p) :
    std::ostream(ib = new RemoteOutputFileBuf()) {open(p);}

  //! open a file
  void open(const char *p);
  
  //! close a file
  void close();
  
  //! test if file is open
  bool is_open();
 
  //! destructor
  ~RemoteOutputFileStream() {close();}

  // No default constructor
};



//--------------------------------------------------------------------------------
//! RemoteInputFileStream class
/*! Stream class using the RemoteInputFileBuf facility
 */
class RemoteInputFileStream : public std::istream
{
private:
  RemoteInputFileBuf*  ib;

public:
  //! Constructor
  RemoteInputFileStream() :
    std::istream(ib = new RemoteInputFileBuf()) {}

  //! Construct from a file
  RemoteInputFileStream(const char* p) :
    std::istream(ib = new RemoteInputFileBuf()) {open(p);}

  //! open a file
  void open(const char *p);
  
  //! close a file
  void close();
  
  //! test if file is open
  bool is_open();
 
  //! destructor
  ~RemoteInputFileStream() {close();}

  // No default constructor
};



/*! @} */   // end of group io

} // namespace QDPUtil
