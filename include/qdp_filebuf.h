// -*- C++ -*-

/*! @file
 * @brief Remote file support
 *
 * The routines here are only used in support of remote file IO.
 * If the compile time flag USE_REMOTE_QIO is not defined, these
 * classes function as conventional streams writing/reading from
 * local (on node) files.
 */

#include <ostream>

#include <streambuf>

namespace QDPUtil
{

/*! @ingroup io
 *
 * @{
 */

//--------------------------------------------------------------------------------
//! Initialize remote file system
/*!
  QIO wrapper.
  \param remote_node
  \param useP
*/
void RemoteFileInit(const char *remote_node, bool useP);

//! Shutdown remote file system
/*!
  QIO wrapper.
*/
void RemoteFileShutdown();

//! Open a remote file
/*!
  QIO wrapper.
  \param path
  \param mode
 */
FILE* RemoteFileOpen(const char *path, const char *mode);



//--------------------------------------------------------------------------------
//! RemoteOutputFileBuf class
/*! Use the qdaemon/QIO facility for remote opening of files
 *  The returned file descriptor is wrapped inside the streambuf
 */
class RemoteOutputFileBuf : public std::streambuf 
{
public:
  //! Constructors are always empty
  RemoteOutputFileBuf();

    //! Opens a remote file
    /*!
      If this object already has a file open, then this returns without
      doing anything.
    */
  void open(const char *filename, std::ios_base::openmode mode);
  
  //! Closes the remote file
  void close();
  
    //! Queries whether the file is open
    /*!
      \return true if the file is open; false otherwise.
    */
  bool is_open();

    /*! Closes the remote file    */
  ~RemoteOutputFileBuf(); 

protected:
    //! Write a character to the file
    /*!
      \param c The character to write
      \return The character just written, or EOF if it could not be written.
    */
  virtual int_type overflow (int_type c);

    //! Write characters to the file
    /*!
      \param s The characters to write
      \return The number of characters written.
    */
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
  void open(const char *p, std::ios_base::openmode mode);
  
  //! close a file
  void close();
  
  //! test if file is open
  bool is_open();
 
  //! destructor
  ~RemoteInputFileBuf();

protected:
  //! insert new characters into the buffer
  virtual int_type underflow ();

private:
  /* data buffer:
   * - at most, four characters in putback area plus
   * - at most, six characters in ordinary read buffer
   */
  int bufferSize;           // size of the data buffer
  char *buffer;             // data buffer

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
  RemoteOutputFileStream(const char* p, std::ios_base::openmode mode) :
    std::ostream(ib = new RemoteOutputFileBuf()) {open(p,mode);}

  //! open a file
  void open(const char *p, std::ios_base::openmode mode);
  
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
  RemoteInputFileStream(const char* p, std::ios_base::openmode mode) :
    std::istream(ib = new RemoteInputFileBuf()) {open(p,mode);}

  //! open a file
  void open(const char *p, std::ios_base::openmode mode);
  
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
