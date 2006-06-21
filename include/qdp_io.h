// -*- C++ -*-
// $Id: qdp_io.h,v 1.21 2006-06-21 13:00:02 bjoo Exp $

/*! @file
 * @brief IO support
 */

#include <string>
#include <fstream>
#include <sstream>
#include "qdp_byteorder.h"

QDP_BEGIN_NAMESPACE(QDP);


/*! @defgroup io IO
 *
 * File input and output operations on QDP types
 *
 * @{
 */

//--------------------------------------------------------------------------------
//! Text input class
/*!
  This class is used to read data from a text file. Input is done on the
  primary node and all nodes end up with the same data.

  The read methods are also wrapped by externally defined >> operators,
*/

class TextReader
{
public:
  TextReader();
  /*!
    Closes the last file opened
  */
  ~TextReader(); 

  /*!
    Opens a file for reading.
    \param p The name of the file
  */
  explicit TextReader(const std::string& p);

  //! Opens a file for reading.
  /*!
    \param p The name of the file
  */
  void open(const std::string& p);

  //! Closes the last file opened
  void close();

  //! Queries whether the file is open
  /*!
    \return true if the file is open; false otherwise.
  */
  bool is_open();

  //!Checks status of the previous IO operation.
  /*!
    \return true if some failure occurred in the previous IO operation
  */
  bool fail();

  // Readers for builtin types
  void read(std::string& result);
  void read(char& result);
  void read(int& result);
  void read(unsigned int& result);
  void read(short int& result);
  void read(unsigned short int& result);
  void read(long int& result);
  void read(unsigned long int& result);
  void read(float& result);
  void read(double& result);
  void read(bool& result);

protected:

  //! The universal data-reader.
  /*!
    All the read functions call this.
    \param output The location to which the datum is read.
  */
  template< typename T>
  void
  readPrimitive(T& output);

  //! Get the internal input stream
  std::istream& getIstream() {return f;}

private:
#if defined(USE_REMOTE_QIO)
  QDPUtil::RemoteInputFileStream f;
#else
  ifstream f;
#endif
};


// Different bindings for same operators
TextReader& operator>>(TextReader& txt, std::string& input);
TextReader& operator>>(TextReader& txt, char& input);
TextReader& operator>>(TextReader& txt, int& input);
TextReader& operator>>(TextReader& txt, unsigned int& input);
TextReader& operator>>(TextReader& txt, short int& input);
TextReader& operator>>(TextReader& txt, unsigned short int& input);
TextReader& operator>>(TextReader& txt, long int& input);
TextReader& operator>>(TextReader& txt, unsigned long int& input);
TextReader& operator>>(TextReader& txt, float& input);
TextReader& operator>>(TextReader& txt, double& input);
TextReader& operator>>(TextReader& txt, bool& input);


//-----------------------------------------
//! Text output class
/*!
  This class is used to write data to a text file.
  Output is done from the primary node only..
  
  The write methods are also wrapped by externally defined >> operators,
*/

class TextWriter
{
public:
  TextWriter();

  /*!
    Closes the last file opened
  */
  ~TextWriter();

  /*!
    Opens a file for writing.
    \param p The name of the file
  */
  explicit TextWriter(const std::string& p);

  //! Queries whether the file is open
  /*!
    \return true if the file is open; false otherwise.
  */
  bool is_open();

  /*!
    Opens a file for writing.
    \param p The name of the file
  */
  void open(const std::string& p);

  //! Closes the last file opened
  void close();

  //! Flushes the buffer
  void flush();

  //!Checks status of the previous IO operation.
  /*!
    \return true if some failure occurred in previous IO operation
  */
  bool fail();

  // Overloaded Writer Functions
  void write(const std::string& output);
  void write(const char* output);
  void write(const char& output);
  void write(const int& output);
  void write(const unsigned int& output);
  void write(const short int& output);
  void write(const unsigned short int& output);
  void write(const long int& output);
  void write(const unsigned long int& output);
  void write(const float& output);
  void write(const double& output);
  void write(const bool& output);

protected:
  //! The universal data-writer.
  /*!
    All the write functions call this.
    \param output The location of the datum to be written.
  */
  template< typename T>
  void
  writePrimitive(const T& output);

  //! Get the internal output stream
  std::ostream& getOstream() {return f;}

private:
#if defined(USE_REMOTE_QIO)
  QDPUtil::RemoteOutputFileStream f;
#else
  ofstream f;
#endif
};


// Different bindings for same operators
TextWriter& operator<<(TextWriter& txt, const std::string& output);
TextWriter& operator<<(TextWriter& txt, const char* output);
TextWriter& operator<<(TextWriter& txt, char output);
TextWriter& operator<<(TextWriter& txt, int output);
TextWriter& operator<<(TextWriter& txt, unsigned int output);
TextWriter& operator<<(TextWriter& txt, short int output);
TextWriter& operator<<(TextWriter& txt, unsigned short int output);
TextWriter& operator<<(TextWriter& txt, long int output);
TextWriter& operator<<(TextWriter& txt, unsigned long int output);
TextWriter& operator<<(TextWriter& txt, float output);
TextWriter& operator<<(TextWriter& txt, double output);
TextWriter& operator<<(TextWriter& txt, bool output);


//--------------------------------------------------------------------------------
//!  Binary input class
/*!
  This class is used to read data from a binary file. The data in the file
  is assumed to be big-endian. If the host nachine is little-endian, the data
  is byte-swapped. All nodes end up with the same data
  
  The read methods are also wrapped by externally defined functions
  and >> operators,   
*/
class BinaryReader
{
public:
  BinaryReader();

  /*!
    Closes the last file opened
  */
  ~BinaryReader();

  /*!
    Opens a file for reading.
    \param p The name of the file
  */
  explicit BinaryReader(const std::string& p);

  //! Queries whether the file is open
  /*!
    \return true if the file is open; false otherwise.
  */
  bool is_open();

  //! Opens a file for reading.
  /*!
    \param p The name of the file
  */
  void open(const std::string& p);

  //! Closes the last file opened
  void close();

  //!Checks status of the previous IO operation.
  /*!
    \return true if some failure occurred in the previous IO operation
  */
  bool fail();

  //! Read data on the primary node only
  /*!
    \param output The location to which data is read
    \param nbytes The size in bytes of each datum
    \param The number of data.
  */
  void readArrayPrimaryNode(char* output, size_t nbytes, size_t nmemb);

  //! Read data on the primary node and broadcast to all nodes.
  /*!
    \param output The location to which data is read
    \param nbytes The size in bytes of each datum
    \param The number of data.
  */
  void readArray(char* output, size_t nbytes, size_t nmemb);

  // Overloaded reader functions

  //! Read some max number of characters - 1 upto and excluding a newline
  /*! This is the getline function for the underlying stream */
  void read(std::string& result, size_t nbytes);

  void read(char& result);
  void read(int& result);
  void read(unsigned int& result);
  void read(short int& result);
  void read(unsigned short int& result);
  void read(long int& result);
  void read(unsigned long int& result);
  void read(float& result);
  void read(double& result);
  void read(bool& result);

  //! Get the current checksum
  QDPUtil::n_uint32_t getChecksum() const;
  
protected:

  //! The universal data-reader.
  /*!
    All the read functions call this.
    \param output The location to which the datum is read.
  */
  template< typename T>
  void
  readPrimitive(T& output);

  //! Get the internal input stream
  std::istream& getIstream() {return f;}

private:
  //! Checksum
  QDPUtil::n_uint32_t checksum;

#if defined(USE_REMOTE_QIO)
  QDPUtil::RemoteInputFileStream f;
#else
  ifstream f;
#endif
};

// Telephone book of basic primitives
void read(BinaryReader& bin, std::string& input, size_t maxBytes);
void read(BinaryReader& bin, char& input);
void read(BinaryReader& bin, int& input);
void read(BinaryReader& bin, unsigned int& input);
void read(BinaryReader& bin, short int& input);
void read(BinaryReader& bin, unsigned short int& input);
void read(BinaryReader& bin, long int& input);
void read(BinaryReader& bin, unsigned long int& input);
void read(BinaryReader& bin, float& input);
void read(BinaryReader& bin, double& input);
void read(BinaryReader& bin, bool& input);

// Different bindings for same operators
BinaryReader& operator>>(BinaryReader& bin, char& input);
BinaryReader& operator>>(BinaryReader& bin, int& input);
BinaryReader& operator>>(BinaryReader& bin, unsigned int& input);
BinaryReader& operator>>(BinaryReader& bin, short int& input);
BinaryReader& operator>>(BinaryReader& bin, unsigned short int& input);
BinaryReader& operator>>(BinaryReader& bin, long int& input);
BinaryReader& operator>>(BinaryReader& bin, unsigned long int& input);
BinaryReader& operator>>(BinaryReader& bin, float& input);
BinaryReader& operator>>(BinaryReader& bin, double& input);
BinaryReader& operator>>(BinaryReader& bin, bool& input);

//! Read a binary multi1d object
/*!
  This assumes that the number of elements to be read is also written in
  the file, \e i.e. that the data was written with the corresponding write
  code.
  \param bin The initialised binary reader
  \param d The data to be filled.

  \pre The binary reader must have opened the file.
  \post The multi1d can be resized.
*/
template<class T>
inline
void read(BinaryReader& bin, multi1d<T>& d)
{
  int n;
  read(bin, n);    // the size is always written, even if 0
  d.resize(n);

  for(int i=0; i < d.size(); ++i)
    read(bin, d[i]);
}

//! Read a binary multi1d object
/*!
  This assumes that the number of elements to be read is not written in
  the file, \e i.e. that the data was written with the corresponding write
  code. The number of elements must therefore be supplied by the caller
  \param bin The initialised binary reader
  \param d The data to be filled.
  \param num The number of elements.

  \pre The binary reader must have opened the file.
  \pre The multi1d must have space for at least \a num elements.  
*/
template<class T>
inline
void read(BinaryReader& bin, multi1d<T>& d, int num)
{
  for(int i=0; i < num; ++i)
    read(bin, d[i]);
}

//! Read a binary multi2d object
/*!
  This assumes that the number of elements to be read is not written in
  the file, \e i.e. that the data was written with the corresponding write
  code. The number of elements must therefore be supplied by the caller
  \param bin The initialised binary reader
  \param d The data to be filled.
  \param num1 The first dimension of the array
  \param num2 The second dimension of the array..  

  \pre The binary reader must have opened the file.
  \pre The multi2d must have space for at least \a num elements.  
*/

template<class T>
inline
void read(BinaryReader& bin, multi2d<T>& d, int num1, int num2)
{

  for(int i=0; i < num2; ++i)
    for(int j=0; j < num1; ++j)
      read(bin, d[j][i]);

}


//! Read a binary multi2d element
/*!
  This assumes that the number of elements to be read is also written in
  the file, \e i.e. that the data was written with the corresponding write
  code.
  \param bin The initialised binary reader
  \param d The data to be filled.

  \pre The binary reader must have opened the file.
  \post The multi2d can be resized.
*/
template<class T>
inline
void read(BinaryReader& bin, multi2d<T>& d)
{
  int n1;
  int n2;
  read(bin, n1);    // the size is always written, even if 0
  read(bin, n2);    // the size is always written, even if 0
  d.resize(n1,n2);
  
  for(int i=0; i < d.size1(); ++i)
    for(int j=0; j < d.size2(); ++j)
    {
      read(bin, d[j][i]);
    }

}



//! Read a binary multi3d element
/*!
  This assumes that the number of elements to be read is also written in
  the file, \e i.e. that the data was written with the corresponding write
  code.
  \param bin The initialised binary reader
  \param d The data to be filled.

  \pre The binary reader must have opened the file.
  \post The multi2d can be resized.
*/
template<class T>
inline
void read(BinaryReader& bin, multi3d<T>& d)
{
  int n1;
  int n2;
  int n3;
  read(bin, n1);    // the size is always written, even if 0
  read(bin, n2);    // the size is always written, even if 0
  read(bin, n3);    // the size is always written, even if 0

  // Destructively resize the array
  d.resize(n3,n2,n1);

  for(int i=0; i < d.size1(); ++i)
    for(int j=0; j < d.size2(); ++j)
      for(int k=0; k < d.size3(); ++k)
      {
	read(bin, d[k][j][i]);
      }

}

//!  Binary output class
/*!
  This class is used to write data to a binary file. The data in the file
  is big-endian. If the host nachine is little-endian, the data
  is byte-swapped.   Output is done from the primary node only.

  Files need to be opened before any of the write methods are used  
  
  The write methods are also wrapped by externally defined functions
  and << operators,   
*/
class BinaryWriter
{
public:
  BinaryWriter();

  /*!
    Closes the last file opened
  */
  ~BinaryWriter();

  /*!
    Opens a file for writing.
    \param p The name of the file
  */
  explicit BinaryWriter(const std::string& p);

  //! Queries whether the file is open
  /*!
    \return true if the file is open; false otherwise.
  */
  bool is_open();
    
  /*!
    Opens a file for writing.
    \param p The name of the file
  */
  void open(const std::string& p);

  //! Closes the last file opened   
  void close();

  //! Flushes the buffer
  void flush();

  //!Checks status of the previous IO operation.
  /*!
    \return true if some failure occurred in previous IO operation
  */
  bool fail();

  //! Write data from the primary node.
  /*!
    \param output The data to write
    \param nbytes The size in bytes of each datum
    \param The number of data.
  */
  void writeArray(const char* output, size_t nbytes, size_t nmemb);

  // Overloaded Writer Functions

  /*!
    A newline is appended to the written string.
  */
  void write(const std::string& output);
  /*!
    A newline is appended to the written string.
  */
  void write(const char* output);
  void write(const char& output);
  void write(const int& output);
  void write(const unsigned int& output);
  void write(const short int& output);
  void write(const unsigned short int& output);
  void write(const long int& output);
  void write(const unsigned long int& output);
  void write(const float& output);
  void write(const double& output);
  void write(const bool& output);

  //! Get the current checksum
  QDPUtil::n_uint32_t getChecksum() const;
  
protected:

  //! The universal data-writer.
  /*!
    All the write functions call this.
    \param output The location of the datum to be written.
  */
  template< typename T>
  void
  writePrimitive(const T& output);

  //! Get the internal output stream
  std::ostream& getOstream() {return f;}

private:
  //! Checksum
  QDPUtil::n_uint32_t checksum;

#if defined(USE_REMOTE_QIO)
  QDPUtil::RemoteOutputFileStream f;
#else
  ofstream f;
#endif
};


// Telephone book of basic primitives
void write(BinaryWriter& bin, const std::string& output);
void write(BinaryWriter& bin, const char* output);
void write(BinaryWriter& bin, char output);
void write(BinaryWriter& bin, int output);
void write(BinaryWriter& bin, unsigned int output);
void write(BinaryWriter& bin, short int output);
void write(BinaryWriter& bin, unsigned short int output);
void write(BinaryWriter& bin, long int output);
void write(BinaryWriter& bin, unsigned long int output);
void write(BinaryWriter& bin, float output);
void write(BinaryWriter& bin, double output);
void write(BinaryWriter& bin, bool output);

// Different bindings for same operators
BinaryWriter& operator<<(BinaryWriter& bin, const std::string& output);
BinaryWriter& operator<<(BinaryWriter& bin, const char* output);
BinaryWriter& operator<<(BinaryWriter& bin, char output);
BinaryWriter& operator<<(BinaryWriter& bin, int output);
BinaryWriter& operator<<(BinaryWriter& bin, unsigned int output);
BinaryWriter& operator<<(BinaryWriter& bin, short int output);
BinaryWriter& operator<<(BinaryWriter& bin, unsigned short int output);
BinaryWriter& operator<<(BinaryWriter& bin, long int output);
BinaryWriter& operator<<(BinaryWriter& bin, unsigned long int output);
BinaryWriter& operator<<(BinaryWriter& bin, float output);
BinaryWriter& operator<<(BinaryWriter& bin, double output);
BinaryWriter& operator<<(BinaryWriter& bin, bool output);

//! Write all of a binary multi1d object
/*!
  This also writes the number of elements to the file.
  \param bin The initialised binary reader
  \param d The data to be filled.

  \pre The binary reader must have opened the file.
*/
template<class T>
inline
void write(BinaryWriter& bin, const multi1d<T>& d)
{
  write(bin, d.size());    // always write the size
  for(int i=0; i < d.size(); ++i)
    write(bin, d[i]);
}

//! Write some or all of a binary multi1d object
/*!
  This does not write the number of elements to the file.
  \param bin The initialised binary writer
  \param d The data to be filled.
  \param num The number of elements to write.

  \pre The binary writer must have opened the file.
*/
template<class T>
inline
void write(BinaryWriter& bin, const multi1d<T>& d, int num)
{
  for(int i=0; i < num; ++i)
    write(bin, d[i]);
}



//! Write a binary multi2d element
template<class T>
inline
void write(BinaryWriter& bin, const multi2d<T>& d)
{
  write(bin, d.size2());    // always write the size
  write(bin, d.size1());    // always write the size

  for(int i=0; i < d.size1(); ++i)
    for(int j=0; j < d.size2(); ++j)
    {
      write(bin, d[j][i]);
    }

}

//! Write a fixed number of binary multi2d element - no element count written
template<class T>
inline
void write(BinaryWriter& bin, const multi2d<T>& d, int num1, int num2)
{
  for(int i=0; i < num2; ++i)
    for(int j=0; j < num1; ++j)
      write(bin, d[j][i]);
}



//! Write a binary multi2d element
template<class T>
inline
void write(BinaryWriter& bin, const multi3d<T>& d)
{
  write(bin, d.size3());    // always write the size
  write(bin, d.size2());    // always write the size
  write(bin, d.size1());    // always write the size

  for(int i=0; i < d.size1(); ++i)
    for(int j=0; j < d.size2(); ++j)
      for(int k=0; k < d.size3(); ++k)
	write(bin, d[k][j][i]);

}

//! Write a fixed number of binary multi3d element - no element count written
template<class T>
inline
void write(BinaryWriter& bin, const multi3d<T>& d, 
	   int num1, int num2, int num3)
{
  for(int k=0; k < num3 ; ++k)
    for(int j=0; j < num2; ++j)
      for(int i=0; i < num1; ++i)
	write(bin, d[i][j][k]);

}


/*! @} */   // end of group io
QDP_END_NAMESPACE();
