// -*- C++ -*-
// $Id: qdp_io.h,v 1.16 2004-11-22 19:31:31 edwards Exp $

/*! @file
 * @brief IO support
 */

#include <string>
#include <fstream>
#include <sstream>

QDP_BEGIN_NAMESPACE(QDP);


/*! @defgroup io IO
 *
 * File input and output operations on QDP types
 *
 * @{
 */

//--------------------------------------------------------------------------------
//! Simple input text class
class TextReader
{
public:
  TextReader();
  ~TextReader();
  explicit TextReader(const std::string& p);

  //! Open file
  void open(const std::string& p);

  //! Close file
  void close();
  bool is_open();

  //! Return true if some failure occurred in previous IO operation
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
  // The universal data-reader. All the read functions call this
  template< typename T>
  void
  readPrimitive(T& output);

  // Get the internal ostream
  std::istream& getIstream() {return f;}

private:
  QDPUtil::RemoteInputFileStream f;
// ifstream f;
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
//! Simple output text class
class TextWriter
{
public:
  TextWriter();
  ~TextWriter();
  explicit TextWriter(const std::string& p);

  bool is_open();
  void open(const std::string& p);
  void close();

  //! Flush the buffer
  void flush();

  //! Return true if some failure occurred in previous IO operation
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
  // The universal data-write. All the write functions call this
  template< typename T>
  void
  writePrimitive(const T& output);

  // Get the internal ostream
  std::ostream& getOstream() {return f;}

private:
  QDPUtil::RemoteOutputFileStream f;
// ofstream f;
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
//! Simple output binary class
class BinaryReader
{
public:
  BinaryReader();
  ~BinaryReader();
  explicit BinaryReader(const std::string& p);

  bool is_open();
  void open(const std::string& p);
  void close();

  //! Return true if some failure occurred in previous IO operation
  bool fail();

  //! Basic read function on the primary node
  void readArrayPrimaryNode(char* output, size_t nbytes, size_t nmemb);

  //! Read array of bytes and broadcast to all nodes
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

protected:
  // The universal data-reader. All the read functions call this
  template< typename T>
  void
  readPrimitive(T& output);

  // Get the internal ostream
  std::istream& getIstream() {return f;}

private:
  QDPUtil::RemoteInputFileStream f;
// ifstream f;
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

//! Read a binary multi1d element
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

//! Read a fixed number of binary multi1d element - no element count expected
template<class T>
inline
void read(BinaryReader& bin, multi1d<T>& d, int num)
{
  for(int i=0; i < num; ++i)
    read(bin, d[i]);
}


//! Simple output binary class
class BinaryWriter
{
public:
  BinaryWriter();
  ~BinaryWriter();
  explicit BinaryWriter(const std::string& p);

  bool is_open();
  void open(const std::string& p);
  void close();

  //! Flush the buffer
  void flush();

  //! Return true if some failure occurred in previous IO operation
  bool fail();

  // Basic write function
  void writeArray(const char* output, size_t nbytes, size_t nmemb);

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
  // The universal data-write. All the write functions call this
  template< typename T>
  void
  writePrimitive(const T& output);

  // Get the internal ostream
  std::ostream& getOstream() {return f;}

private:
  QDPUtil::RemoteOutputFileStream f;
// ofstream f;
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

//! Write a binary multi1d element
template<class T>
inline
void write(BinaryWriter& bin, const multi1d<T>& d)
{
  write(bin, d.size());    // always write the size
  for(int i=0; i < d.size(); ++i)
    write(bin, d[i]);
}

//! Write a fixed number of binary multi1d element - no element count written
template<class T>
inline
void write(BinaryWriter& bin, const multi1d<T>& d, int num)
{
  for(int i=0; i < num; ++i)
    write(bin, d[i]);
}

/*! @} */   // end of group io
QDP_END_NAMESPACE();
