// -*- C++ -*-
// $Id: qdp_stream_io.h,v 1.1 2007-06-10 14:32:09 edwards Exp $
/*! @file
 * @brief Stream IO support. Read/write QDP types to an object held in memory
 */

#ifndef QDP_STREAM_IO_H
#define QDP_STREAM_IO_H

#include <string>
#include <fstream>
#include <sstream>
#include "qdp_byteorder.h"

namespace QDP
{

  /*! @defgroup io IO
   *
   * Stream input and output operations on QDP types
   *
   * @{
   */

  //--------------------------------------------------------------------------------
  //!  Binary input class
  /*!
    This class is used to read data from a binary object. The data in the object
    is assumed to be big-endian. If the host nachine is little-endian, the data
    is byte-swapped. All nodes end up with the same data
  
    The read methods are also wrapped by externally defined functions
    and >> operators,   
  */
  class BinaryBufferReader
  {
  public:
    BinaryBufferReader();

    /*!
      Destroy the object
    */
    ~BinaryBufferReader();

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

    //! Internal storage
    std::istringstream f;
  };

  // Telephone book of basic primitives
  void read(BinaryBufferReader& bin, std::string& input, size_t maxBytes);
  void read(BinaryBufferReader& bin, char& input);
  void read(BinaryBufferReader& bin, int& input);
  void read(BinaryBufferReader& bin, unsigned int& input);
  void read(BinaryBufferReader& bin, short int& input);
  void read(BinaryBufferReader& bin, unsigned short int& input);
  void read(BinaryBufferReader& bin, long int& input);
  void read(BinaryBufferReader& bin, unsigned long int& input);
  void read(BinaryBufferReader& bin, float& input);
  void read(BinaryBufferReader& bin, double& input);
  void read(BinaryBufferReader& bin, bool& input);

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
  void read(BinaryBufferReader& bin, multi1d<T>& d)
  {
    int n;
    read(bin, n);    // the size is always written, even if 0
    d.resize(n);

    for(int i=0; i < d.size(); ++i)
      read(bin, d[i]);
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
  void read(BinaryBufferReader& bin, multi2d<T>& d)
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
  void read(BinaryBufferReader& bin, multi3d<T>& d)
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

  // Different bindings for same operators
  BinaryBufferReader& operator>>(BinaryBufferReader& bin, char& input);
  BinaryBufferReader& operator>>(BinaryBufferReader& bin, int& input);
  BinaryBufferReader& operator>>(BinaryBufferReader& bin, unsigned int& input);
  BinaryBufferReader& operator>>(BinaryBufferReader& bin, short int& input);
  BinaryBufferReader& operator>>(BinaryBufferReader& bin, unsigned short int& input);
  BinaryBufferReader& operator>>(BinaryBufferReader& bin, long int& input);
  BinaryBufferReader& operator>>(BinaryBufferReader& bin, unsigned long int& input);
  BinaryBufferReader& operator>>(BinaryBufferReader& bin, float& input);
  BinaryBufferReader& operator>>(BinaryBufferReader& bin, double& input);
  BinaryBufferReader& operator>>(BinaryBufferReader& bin, bool& input);

  //----------------------------------------------------------------------------------------
  //!  Binary output class
  /*!
    This class is used to write data to a binary file. The data in the file
    is big-endian. If the host nachine is little-endian, the data
    is byte-swapped.   Output is done from the primary node only.

    Files need to be opened before any of the write methods are used  
  
    The write methods are also wrapped by externally defined functions
    and << operators,   
  */
  class BinaryBufferWriter
  {
  public:
    BinaryBufferWriter();

    /*!
      Closes the object
    */
    ~BinaryBufferWriter();

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
    std::ostringstream& getOstream() {return f;}

  private:
    //! Checksum
    QDPUtil::n_uint32_t checksum;

    //! Internal storage
    ostringstream f;
  };


  // Telephone book of basic primitives
  void write(BinaryBufferWriter& bin, const std::string& output);
  void write(BinaryBufferWriter& bin, const char* output);
  void write(BinaryBufferWriter& bin, char output);
  void write(BinaryBufferWriter& bin, int output);
  void write(BinaryBufferWriter& bin, unsigned int output);
  void write(BinaryBufferWriter& bin, short int output);
  void write(BinaryBufferWriter& bin, unsigned short int output);
  void write(BinaryBufferWriter& bin, long int output);
  void write(BinaryBufferWriter& bin, unsigned long int output);
  void write(BinaryBufferWriter& bin, float output);
  void write(BinaryBufferWriter& bin, double output);
  void write(BinaryBufferWriter& bin, bool output);

  //! Write all of a binary multi1d object
  /*!
    This also writes the number of elements to the file.
    \param bin The initialised binary reader
    \param d The data to be filled.

    \pre The binary reader must have opened the file.
  */
  template<class T>
  inline
  void write(BinaryBufferWriter& bin, const multi1d<T>& d)
  {
    write(bin, d.size());    // always write the size
    for(int i=0; i < d.size(); ++i)
      write(bin, d[i]);
  }


  //! Write a binary multi2d element
  template<class T>
  inline
  void write(BinaryBufferWriter& bin, const multi2d<T>& d)
  {
    write(bin, d.size2());    // always write the size
    write(bin, d.size1());    // always write the size

    for(int i=0; i < d.size1(); ++i)
      for(int j=0; j < d.size2(); ++j)
      {
	write(bin, d[j][i]);
      }

  }


  //! Write a binary multi2d element
  template<class T>
  inline
  void write(BinaryBufferWriter& bin, const multi3d<T>& d)
  {
    write(bin, d.size3());    // always write the size
    write(bin, d.size2());    // always write the size
    write(bin, d.size1());    // always write the size

    for(int i=0; i < d.size1(); ++i)
      for(int j=0; j < d.size2(); ++j)
	for(int k=0; k < d.size3(); ++k)
	  write(bin, d[k][j][i]);

  }


  // Different bindings for same operators
  BinaryBufferWriter& operator<<(BinaryBufferWriter& bin, const std::string& output);
  BinaryBufferWriter& operator<<(BinaryBufferWriter& bin, const char* output);
  BinaryBufferWriter& operator<<(BinaryBufferWriter& bin, char output);
  BinaryBufferWriter& operator<<(BinaryBufferWriter& bin, int output);
  BinaryBufferWriter& operator<<(BinaryBufferWriter& bin, unsigned int output);
  BinaryBufferWriter& operator<<(BinaryBufferWriter& bin, short int output);
  BinaryBufferWriter& operator<<(BinaryBufferWriter& bin, unsigned short int output);
  BinaryBufferWriter& operator<<(BinaryBufferWriter& bin, long int output);
  BinaryBufferWriter& operator<<(BinaryBufferWriter& bin, unsigned long int output);
  BinaryBufferWriter& operator<<(BinaryBufferWriter& bin, float output);
  BinaryBufferWriter& operator<<(BinaryBufferWriter& bin, double output);
  BinaryBufferWriter& operator<<(BinaryBufferWriter& bin, bool output);

  template<class T>
  inline
  BinaryBufferWriter& operator<<(BinaryBufferWriter& bin, const multi1d<T>& output)
  {
    write(bin, output);
    return bin;
  }

  template<class T>
  inline
  BinaryBufferWriter& operator<<(BinaryBufferWriter& bin, const multi2d<T>& output)
  {
    write(bin, output);
    return bin;
  }

  template<class T>
  inline
  BinaryBufferWriter& operator<<(BinaryBufferWriter& bin, const multi3d<T>& output)
  {
    write(bin, output);
    return bin;
  }
}  // namespace QDP

#endif
