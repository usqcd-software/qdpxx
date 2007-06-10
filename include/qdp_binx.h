// -*- C++ -*-
// $Id: qdp_binx.h,v 1.6 2007-06-10 14:32:08 edwards Exp $

/*! @file
 * @brief IO support with BinX 
 */
#ifndef QDP_BINX_INC
#define QDP_BINX_INC

#include <string>
#include <fstream>
#include <sstream>


namespace QDP
{
  /*! @ingroup io
   *
   * File input and output operations on QDP types
   *
   * @{
   */

  //! Class to write binary data and BinX markup.
  /*!
    This writes binary data to a file using a standard BinaryFileWriter
    and also can write the corresponding BinX markup to an XML file.
    The binary data in the file
    is big-endian. If the host nachine is little-endian, the data
    is byte-swapped.   Output is done from the primary node only.
    Not all fundamental data types get a BinX markup.  XML writing can be
    disabled.

    Files need to be opened before any of the write methods are used

    There are external functions and operators wrapping the write methods.
  */
  class BinxWriter
  {
  public:
    BinxWriter();

    /*!
      Closes the last binary and BinX files opened.
    */
    ~BinxWriter();

    /*!
      Opens a files for binary writing and an BinX file with the same name
      but with \c _binx.xml appended.
      \param p  The filename stem.

      \post XML writing is enabled.
    */
    explicit BinxWriter(const std::string& p);

    //! Queries whether the binary file is open
    /*!
      \return true if the binary file is open; false otherwise.
    */
    bool is_open();

    /*!
      Opens a files for binary writing and an BinX file with the same name
      but with \c _binx.xml appended.
      \param p  The filename stem.

      \post XML writing is enabled.      
    */
    void open(const std::string& p);

    //! Closes the last binary and BinX files opened.
    void close();

    //! Flush the buffer
    void flush();

    //!Checks status of the previous IO operation.
    /*!
      \return true if some failure occurred in previous IO operation
    */
    bool fail();

    //! Switch off XML writing
    void no_xml() { write_xml = false ; }

    //! Switch on XML writing    
    void yes_xml() { write_xml = true ; } 

    //! Basic data writing function
    /*!
      No BinX written.
      \param output Pointer to the data to write
      \param nbytes The number of bytes per datum
      \param nmemb  The number of data
    */
    void writeArray(const char* output, size_t nbytes, size_t nmemb);

    // Overloaded Writer Functions

    /*!
      Writes a string to the binary file.
      A newline is appended to the written string.
      BinX is also written if XML writing is enabled.
    */
    void write(const std::string& output);

    /*!
      Writes a string to the binary file.
      A newline is appended to the written string.      
      BinX is also written if XML writing is enabled.
    */
    void write(const char* output);

    /*!
      Writes a character to the binary file.
      BinX is also written if XML writing is enabled.
    */
    void write(const char& output);

    /*!
      Writes an integer to the binary file.
      BinX is also written if XML writing is enabled.
    */
    void write(const int& output);

    /*! Writes an integer to the binary file. No BinX is written. */
    void write(const unsigned int& output);

    /*! Writes an integer to the binary file. No BinX is written. */    
    void write(const short int& output);

    /*! Writes an integer to the binary file. No BinX is written. */    
    void write(const unsigned short int& output);

    /*! Writes an integer to the binary file. No BinX is written. */
    void write(const long int& output);

    /*! Writes an integer to the binary file. No BinX is written. */    
    void write(const unsigned long int& output);

    /*!
      Writes a floating-point number to the binary file.
      BinX is also written if XML writing is enabled.
    */
    void write(const float& output);

    /*!
      Writes a floating-point number to the binary file.
      BinX is also written if XML writing is enabled.
    */
    void write(const double& output);

    /*! Writes an integer to the binary file. No BinX is written. */
    void write(const bool& output);

    //! Writes  BinX markup for an array
    /*!
      This is written regardless of whether XML writing is enable.

      \param output Not used
      \param dim The size of the binary array

      \post XML writing is disabled.
    */
    void write_1D_header(const int& output,const int& dim) ;

  protected:


  private:
    XMLFileWriter *toxml;
    BinaryFileWriter *tobinary ;
    bool write_xml ; 

  };


  // Telephone book of basic primitives

  //! Wrapper for BinxWriter method. 
  void write(BinxWriter& bin, const std::string& output);
  //! Wrapper for BinxWriter method. 
  void write(BinxWriter& bin, const char* output);
  //! Wrapper for BinxWriter method. 
  void write(BinxWriter& bin, char output);
  //! Wrapper for BinxWriter method. 
  void write(BinxWriter& bin, int output);
  //! Wrapper for BinxWriter method. 
  void write(BinxWriter& bin, unsigned int output);
  //! Wrapper for BinxWriter method. 
  void write(BinxWriter& bin, short int output);
  //! Wrapper for BinxWriter method. 
  void write(BinxWriter& bin, unsigned short int output);
  //! Wrapper for BinxWriter method. 
  void write(BinxWriter& bin, long int output);
  //! Wrapper for BinxWriter method. 
  void write(BinxWriter& bin, unsigned long int output);
  //! Wrapper for BinxWriter method. 
  void write(BinxWriter& bin, float output);
  //! Wrapper for BinxWriter method. 
  void write(BinxWriter& bin, double output);
  //! Wrapper for BinxWriter method. 
  void write(BinxWriter& bin, bool output);

  // Different bindings for same operators
  //! Operator wrapper for BinxWriter method. 
  BinxWriter& operator<<(BinxWriter& bin, const std::string& output);
  //! Operator wrapper for BinxWriter method. 
  BinxWriter& operator<<(BinxWriter& bin, const char* output);
  //! Operator wrapper for BinxWriter method. 
  BinxWriter& operator<<(BinxWriter& bin, char output);
  //! Operator wrapper for BinxWriter method. 
  BinxWriter& operator<<(BinxWriter& bin, int output);
  //! Operator wrapper for BinxWriter method. 
  BinxWriter& operator<<(BinxWriter& bin, unsigned int output);
  //! Operator wrapper for BinxWriter method. 
  BinxWriter& operator<<(BinxWriter& bin, short int output);
  //! Operator wrapper for BinxWriter method. 
  BinxWriter& operator<<(BinxWriter& bin, unsigned short int output);
  //! Operator wrapper for BinxWriter method. 
  BinxWriter& operator<<(BinxWriter& bin, long int output);
  //! Operator wrapper for BinxWriter method. 
  BinxWriter& operator<<(BinxWriter& bin, unsigned long int output);
  //! Operator wrapper for BinxWriter method. 
  BinxWriter& operator<<(BinxWriter& bin, float output);
  //! Operator wrapper for BinxWriter method. 
  BinxWriter& operator<<(BinxWriter& bin, double output);
  //! Operator wrapper for BinxWriter method. 
  BinxWriter& operator<<(BinxWriter& bin, bool output);

  //! Write a binary multi1d element
  /*!
    This writes the size of the multi1d as well as the data. If XML writing
    is enabled, the BinX corresponding to the size datum is written.
    No BinX is written for the array data.

    \param bin The writer
    \param d The data
    \param num The number of data

    \pre The writer should have already opened its files  
    \post XML writing is enabled
  */
  template<class T>
  inline
  void write(BinxWriter& bin, const multi1d<T>& d)
  {
    write(bin, d.size());    // always write the size
    bin.no_xml() ;
    bin.write_1D_header(d[0],d.size()); 
    for(int i=0; i < d.size(); ++i)
      write(bin, d[i]);
    bin.yes_xml() ;

  }

  //! Write a fixed number of binary multi1d element - no element count written
  //! Write a binary multi1d element
  /*!
    No BinX is written for the array data.
  
    \param bin The writer
    \param d The data
    \param num The number of data

    \pre The writer should have already opened its files
    \post XML writing is enabled
  */
  template<class T>
  inline
  void write(BinxWriter& bin, const multi1d<T>& d, int num)
  {
    bin.no_xml() ;
    bin.write_1D_header(d[0],d.size()); 
    for(int i=0; i < num; ++i)
      write(bin, d[i]);
    bin.yes_xml() ;
  }

  /*! @} */   // end of group io
} // namespace QDP


#endif
