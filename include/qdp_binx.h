// -*- C++ -*-
// $Id: qdp_binx.h,v 1.4 2004-03-27 20:43:44 mcneile Exp $

/*! @file
 * @brief IO support
 */
#ifndef QDP_BINX_INC
#define QDP_BINX_INC

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

//! Simple output binary class
class BinxWriter
{
public:
  BinxWriter();
  ~BinxWriter();
  explicit BinxWriter(const std::string& p);

  bool is_open();
  void open(const std::string& p);
  void close();

  //! Flush the buffer
  void flush();

  //! Return true if some failure occurred in previous IO operation
  bool fail();

  //! switch on and off xml writing
  void no_xml() { write_xml = false ; } 
  void yes_xml() { write_xml = true ; } 

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

  void write_1D_header(const int& output,const int& dim) ;

protected:


private:
  XMLFileWriter *toxml;
  BinaryWriter *tobinary ;
  bool write_xml ; 

};


// Telephone book of basic primitives
void write(BinxWriter& bin, const std::string& output);
void write(BinxWriter& bin, const char* output);
void write(BinxWriter& bin, char output);
void write(BinxWriter& bin, int output);
void write(BinxWriter& bin, unsigned int output);
void write(BinxWriter& bin, short int output);
void write(BinxWriter& bin, unsigned short int output);
void write(BinxWriter& bin, long int output);
void write(BinxWriter& bin, unsigned long int output);
void write(BinxWriter& bin, float output);
void write(BinxWriter& bin, double output);
void write(BinxWriter& bin, bool output);

// Different bindings for same operators
BinxWriter& operator<<(BinxWriter& bin, const std::string& output);
BinxWriter& operator<<(BinxWriter& bin, const char* output);
BinxWriter& operator<<(BinxWriter& bin, char output);
BinxWriter& operator<<(BinxWriter& bin, int output);
BinxWriter& operator<<(BinxWriter& bin, unsigned int output);
BinxWriter& operator<<(BinxWriter& bin, short int output);
BinxWriter& operator<<(BinxWriter& bin, unsigned short int output);
BinxWriter& operator<<(BinxWriter& bin, long int output);
BinxWriter& operator<<(BinxWriter& bin, unsigned long int output);
BinxWriter& operator<<(BinxWriter& bin, float output);
BinxWriter& operator<<(BinxWriter& bin, double output);
BinxWriter& operator<<(BinxWriter& bin, bool output);

//! Write a binary multi1d element
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
QDP_END_NAMESPACE();


#endif
