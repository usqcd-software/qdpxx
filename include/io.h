// -*- C++ -*-
// $Id: io.h,v 1.5 2002-10-12 04:10:15 edwards Exp $

/*! @file
 * @brief IO support
 */

#include <string>
#include <fstream>
#include <sstream>

QDP_BEGIN_NAMESPACE(QDP);

using std::string;

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
  explicit TextReader(const char* p);

  //! Open file
  void open(const char* p);

  //! Close file
  void close();
  bool is_open();

  //! Read from stream
  template<class T>
  TextReader& operator>>(T& d) {f >> d; return *this;}


private:
  std::ifstream f;
};



//! Simple output text class
class TextWriter
{
public:
  TextWriter();
  ~TextWriter();
  explicit TextWriter(const char* p);

  bool is_open();
  void open(const char* p);
  void close();

  //! Write to stream
  template<class T>
  TextWriter& operator<<(const T& d) {f << d; return *this;}

private:
  std::ofstream f;
};



//--------------------------------------------------------------------------------
//! Namelist constructor object
#if 0
// This does not work the way I like...
/*! 
 * This ties together a name and a ref which can be read or written
 * via namelist
 */
template<class T>
class Nml
{
public:
  Nml(const char* s, const T& d): name(s), obj(d) {}

  const char* name;
  const T&  obj;
};
#endif


//! Simple input namelist class
class NmlReader
{
public:
  NmlReader();
  ~NmlReader();
  explicit NmlReader(const char* p);

  //! Open file
  void open(const char* p);

  //! Close file
  void close();
  bool is_open();

  //! Read from stream
  template<class T>
  NmlReader& operator>>(T& d) {f >> d; return *this;}


private:
  std::ifstream f;
};



//! Simple output namelist class
class NmlWriter
{
public:
  NmlWriter();
  ~NmlWriter();
  explicit NmlWriter(const char* p);

  bool is_open();
  void open(const char* p);
  void close();

  //! Write to stream
//  template<class T>
//  NmlWriter& operator<<(const T& d) {f << d; return *this;}

//  template<class T>
//  friend NmlWriter& operator<<(NmlWriter& s, const T& d);

  std::ofstream& get() {return f;}

private:
  bool iop;
  std::ofstream f;
};


//! Push a namelist group 
NmlWriter& push(NmlWriter& nml, const string& s);

//! Pop a namelist group
NmlWriter& pop(NmlWriter& nml);

//! Write a comment
NmlWriter& operator<<(NmlWriter& nml, const char* s);

//! Write a namelist element
template<class T>
NmlWriter& write(NmlWriter& nml, const string& s, const T& d)
{
  nml.get() << " " << s << " = " << d << ",\n";
  return nml;
}

//! Write an outer scalar namelist element
/*! The second arg is the string for the variable name */
template<class T>
NmlWriter& write(NmlWriter& nml, const string& s, const OScalar<T>& d)
{
  nml.get() << " " << s << " = ";
  nml << d; 
  return nml;
}

//! Write an outer lattice namelist element
/*! The second arg is the string for the variable name */
template<class T>
NmlWriter& write(NmlWriter& nml, const string& s, const OLattice<T>& d)
{
  nml.get() << " " << s << " = ";
  nml << d; 
  return nml;
}

//! Write a namelist multi1d element
template<class T>
NmlWriter& write(NmlWriter& nml, const string& s, const multi1d<T>& s1)
{
  for(int i=0; i < s1.size(); ++i)
  {
    std::ostringstream ost;
    ost << s << "[" << i << "]";
    write(nml, ost.str(), s1[i]);
  }
  return nml;
}

//! Write a namelist multi2d element
template<class T> 
NmlWriter& write(NmlWriter& nml, const string& s, const multi2d<T>& s1)
{
  for(int j=0; j < s1.size1(); ++j)
    for(int i=0; i < s1.size2(); ++i)
    {
      std::ostringstream ost;
      ost << s << "[" << i << "][" << j << "]";
      write(nml, ost.str(), s1[i][j]);
    }
  return nml;
}

#define WRITE_NAMELIST(nml,a) write(nml,#a,a)
#define Write(nml,a) write(nml,#a,a)


//------------------------------------------------
#if 0
//! Namelist style input object
class nml_obj
{
private:
  istream foo;
};

//! Read a namelist group
template<class T>
NmlWriter& read(NmlWriter& nml, const string& s, const QDPType<T>& s1);
#endif




//--------------------------------------------------------------------------------
//! Simple output binary class
class BinaryReader
{
public:
  BinaryReader();
  ~BinaryReader();
  explicit BinaryReader(const char* p);

  bool is_open();
  void open(const char* p);
  void close();

//  //! Binary writer
//  template<class T>
//  BinaryReader& read(const T& d) {fread((void *)&d,sizeof(T),1,f); return *this;}

  //! Read End-Of-Record mark
  BinaryReader& eor();

  FILE* get() {return f;}

private:
  // I would like to use a stream, but at this moment not positive
  // of the interplay of stream's, streambuf's, and C file-desc.
  // So, just use a C filedesc.
//  std::ofstream f;
  FILE* f;
  bool iop;
};


//! Read a binary element
template<class T>
BinaryReader& read(BinaryReader& bin, const T& d)
{
  fread((void *)&d, sizeof(T), 1, bin.get()); 
  return bin;
}



//! Simple output binary class
class BinaryWriter
{
public:
  BinaryWriter();
  ~BinaryWriter();
  explicit BinaryWriter(const char* p);

  bool is_open();
  void open(const char* p);
  void close();

  //! Binary writer
//  template<class T>
//  BinaryWriter& write(const T& d) {fwrite((void *)&d,sizeof(T),1,f); return *this;}

  //! Write End-Of-Record mark
  BinaryWriter& eor();

  FILE* get() {return f;}

private:
  // I would like to use a stream, but at this moment not positive
  // of the interplay of stream's, streambuf's, and C file-desc.
  // So, just use a C filedesc.
//  std::ofstream f;
  FILE* f;
  bool iop;
};

//! Write a binary element
template<class T>
BinaryWriter& write(BinaryWriter& bin, const T& d)
{
  fwrite((void *)&d, sizeof(T), 1, bin.get()); 
  return bin;
}

/*! @} */   // end of group io
QDP_END_NAMESPACE();
