// -*- C++ -*-
// $Id: io.h,v 1.14 2003-04-23 04:46:02 edwards Exp $

/*! @file
 * @brief IO support
 */

#include <string>
#include <fstream>
#include <sstream>

#include "qcd-nml.h"

QDP_BEGIN_NAMESPACE(QDP);

using std::string;

// Useful prototypes
size_t bfread(void *ptr, size_t size, size_t nmemb, FILE *stream);
size_t bfwrite(void *ptr, size_t size, size_t nmemb, FILE *stream);


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

  std::ifstream& get() {return f;}

private:
  bool iop;
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

  std::ofstream& get() {return f;}

private:
  bool iop;
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


//! Namelist reader class
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

  //! Push a namelist group 
  NmlReader& push(const string& s);

  //! Pop a namelist group
  NmlReader& pop();

private:
  int stack_cnt;
  bool iop;
  section *abs;    // Abstract - holds parse tree
};

//! Push a namelist group 
NmlReader& push(NmlReader& nml, const string& s);

//! Pop a namelist group
NmlReader& pop(NmlReader& nml);

//! Function overload read of  int
NmlReader& read(NmlReader& nml, const string& s, int& d);

//! Function overload read of  float
NmlReader& read(NmlReader& nml, const string& s, float& d);

//! Function overload read of  double
NmlReader& read(NmlReader& nml, const string& s, double& d);

//! Read a namelist multi1d element
template<class T>
inline
NmlReader& read(NmlReader& nml, const string& s, multi1d<T>& s1)
{
  for(int i=0; i < s1.size(); ++i)
    read(nml, s, s1[i]);

  return nml;
}

//! Read a namelist multi2d element
template<class T> 
inline
NmlReader& read(NmlReader& nml, const string& s, multi2d<T>& s1)
{
  for(int j=0; j < s1.size1(); ++j)
    for(int i=0; i < s1.size2(); ++i)
      read(nml, s, s1[i][j]);

  return nml;
}

#define READ_NAMELIST(nml,a) read(nml,#a,a)
#define Read(nml,a) read(nml,#a,a)




//-----------------------------------------
// namelist writer support
//! Simple output namelist class
class NmlWriter
{
public:
  NmlWriter();
  ~NmlWriter();
  explicit NmlWriter(const char* p);

  //! Open file
  void open(const char* p);

  //! Close file
  void close();
  bool is_open();

  //! Push a namelist group 
  NmlWriter& push(const string& s);

  //! Pop a namelist group
  NmlWriter& pop();

  std::ofstream& get() {return f;}

private:
  int stack_cnt;
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
inline
NmlWriter& write(NmlWriter& nml, const string& s, const T& d)
{
  if (Layout::primaryNode()) 
    nml.get() << " " << s << " = " << d << " ,\n";
  return nml;
}

//! Write an outer scalar namelist element
/*! The second arg is the string for the variable name */
template<class T>
inline
NmlWriter& write(NmlWriter& nml, const string& s, const OScalar<T>& d)
{
  if (Layout::primaryNode()) 
    nml.get() << " " << s << " = ";

  nml << d; 
  return nml;
}

//! Write an outer lattice namelist element
/*! The second arg is the string for the variable name */
template<class T>
inline
NmlWriter& write(NmlWriter& nml, const string& s, const OLattice<T>& d)
{
  if (Layout::primaryNode()) 
    nml.get() << " " << s << " = ";

  nml << d; 
  return nml;
}

//! Write a namelist multi1d element
template<class T>
inline
NmlWriter& write(NmlWriter& nml, const string& s, const multi1d<T>& s1)
{
  for(int i=0; i < s1.size(); ++i)
  {
    std::ostringstream ost;
    if (Layout::primaryNode()) 
      ost << s << "[ " << i << " ]";
    write(nml, ost.str(), s1[i]);
  }
  return nml;
}

//! Write a namelist multi2d element
template<class T> 
inline
NmlWriter& write(NmlWriter& nml, const string& s, const multi2d<T>& s1)
{
  for(int j=0; j < s1.size1(); ++j)
    for(int i=0; i < s1.size2(); ++i)
    {
      std::ostringstream ost;
      if (Layout::primaryNode()) 
	ost << s << "[ " << i << " ][ " << j << " ]";
      write(nml, ost.str(), s1[i][j]);
    }
  return nml;
}

#define WRITE_NAMELIST(nml,a) write(nml,#a,a)
#define Write(nml,a) write(nml,#a,a)


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

  FILE* get() {return f;}

private:
  // I would like to use a stream, but at this moment not positive
  // of the interplay of stream's, streambuf's, and C file-desc.
  // So, just use a C filedesc.
//  std::ofstream f;
  FILE* f;
  bool iop;
};


// Read a binary element
// BinaryReader& read(BinaryReader& bin, T& d)
/* See code in architecture specific section */


//! Read a binary multi1d element
template<class T>
inline
BinaryReader& read(BinaryReader& bin, multi1d<T>& d)
{
  for(int i=0; i < d.size(); ++i)
    read(bin, d[i]);

  return bin;
}

//! Read a binary multi2d element
template<class T>
inline
BinaryReader& read(BinaryReader& bin, multi2d<T>& d)
{
  for(int j=0; j < d.size2(); ++j)
    for(int i=0; i < d.size1(); ++i)
      read(bin, d[j][i]);

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
inline
BinaryWriter& write(BinaryWriter& bin, const T& d)
{
  if (Layout::primaryNode()) 
    bfwrite((void *)&d, sizeof(T), 1, bin.get()); 

  return bin;
}

//! Read a binary multi1d element
template<class T>
inline
BinaryWriter& write(BinaryWriter& bin, const multi1d<T>& d)
{
  for(int i=0; i < d.size(); ++i)
    write(bin, d[i]);

  return bin;
}

//! Read a binary multi2d element
template<class T>
inline
BinaryWriter& write(BinaryWriter& bin, const multi2d<T>& d)
{
  for(int j=0; j < d.size2(); ++j)
    for(int i=0; i < d.size1(); ++i)
      write(bin, d[j][i]);

  return bin;
}

/*! @} */   // end of group io
QDP_END_NAMESPACE();
