// -*- C++ -*-
// $Id: qdp_io.h,v 1.6 2003-06-05 04:15:55 edwards Exp $

/*! @file
 * @brief IO support
 */

#include <string>
#include <fstream>
#include <sstream>

#include "qcd-nml.h"

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

  std::ifstream& get() {return f;}

private:
  bool iop;
  std::ifstream f;
};


// Different bindings for same operators
TextReader& operator>>(TextReader& bin, char& input);
TextReader& operator>>(TextReader& bin, int& input);
TextReader& operator>>(TextReader& bin, unsigned int& input);
TextReader& operator>>(TextReader& bin, short int& input);
TextReader& operator>>(TextReader& bin, unsigned short int& input);
TextReader& operator>>(TextReader& bin, long int& input);
TextReader& operator>>(TextReader& bin, unsigned long int& input);
TextReader& operator>>(TextReader& bin, float& input);
TextReader& operator>>(TextReader& bin, double& input);
TextReader& operator>>(TextReader& bin, bool& input);


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

  std::ofstream& get() {return f;}

private:
  bool iop;
  std::ofstream f;
};


// Different bindings for same operators
TextWriter& operator<<(TextWriter& bin, const std::string& output);
TextWriter& operator<<(TextWriter& bin, const char& output);
TextWriter& operator<<(TextWriter& bin, const int& output);
TextWriter& operator<<(TextWriter& bin, const unsigned int& output);
TextWriter& operator<<(TextWriter& bin, const short int& output);
TextWriter& operator<<(TextWriter& bin, const unsigned short int& output);
TextWriter& operator<<(TextWriter& bin, const long int& output);
TextWriter& operator<<(TextWriter& bin, const unsigned long int& output);
TextWriter& operator<<(TextWriter& bin, const float& output);
TextWriter& operator<<(TextWriter& bin, const double& output);
TextWriter& operator<<(TextWriter& bin, const bool& output);



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
  Nml(const std::string& s, const T& d): name(s), obj(d) {}

  const std::string& name;
  const T&  obj;
};
#endif


//! Namelist reader class
class NmlReader
{
public:
  NmlReader();
  ~NmlReader();
  explicit NmlReader(const std::string& p);

  //! Open file
  void open(const std::string& p);

  //! Close file
  void close();
  bool is_open();

  //! Push a namelist group 
  void push(const std::string& s);

  //! Pop a namelist group
  void pop();

private:
  int stack_cnt;
  bool iop;
  section *abs;    // Abstract - holds parse tree
};

//! Push a namelist group 
void push(NmlReader& nml, const std::string& s);

//! Pop a namelist group
void pop(NmlReader& nml);

//! Function overload read of  int
void read(NmlReader& nml, const std::string& s, int& d);

//! Function overload read of  float
void read(NmlReader& nml, const std::string& s, float& d);

//! Function overload read of  double
void read(NmlReader& nml, const std::string& s, double& d);

//! Function overload read of  bool
void read(NmlReader& nml, const std::string& s, bool& d);

//! Function overload read of  std::string
void read(NmlReader& nml, const std::string& s, std::string& d);

//! Function overload read of  int  into element position n
void read(NmlReader& nml, const std::string& s, int& d, int n);

//! Function overload read of  float  into element position n
void read(NmlReader& nml, const std::string& s, float& d, int n);

//! Function overload read of  double  into element position n
void read(NmlReader& nml, const std::string& s, double& d, int n);

//! Function overload read of  bool  into element position n
void read(NmlReader& nml, const std::string& s, bool& d, int n);

//! Function overload read of  multi1d<int>
void read(NmlReader& nml, const std::string& s, multi1d<int>& d);

//! Function overload read of  multi1d<float>
void read(NmlReader& nml, const std::string& s, multi1d<float>& d);

//! Function overload read of  multi1d<double>
void read(NmlReader& nml, const std::string& s, multi1d<double>& d);

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
  explicit NmlWriter(const std::string& p);

  //! Open file
  void open(const std::string& p);

  //! Close file
  void close();
  bool is_open();

  //! Push a namelist group 
  void push(const std::string& s);

  //! Pop a namelist group
  void pop();

  std::ofstream& get() {return f;}

private:
  int stack_cnt;
  bool iop;
  std::ofstream f;
};


//! Push a namelist group 
void push(NmlWriter& nml, const std::string& s);

//! Pop a namelist group
void pop(NmlWriter& nml);

//! Write a comment
NmlWriter& operator<<(NmlWriter& nml, const std::string& s);
NmlWriter& operator<<(NmlWriter& nml, const char* s);

// Primitive writers
void write(NmlWriter& nml, const std::string& s, const std::string& d);
void write(NmlWriter& nml, const std::string& s, const int& d);
void write(NmlWriter& nml, const std::string& s, const unsigned int& d);
void write(NmlWriter& nml, const std::string& s, const float& d);
void write(NmlWriter& nml, const std::string& s, const double& d);
void write(NmlWriter& nml, const std::string& s, const bool& d);

//! Write an outer scalar namelist element
/*! The second arg is the string for the variable name */
template<class T>
inline
void write(NmlWriter& nml, const std::string& s, const OScalar<T>& d)
{
  if (Layout::primaryNode()) 
    nml.get() << " " << s << " = ";

  nml << d; 
}

//! Write an outer lattice namelist element
/*! The second arg is the string for the variable name */
template<class T>
inline
void write(NmlWriter& nml, const std::string& s, const OLattice<T>& d)
{
  if (Layout::primaryNode()) 
    nml.get() << " " << s << " = ";

  nml << d; 
}

//! Write a namelist multi1d element
template<class T>
inline
void write(NmlWriter& nml, const std::string& s, const multi1d<T>& s1)
{
  for(int i=0; i < s1.size(); ++i)
  {
    std::ostringstream ost;
    if (Layout::primaryNode()) 
      ost << s << "[ " << i << " ]";
    write(nml, ost.str(), s1[i]);
  }
}

//! Write a namelist multi2d element
template<class T> 
inline
void write(NmlWriter& nml, const std::string& s, const multi2d<T>& s1)
{
  for(int j=0; j < s1.size1(); ++j)
    for(int i=0; i < s1.size2(); ++i)
    {
      std::ostringstream ost;
      if (Layout::primaryNode()) 
	ost << s << "[ " << i << " ][ " << j << " ]";
      write(nml, ost.str(), s1[i][j]);
    }
}

// Different bindings for same operators
NmlWriter& operator<<(NmlWriter& nml, const float& d);
NmlWriter& operator<<(NmlWriter& nml, const int& d);
NmlWriter& operator<<(NmlWriter& nml, const unsigned int& d);
NmlWriter& operator<<(NmlWriter& nml, const double& d);
NmlWriter& operator<<(NmlWriter& nml, const bool& d);


#define WRITE_NAMELIST(nml,a) write(nml,#a,a)
#define Write(nml,a) write(nml,#a,a)


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
  std::ifstream f;
  bool iop;
};

// Telephone book of basic primitives
void read(BinaryReader& bin, std::string& output, size_t maxBytes);
void read(BinaryReader& bin, char& output);
void read(BinaryReader& bin, int& output);
void read(BinaryReader& bin, unsigned int& output);
void read(BinaryReader& bin, short int& output);
void read(BinaryReader& bin, unsigned short int& output);
void read(BinaryReader& bin, long int& output);
void read(BinaryReader& bin, unsigned long int& output);
void read(BinaryReader& bin, float& output);
void read(BinaryReader& bin, double& output);
void read(BinaryReader& bin, bool& output);

// Different bindings for same operators
BinaryReader& operator>>(BinaryReader& bin, char& output);
BinaryReader& operator>>(BinaryReader& bin, int& output);
BinaryReader& operator>>(BinaryReader& bin, unsigned int& output);
BinaryReader& operator>>(BinaryReader& bin, short int& output);
BinaryReader& operator>>(BinaryReader& bin, unsigned short int& output);
BinaryReader& operator>>(BinaryReader& bin, long int& output);
BinaryReader& operator>>(BinaryReader& bin, unsigned long int& output);
BinaryReader& operator>>(BinaryReader& bin, float& output);
BinaryReader& operator>>(BinaryReader& bin, double& output);
BinaryReader& operator>>(BinaryReader& bin, bool& output);

//! Read a binary multi1d element
template<class T>
inline
void read(BinaryReader& bin, multi1d<T>& d)
{
  for(int i=0; i < d.size(); ++i)
    read(bin, d[i]);
}

//! Read a binary multi2d element
template<class T>
inline
void read(BinaryReader& bin, multi2d<T>& d)
{
  for(int j=0; j < d.size2(); ++j)
    for(int i=0; i < d.size1(); ++i)
      read(bin, d[j][i]);
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

  // Basic write function
  void writeArray(const char* output, size_t nbytes, size_t nmemb);

  // Overloaded Writer Functions
  void write(const std::string& output);
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
  std::ofstream f;
  bool iop;
};


// Telephone book of basic primitives
void write(BinaryWriter& bin, const std::string& output);
void write(BinaryWriter& bin, const char& output);
void write(BinaryWriter& bin, const int& output);
void write(BinaryWriter& bin, const unsigned int& output);
void write(BinaryWriter& bin, const short int& output);
void write(BinaryWriter& bin, const unsigned short int& output);
void write(BinaryWriter& bin, const long int& output);
void write(BinaryWriter& bin, const unsigned long int& output);
void write(BinaryWriter& bin, const float& output);
void write(BinaryWriter& bin, const double& output);
void write(BinaryWriter& bin, const bool& output);

// Different bindings for same operators
BinaryWriter& operator<<(BinaryWriter& bin, const std::string& output);
BinaryWriter& operator<<(BinaryWriter& bin, const char& output);
BinaryWriter& operator<<(BinaryWriter& bin, const int& output);
BinaryWriter& operator<<(BinaryWriter& bin, const unsigned int& output);
BinaryWriter& operator<<(BinaryWriter& bin, const short int& output);
BinaryWriter& operator<<(BinaryWriter& bin, const unsigned short int& output);
BinaryWriter& operator<<(BinaryWriter& bin, const long int& output);
BinaryWriter& operator<<(BinaryWriter& bin, const unsigned long int& output);
BinaryWriter& operator<<(BinaryWriter& bin, const float& output);
BinaryWriter& operator<<(BinaryWriter& bin, const double& output);
BinaryWriter& operator<<(BinaryWriter& bin, const bool& output);

//! Read a binary multi1d element
template<class T>
inline
void write(BinaryWriter& bin, const multi1d<T>& d)
{
  for(int i=0; i < d.size(); ++i)
    write(bin, d[i]);
}

//! Read a binary multi2d element
template<class T>
inline
void write(BinaryWriter& bin, const multi2d<T>& d)
{
  for(int j=0; j < d.size2(); ++j)
    for(int i=0; i < d.size1(); ++i)
      write(bin, d[j][i]);
}

/*! @} */   // end of group io
QDP_END_NAMESPACE();
