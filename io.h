// -*- C++ -*-
// $Id: io.h,v 1.2 2002-10-01 16:24:41 edwards Exp $
//
// QDP data parallel interface
//
// IO support

#include <fstream>

QDP_BEGIN_NAMESPACE(QDP);

//! Simple input text class
class TextReader
{
public:
  TextReader();
  ~TextReader();
  explicit TextReader(const char* p);

  void open(const char* p);
  void close();

  bool is_open();

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

  template<class T>
  TextWriter& operator<<(const T& d) {f << d; return *this;}

private:
  std::ofstream f;
};



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

#if 0
  // Binary writer
  template<class T>
  BinaryWriter& write(const T& d) {fwrite(f,d);; return *this;}
#endif

  // Ascii writer
  template<class T>
  BinaryWriter& operator<<(const T& d) {f << d; return *this;}

private:
  std::ofstream f;
};



//----------------------------------------------------------------
//! Namelist style reading/writing

  /*! Push a namelist group */
inline ostream& Push(ostream& nml, const char *s) {nml << "&" << s << "\n"; return nml;}

/*! Pop a namelist group */
inline ostream& Pop(ostream& nml) {nml << "&END\n"; return nml;}

/*! Write a namelist element */
template<class T> inline
ostream& Write(ostream& nml, const char *s, const T& s1) 
{nml << " " << s << " = " << s1 << "\n"; return nml;}

/*! Write a namelist multi1d element */
template<class T> inline
ostream& Write(ostream& nml, const char *s, const multi1d<T>& s1)
{
  for(int i=0; i < s1.size(); ++i)
    nml << " " << s << "[" << i << "] = " << s1[i] << "\n";
  return nml;
}

#define WRITE_NAMELIST(nml,a) Write(nml,#a,a)


/*! Write a namelist multi2d element */
template<class T> inline
ostream& Write(ostream& nml, const char *s, const multi2d<T>& s1)
{
  for(int j=0; j < s1.size1(); ++j)
    for(int i=0; i < s1.size2(); ++i)
      nml << " " << s << "[" << i << "][" << j << "] = " << s1[i][j] << "\n";
  return nml;
}

#if 0
/*! Namelist style input object */
class nml_obj
{
private:
  istream foo;
};

/*! Read a namelist group */
template<class T> inline
ostream& read(ostream& nml, const char *s, const QDPType<T>& s1);
#endif


QDP_END_NAMESPACE();
