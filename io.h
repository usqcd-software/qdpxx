// -*- C++ -*-
// $Id: io.h,v 1.1 2002-09-12 18:22:16 edwards Exp $
//
// QDP data parallel interface
//
// IO support

QDP_BEGIN_NAMESPACE(QDP);

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
