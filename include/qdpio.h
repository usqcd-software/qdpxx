// -*- C++ -*-
// $Id: qdpio.h,v 1.4 2003-04-30 21:09:17 edwards Exp $

/*! @file
 * @brief IO support via QIO
 */

#include "qio.h"

QDP_BEGIN_NAMESPACE(QDP);

/*! @defgroup qio QIO
 *
 * File input and output operations on QDP types
 *
 * @{
 */

// Forward declarations
class QDPSerialReader;
class QDPSerialWriter;


//--------------------------------------------------------------------------------
//! XML reader class
class XMLMetaReader
{
public:
  XMLMetaReader();
  ~XMLMetaReader();

  XML_MetaData* get() const {return xml;}

protected:
  XML_MetaData* xml;
};



//! XML Writer class
class XMLMetaWriter
{
public:
  XMLMetaWriter();
  ~XMLMetaWriter();

  XML_MetaData* get() const {return xml;}

protected:
  XML_MetaData* xml;
};



//--------------------------------------------------------------------------------
//! QIO class
class QDPSerialReader
{
public:
  QDPSerialReader();
  ~QDPSerialReader();

  //! Open file
  explicit QDPSerialReader(XMLMetaReader& xml, const std::string& p);

  //! Open file
  void open(XMLMetaReader& xml, const std::string& p);

  //! Close file
  void close();

  //! Is the file open?
  bool is_open();

  //! Read a QDP object
  template<class T, class C>
  void read(XMLMetaReader& xml, QDPType<T,C>& s1)
    {read_t(*this,xml,static_cast<C&>(s1));}

  //! Read an array of objects each in a seperate record
  /*! OOPS, what about xml? Is it repeatedly read but tossed out?? */
  template<class T>
  void read(XMLMetaReader& xml, multi1d<T>& s1)
    {
      for(int i=0; i < s1.size(); ++i)
	read(xml,s1[i]);
    }


  //! Read an array of objects all in a single record
  template<class T>
  void vread(XMLMetaReader& xml, multi1d<T>& s1) {}

  //! Check if end-of-file has been reached
  bool eof() const;

  //!  Check if an unrecoverable error has occurred
  bool bad() const;


  QIO_Reader *get() const {return qio_in;}

private:
  bool iop;
  QIO_Reader *qio_in;
};



//! QIO Writer class
class QDPSerialWriter
{
public:
  QDPSerialWriter();
  ~QDPSerialWriter();

  //! Open file
  explicit QDPSerialWriter(const XMLMetaWriter& xml, const std::string& p);

  //! Open file
  void open(const XMLMetaWriter& xml, const std::string& p);

  //! Close file
  void close();

  //! Is the file open?
  bool is_open();

  //! Write a QDP object
  template<class T, class C>
  void write(const XMLMetaWriter& xml, const QDPType<T,C>& s1)
    {write_t(*this,xml,static_cast<const C&>(s1));}


  //! Write an array of objects each in a seperate record
  template<class T>
  void write(const XMLMetaWriter& xml, const multi1d<T>& s1)
    {
      for(int i=0; i < s1.size(); ++i)
	write(xml,s1[i]);
    }

  //! Write an array of objects all in a single record
  template<class T>
  void vwrite(const XMLMetaWriter& xml, const multi1d<T>& s1) {}

  //!  Check if an unrecoverable error has occurred
  bool bad() const;


  QIO_Writer *get() const {return qio_out;}

private:
  bool iop;
  QIO_Writer *qio_out;
};



#if 0
//--------------------------------------------------------------------------------
//! Write a qdp object
template<class T>
inline
int write(QDPSerialWriter& qdpio, const XMLMetaData& xml, const OScalar<T>& f)
{
  return 0;
}

//! Write an outer scalar XML object element
//! Write an outer lattice namelist element
/*! The second arg is the string for the variable name */
template<class T>
inline
int write(QDPSerialWriter& qdpio, const XMLMetaData& xml, const OLattice<T>& f)
{
  return 0;
}

//! Write a multi1d array of objects
template<class T>
inline
int write(QDPSerialWriter& qdpio, const XMLMetaData& xml, const multi1d<T>& s1)
{
  for(int i=0; i < s1.size(); ++i)
    write(qdpio, xml, s1[i]);

  return 0;
}
#endif

/*! @} */   // end of group qio
QDP_END_NAMESPACE();
