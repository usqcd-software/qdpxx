// -*- C++ -*-
// $Id: qdp_qdpio.h,v 1.6 2003-10-23 20:40:24 edwards Exp $

/*! @file
 * @brief IO support via QIO
 */

#include "qio.h"
#include <sstream>
using namespace std;

QDP_BEGIN_NAMESPACE(QDP);

/*! @defgroup qio QIO
 *
 * File input and output operations on QDP types
 *
 * @{
 */

// Forward declarations
class QDPSerialFileReader;
class QDPSerialFileWriter;


//--------------------------------------------------------------------------------
//! QIO class
class QDPSerialFileReader
{
public:
  QDPSerialFileReader();
  ~QDPSerialFileReader();

  //! Open file
  explicit QDPSerialFileReader(XMLReader& xml, const std::string& p);

  //! Open file
  void open(XMLReader& xml, const std::string& p);

  //! Close file
  void close();

  //! Is the file open?
  bool is_open();

  //! Read a QDP object
  template<class T, class C>
  void read(XMLReader& xml, QDPType<T,C>& s1)
    {read(*this,xml,static_cast<C&>(s1));}

  //! Read an OScalar object
  template<class T>
  void read(XMLReader& xml, OScalar<T>& s1);

  //! Read an OLattice object
  template<class T>
  void read(XMLReader& xml, OLattice<T>& s1);

  //! Read an array of objects each in a seperate record
  /*! OOPS, what about xml? Is it repeatedly read but tossed out?? */
  template<class T>
  void read(XMLReader& xml, multi1d<T>& s1)
    {
      for(int i=0; i < s1.size(); ++i)
	read(xml,s1[i]);
    }


  //! Read an array of objects all in a single record
  template<class T>
  void vread(XMLReader& xml, multi1d<T>& s1) {}

  //! Check if end-of-file has been reached
  bool eof() const;

  //!  Check if an unrecoverable error has occurred
  bool bad() const;


protected:
  QIO_Reader *get() const {return qio_in;}

private:
  bool iop;
  QIO_Reader *qio_in;
};


//! Read an OLattice object
template<class T>
void read(QDPSerialFileReader& qsw, XMLReader& rec_xml, OLattice<T>& s1)
{
  qsw.read(rec_xml,s1);
}


//-------------------------------------------------
//! QIO Writer class
class QDPSerialFileWriter
{
public:
  QDPSerialFileWriter();
  ~QDPSerialFileWriter();

  //! Open file
  explicit QDPSerialFileWriter(XMLBufferWriter& xml, const std::string& p);

  //! Open file
  void open(XMLBufferWriter& xml, const std::string& p);

  //! Close file
  void close();

  //! Is the file open?
  bool is_open();

  //! Write a QDP object
  template<class T, class C>
  void write(XMLBufferWriter& xml, const QDPType<T,C>& s1)
    {write(*this,xml,static_cast<const C&>(s1));}

  //! Write an OScalar object
  template<class T>
  void write(XMLBufferWriter& xml, const OScalar<T>& s1);

  //! Write an OLattice object
  template<class T>
  void write(XMLBufferWriter& xml, const OLattice<T>& s1);

  //! Write an array of objects each in a separate record
  template<class T>
  void write(XMLBufferWriter& xml, const multi1d<T>& s1)
    {
      for(int i=0; i < s1.size(); ++i)
	write(xml,s1[i]);
    }

  //! Write an array of objects all in a single record
  template<class T>
  void vwrite(XMLBufferWriter& xml, const multi1d<T>& s1) {}

  //!  Check if an unrecoverable error has occurred
  bool bad() const;


protected:
  QIO_Writer *get() const {return qio_out;}

private:
  bool iop;
  QIO_Writer *qio_out;
};


//! Write an OLattice object
template<class T>
void write(QDPSerialFileWriter& qsw, XMLBufferWriter& rec_xml, const OLattice<T>& s1)
{
  qsw.write(rec_xml,s1);
}


//-------------------------------------------------
// QIO support
// NOTE: this is exactly the same bit of code as in scalar_specific.h 
//       need to make common only on scalarsite.h  like architectures

//! Function for inserting datum at specified site 
template<class T> void QDPFactoryPut(char *buf, const int crd[], void *arg)
{
  /* Translate arg */
  T *field = (T *)arg;

  /* We expect the data belongs to our node */
  multi1d<int> coord(Nd);
  coord = crd;
  if (Layout::nodeNumber(coord) != Layout::nodeNumber())
  {
    buf = '\0';
    return;
  }

  void *dest = (void *)&(field->elem(Layout::linearSiteIndex(coord)));
  int size = sizeof(field->elem(Layout::linearSiteIndex(coord)));

  memcpy(dest,buf,size);
}


//! Read an OLattice object
/*! This implementation is only correct for scalar ILattice */
template<class T>
void QDPSerialFileReader::read(XMLReader& rec_xml, OLattice<T>& s1)
{
  // Initialize string objects 
  XML_String *xml_c  = XML_string_create(0);
  XML_String *BinX_c = XML_string_create(0);

  int status = QIO_read(get(), xml_c, BinX_c, 
			&(QDPFactoryPut<OLattice<T> >),
                        sizeof(T), (void *)&s1);

  // Use string to initialize XMLReader
  istringstream ss((const string)(XML_string_ptr(xml_c)));
  rec_xml.open(ss);

  // Ignore BinX for now

  XML_string_destroy(BinX_c);
  XML_string_destroy(xml_c);
}


//! Function for extracting datum at specified site 
template< class T> void QDPFactoryGet(char *buf, const int crd[], void *arg)
{
  /* Translate arg */
  T *field = (T *)arg;

  /* We expect the data belongs to our node */
  multi1d<int> coord(Nd);
  coord = crd;
  if (Layout::nodeNumber(coord) != Layout::nodeNumber())
  {
    buf = '\0';
    return;
  }

  void *src = (void  *)&(field->elem(Layout::linearSiteIndex(coord)));
  size_t size = sizeof(field->elem(Layout::linearSiteIndex(coord))); 
  // cout << "Size = " << size << endl;

  memcpy(buf,(const void *)src,size);
}

/* template < class T > void QDPFactoryGet(char *buf, const int crd[], void *arg)
{
  QDP_abort(1);
}
*/
/*
template < class T > void QDPFactoryGet(char *buf, const int crd[], void *arg) {
  QDPFactoryGet< OLattice<T>, T >(buf, crd, arg);
}
*/

//! Write an OLattice object
/*! This implementation is only correct for scalar ILattice */
template<class T>
void QDPSerialFileWriter::write(XMLBufferWriter& rec_xml, const OLattice<T>& s1)
{
  //  cout << "QDPSerialFileWriter::write" << endl;


  // Copy metadata string into simple qio string container
  XML_String* xml_c  = XML_string_create(rec_xml.str().length()+1);  // check if +1 is needed
  XML_string_set(xml_c, rec_xml.str().c_str());

//  cout << "len=" << (rec_xml.str().length()+1) << endl;
//  cout << "rec_c= XXX" << xml_c << "XXX" << endl;

  // For now, create an empty binX field
  ostringstream binx;

  // Make up a dummy binX string
  binx << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << endl;
  binx << "<dataset xmlns=\"http://schemas.nesc.ac.uk/binx/binx\" " << endl;
  binx << "         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" " << endl;
  binx << "         xsi:schemaLocation=\"http://schemas.nesc.ac.uk/binx/binx\" />" << endl;

  
  XML_String* BinX_c = XML_string_create(binx.str().length()+1);  // check if +1 is needed -- Yes it is. Balint

  XML_string_set(BinX_c, binx.str().c_str());

//  cout << "len=" << (rec_xml.str().length()+1) << endl;
//  cout << "BinX_c= XXX" << xml_c << "XXX" << endl;

  // Big call to qio
  int status = QIO_write(get(), xml_c, BinX_c, 
			 &(QDPFactoryGet<OLattice<T> >),
                         sizeof(T), (void *)&s1);

//  cout << "cleanup in serialfile" << endl;

  // Cleanup
  XML_string_destroy(BinX_c);
  XML_string_destroy(xml_c);
}


/*! @} */   // end of group qio
QDP_END_NAMESPACE();
