// -*- C++ -*-
// $Id: qdp_qdpio.h,v 1.17 2004-03-23 03:10:47 edwards Exp $

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

//! File access mode
enum QDP_serialparallel_t
{
  QDPIO_SERIAL,
  QDPIO_PARALLEL
};

//! File format
enum QDP_volfmt_t
{
  QDPIO_SINGLEFILE,
  QDPIO_MULTIFILE
};

//! File open mode
enum QDP_filemode_t
{
  QDPIO_CREATE,
  QDPIO_OPEN,
  QDPIO_APPEND,
};

//! QDPIO state
enum QDP_iostate_t
{
  QDPIO_goodbit  = 0x0000,
  QDPIO_eofbit   = 0x0001,
  QDPIO_failbit  = 0x0010,
  QDPIO_badbit   = 0x0100,
};

//--------------------------------------------------------------------------------
//! QIO class
class QDPFileReader
{
public:
  //! Partial constructor
  QDPFileReader();

  //! Destructor
  ~QDPFileReader();

  //! Open file
  QDPFileReader(XMLReader& xml, const std::string& path,
		QDP_serialparallel_t qdp_serpar);

  //! Open file
  void open(XMLReader& xml, const std::string& path,
	    QDP_serialparallel_t qdp_serpar);

  //! Close file
  void close();

  //! Is the file open?
  bool is_open();

  //! Read a QDP object
  template<class T, class C>
  void read(XMLReader& xml, QDPType<T,C>& s1)
    {this->read(xml,static_cast<C&>(s1));}

  //! Read an OScalar object
  template<class T>
  void read(XMLReader& xml, OScalar<T>& s1);

  //! Read an OLattice object
  template<class T>
  void read(XMLReader& xml, OLattice<T>& s1);

  //! Read an array of objects all in a single record
  template<class T>
  void read(XMLReader& xml, multi1d< OScalar<T> >& s1);

  //! Read an array of objects all in a single record
  template<class T>
  void read(XMLReader& xml, multi1d< OLattice<T> >& s1);

  //! Check if end-of-file has been reached
  bool eof() const;

  //! Check if an unrecoverable error has occurred
  bool bad() const;

  //! Sets a new value for the control state ignoring the existing value
  void clear(QDP_iostate_t state = QDPIO_goodbit);

protected:
  QIO_Reader *get() const {return qio_in;}

private:
  QDP_iostate_t iostate;
  bool iop;
  QIO_Reader *qio_in;
};


// Convenience functions
//! Read an OScalar object
template<class T>
void read(QDPFileReader& qsw, XMLReader& rec_xml, OScalar<T>& s1)
{
  qsw.read(rec_xml,s1);
}

//! Read an array of OLattice objects
template<class T>
void read(QDPFileReader& qsw, XMLReader& rec_xml, OLattice<T>& s1)
{
  qsw.read(rec_xml,s1);
}

//! Read an array of OScalar object
template<class T>
void read(QDPFileReader& qsw, XMLReader& rec_xml, multi1d< OScalar<T> >& s1)
{
  qsw.read(rec_xml,s1);
}

//! Read an array of OLattice objects
template<class T>
void read(QDPFileReader& qsw, XMLReader& rec_xml, multi1d< OLattice<T> >& s1)
{
  qsw.read(rec_xml,s1);
}

//! Close a QDPFileReader
void close(QDPFileReader& qsw);

//! Is a QDPFileReader open
bool is_open(QDPFileReader& qsw);


//-------------------------------------------------
//! QIO Writer class
class QDPFileWriter
{
public:
  //! Partial constructor
  QDPFileWriter();

  //! Destructor
  ~QDPFileWriter();

  //! Open file
  QDPFileWriter(XMLBufferWriter& xml, const std::string& path,
		QDP_volfmt_t qdp_volfmt,
		QDP_serialparallel_t qdp_serpar,
		QDP_filemode_t qdp_mode);
  
  //! Open file
  void open(XMLBufferWriter& xml, const std::string& path,
	    QDP_volfmt_t qdp_volfmt,
	    QDP_serialparallel_t qdp_serpar,
	    QDP_filemode_t qdp_mode);

  //! Close file
  void close();

  //! Is the file open?
  bool is_open();

  //! Write a QDP object
  template<class T, class C>
  void write(XMLBufferWriter& xml, const QDPType<T,C>& s1)
    {this->write(xml,static_cast<const C&>(s1));}

  //! Write an OScalar object
  template<class T>
  void write(XMLBufferWriter& xml, const OScalar<T>& s1);

  //! Write an OLattice object
  template<class T>
  void write(XMLBufferWriter& xml, const OLattice<T>& s1);

  //! Write an array of objects all in a single record
  template<class T>
  void write(XMLBufferWriter& xml, const multi1d< OScalar<T> >& s1);

  //! Write an array of objects all in a single record
  template<class T>
  void write(XMLBufferWriter& xml, const multi1d< OLattice<T> >& s1);

  //!  Check if an unrecoverable error has occurred
  bool bad() const;

  //! Sets a new value for the control state ignoring the existing value
  void clear(QDP_iostate_t state = QDPIO_goodbit);

protected:
  QIO_Writer *get() const {return qio_out;}

private:
  QDP_iostate_t iostate;
  bool iop;
  QIO_Writer *qio_out;
};


// Convenience functions
//! Write an OScalar object
template<class T>
void write(QDPFileWriter& qsw, XMLBufferWriter& rec_xml, const OScalar<T>& s1)
{
  qsw.write(rec_xml,s1);
}

//! Write an OLattice object
template<class T>
void write(QDPFileWriter& qsw, XMLBufferWriter& rec_xml, const OLattice<T>& s1)
{
  qsw.write(rec_xml,s1);
}

//! Write an array of OScalar objects
template<class T>
void write(QDPFileWriter& qsw, XMLBufferWriter& rec_xml, const multi1d< OScalar<T> >& s1)
{
  qsw.write(rec_xml,s1);
}

//! Write an array of OLattice objects
template<class T>
void write(QDPFileWriter& qsw, XMLBufferWriter& rec_xml, const multi1d< OLattice<T> >& s1)
{
  qsw.write(rec_xml,s1);
}

//! Close a QDPFileWriter
void close(QDPFileWriter& qsw);

//! Is a QDPFileWriter open
bool is_open(QDPFileWriter& qsw);



//-------------------------------------------------
// QIO support
//
// Scalar support

//! Function for inserting datum at specified site 
template<class T> void QDPOScalarFactoryPut(char *buf, size_t linear, int count, void *arg)
{
  /* Translate arg */
  T *field = (T *)arg;

  void *dest = (void*)(field+linear);
  memcpy(dest,(const void*)buf,count*sizeof(T));
}


//! Read an OScalar object
/*! This implementation is only correct for scalar ILattice */
template<class T>
void QDPFileReader::read(XMLReader& rec_xml, OScalar<T>& s1)
{
  QIO_RecordInfo* info = QIO_create_record_info(QIO_GLOBAL, "Scalar", "F", Nc, Ns, 
						sizeof(T), 1);

  // Initialize string objects 
  QIO_String *xml_c  = QIO_string_create(0);

  if (QIO_read(get(), info, xml_c,
   	       &(QDPOScalarFactoryPut<T>),
               sizeof(T), 
	       sizeof(typename WordType<T>::Type_t), 
	       (void *)s1.elem()) != QIO_SUCCESS)
  {
    QDPIO::cerr << "QDPFileReader: error reading file" << endl;
    clear(QDPIO_badbit);
  }

  // Use string to initialize XMLReader
  istringstream ss;
  if (Layout::primaryNode())
  {
    string foo = QIO_string_ptr(xml_c);
    ss.str(foo);
  }
  rec_xml.open(ss);

  QIO_string_destroy(xml_c);
  QIO_destroy_record_info(info);
}


//! Read an array of OScalar objects
/*! This implementation is only correct for scalar ILattice */
template<class T>
void QDPFileReader::read(XMLReader& rec_xml, multi1d< OScalar<T> >& s1)
{
  QIO_RecordInfo* info = QIO_create_record_info(QIO_GLOBAL, "Scalar", "F", Nc, Ns, 
						sizeof(T), s1.size()); // need size for now

  // Initialize string objects 
  QIO_String *xml_c  = QIO_string_create(0);

  if (QIO_read(get(), info, xml_c,
   	       &(QDPOScalarFactoryPut<T>),
               s1.size()*sizeof(T), 
	       sizeof(typename WordType<T>::Type_t), 
	       (void *)s1.slice()) != QIO_SUCCESS)
  {
    QDPIO::cerr << "QDPFileReader: error reading file" << endl;
    clear(QDPIO_badbit);
  }

  // Use string to initialize XMLReader
  istringstream ss;
  if (Layout::primaryNode())
  {
    string foo = QIO_string_ptr(xml_c);
    ss.str(foo);
  }
  rec_xml.open(ss);

  QIO_string_destroy(xml_c);
  QIO_destroy_record_info(info);
}


//! Function for extracting datum at specified site 
template<class T> void QDPOScalarFactoryGet(char *buf, size_t linear, int count, void *arg)
{
  /* Translate arg */
  T *field = (T *)arg;

  void *src = (void*)(field+linear);
  memcpy(buf,(const void*)src,count*sizeof(T));
}


//! Write an OScalar object
/*! This implementation is only correct for scalar ILattice */
template<class T>
void QDPFileWriter::write(XMLBufferWriter& rec_xml, const OScalar<T>& s1)
{
  QIO_RecordInfo* info = QIO_create_record_info(QIO_GLOBAL, "Scalar", "F", Nc, Ns, 
						sizeof(T), 1);

  // Copy metadata string into simple qio string container
  QIO_String* xml_c;
  if (Layout::primaryNode())
    xml_c = QIO_string_set(rec_xml.str().c_str());
  else
    xml_c = QIO_string_create(0);

  if (xml_c == NULL)
  {
    QDPIO::cerr << "QDPFileWriter::write - error in creating XML string" << endl;
    QDP_abort(1);
  }

  // Big call to qio
  if (QIO_write(get(), info, xml_c,
	        &(QDPOScalarFactoryGet<T>),
                sizeof(T), 
 	        sizeof(typename WordType<T>::Type_t), 
	        (void *)s1.elem()) != QIO_SUCCESS)
  {
    QDPIO::cerr << "QDPFileWriter: error in write" << endl;
    clear(QDPIO_badbit);
  }

  // Cleanup
  QIO_string_destroy(xml_c);
  QIO_destroy_record_info(info);
}


//! Write an array of OScalar objects
/*! This implementation is only correct for scalar ILattice */
template<class T>
void QDPFileWriter::write(XMLBufferWriter& rec_xml, const multi1d< OScalar<T> >& s1)
{
  QIO_RecordInfo* info = QIO_create_record_info(QIO_GLOBAL, "Scalar", "F", Nc, Ns, 
						sizeof(T), s1.size());

  // Copy metadata string into simple qio string container
  QIO_String* xml_c;
  if (Layout::primaryNode())
    xml_c = QIO_string_set(rec_xml.str().c_str());
  else
    xml_c = QIO_string_create(0);

  if (xml_c == NULL)
  {
    QDPIO::cerr << "QDPFileWriter::write - error in creating XML string" << endl;
    QDP_abort(1);
  }

  // Big call to qio
  if (QIO_write(get(), info, xml_c,
	        &(QDPOScalarFactoryGet<T>),
                s1.size()*sizeof(T), 
 	        sizeof(typename WordType<T>::Type_t), 
	        (void *)s1.slice()) != QIO_SUCCESS)
  {
    QDPIO::cerr << "QDPFileWriter: error in write" << endl;
    clear(QDPIO_badbit);
  }

  // Cleanup
  QIO_string_destroy(xml_c);
  QIO_destroy_record_info(info);
}




//-------------------------------------------------
// QIO support
// NOTE: this is exactly the same bit of code as in scalar_specific.h 
//       need to make common only on scalarsite.h  like architectures

//! Function for inserting datum at specified site 
template<class T> void QDPOLatticeFactoryPut(char *buf, size_t linear, int count, void *arg)
{
  /* Translate arg */
  T *field = (T *)arg;

  void *dest = (void*)(field+linear);
  memcpy(dest,(const void*)buf,count*sizeof(T));
}


//! Read an OLattice object
/*! This implementation is only correct for scalar ILattice */
template<class T>
void QDPFileReader::read(XMLReader& rec_xml, OLattice<T>& s1)
{
  QIO_RecordInfo* info = QIO_create_record_info(QIO_FIELD, "Lattice", "F", Nc, Ns, 
						sizeof(T), 1);

  // Initialize string objects 
  QIO_String *xml_c  = QIO_string_create(0);

  if (QIO_read(get(), info, xml_c,
   	       &(QDPOLatticeFactoryPut<T>),
               sizeof(T), 
	       sizeof(typename WordType<T>::Type_t), 
	       (void *)s1.getF()) != QIO_SUCCESS)
  {
    QDPIO::cerr << "QDPFileReader: error reading file" << endl;
    clear(QDPIO_badbit);
  }

  // Use string to initialize XMLReader
  istringstream ss;
  if (Layout::primaryNode())
  {
    string foo = QIO_string_ptr(xml_c);
    ss.str(foo);
  }
  rec_xml.open(ss);

  QIO_string_destroy(xml_c);
  QIO_destroy_record_info(info);
}


//! Read an array of OLattice objects
/*! This implementation is only correct for scalar ILattice */
template<class T>
void QDPFileReader::read(XMLReader& rec_xml, multi1d< OLattice<T> >& s1)
{
  QIO_RecordInfo* info = QIO_create_record_info(QIO_FIELD, "Lattice", "F", Nc, Ns, 
						sizeof(T), s1.size());

  // Initialize string objects 
  QIO_String *xml_c  = QIO_string_create(0);

  // Unfortunately, make one big contiguous array
  T *ssa = new T[s1.size() * Layout::sitesOnNode()];

  if (QIO_read(get(), info, xml_c,
   	       &(QDPOLatticeFactoryPut<T>),
               s1.size()*sizeof(T), 
	       sizeof(typename WordType<T>::Type_t), 
	       ssa) != QIO_SUCCESS)
  {
    QDPIO::cerr << "QDPFileReader: error reading file" << endl;
    clear(QDPIO_badbit);
  }

  for(int nn=0, mu=0; mu < s1.size(); ++mu)
    for(int site=0; site < Layout::sitesOnNode(); ++site)
      s1[mu].elem(site) = ssa[nn++];

  // Use string to initialize XMLReader
  istringstream ss;
  if (Layout::primaryNode())
  {
    string foo = QIO_string_ptr(xml_c);
    ss.str(foo);
  }
  rec_xml.open(ss);

  delete[] ssa;
  QIO_string_destroy(xml_c);
  QIO_destroy_record_info(info);
}


//! Function for extracting datum at specified site 
template<class T> void QDPOLatticeFactoryGet(char *buf, size_t linear, int count, void *arg)
{
  /* Translate arg */
  T *field = (T *)arg;

  void *src = (void*)(field+linear);
  memcpy(buf,(const void*)src,count*sizeof(T));
}


//! Write an OLattice object
/*! This implementation is only correct for scalar ILattice */
template<class T>
void QDPFileWriter::write(XMLBufferWriter& rec_xml, const OLattice<T>& s1)
{
  QIO_RecordInfo* info = QIO_create_record_info(QIO_FIELD, "Lattice", "F", Nc, Ns, 
						sizeof(T), 1);

  // Copy metadata string into simple qio string container
  QIO_String* xml_c;
  if (Layout::primaryNode())
    xml_c = QIO_string_set(rec_xml.str().c_str());
  else
    xml_c = QIO_string_create(0);

  if (xml_c == NULL)
  {
    QDPIO::cerr << "QDPFileWriter::write - error in creating XML string" << endl;
    QDP_abort(1);
  }

  // Big call to qio
  if (QIO_write(get(), info, xml_c,
	        &(QDPOLatticeFactoryGet<T>),
                sizeof(T), 
 	        sizeof(typename WordType<T>::Type_t), 
	        (void *)s1.getF()) != QIO_SUCCESS)
  {
    QDPIO::cerr << "QDPFileWriter: error in write" << endl;
    clear(QDPIO_badbit);
  }

  // Cleanup
  QIO_string_destroy(xml_c);
  QIO_destroy_record_info(info);
}


//! Write an array of OLattice objects
/*! This implementation is only correct for scalar ILattice */
template<class T>
void QDPFileWriter::write(XMLBufferWriter& rec_xml, const multi1d< OLattice<T> >& s1)
{
  QIO_RecordInfo* info = QIO_create_record_info(QIO_FIELD, "Lattice", "F", Nc, Ns, 
						sizeof(T), s1.size());

  // Copy metadata string into simple qio string container
  QIO_String* xml_c;
  if (Layout::primaryNode())
    xml_c = QIO_string_set(rec_xml.str().c_str());
  else
    xml_c = QIO_string_create(0);

  if (xml_c == NULL)
  {
    QDPIO::cerr << "QDPFileWriter::write - error in creating XML string" << endl;
    QDP_abort(1);
  }

  // Unfortunately, make one big contiguous array
  T *ssa = new T[s1.size() * Layout::sitesOnNode()];
  for(int nn=0, mu=0; mu < s1.size(); ++mu)
    for(int site=0; site < Layout::sitesOnNode(); ++site)
      ssa[nn++] = s1[mu].elem(site);

  // Big call to qio
  if (QIO_write(get(), info, xml_c,
	        &(QDPOLatticeFactoryGet<T>),
                s1.size()*sizeof(T), 
 	        sizeof(typename WordType<T>::Type_t), 
	        ssa) != QIO_SUCCESS)
  {
    QDPIO::cerr << "QDPFileWriter: error in write" << endl;
    clear(QDPIO_badbit);
  }

  // Cleanup
  delete[] ssa;
  QIO_string_destroy(xml_c);
  QIO_destroy_record_info(info);
}


/*! @} */   // end of group qio
QDP_END_NAMESPACE();
