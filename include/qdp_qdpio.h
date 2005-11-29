// -*- C++ -*-
// $Id: qdp_qdpio.h,v 1.28 2005-11-29 19:34:22 bjoo Exp $

/*! @file
 * @brief IO support via QIO
 */

#ifndef QDP_QDPIO_H
#define QDP_QDPIO_H

#include "qio.h"
#include <sstream>
#include "qdp_defs.h"

using namespace std;

QDP_BEGIN_NAMESPACE(QDP);

/*! @defgroup qio QIO
 *
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
  QDPIO_MULTIFILE,
  QDPIO_PARTFILE
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


//! A little namespace to map the QDP types to the right strings
namespace QIOStrings { 

  // Catch all base (Hopefully never called)
  template<typename T> 
  inline
  void QIOTypeStringFromType(std::string& tname, const T& t) 
  {    
    tname = "QDP_GenericType";
  }
  
  // Backward compatibility
  template<typename T>
  inline
  void QIOTypeStringFromType(std::string& tname, 
			     const OScalar<T>& t) 
  { 
    tname  = "Scalar";
  }
  
  template<typename T>
  inline
  void QIOTypeStringFromType(std::string& tname, 
			     const OLattice<T>& t) 
  {
    tname  = "Lattice";
  }
  
  // Backward compatibility
  template<typename T>
  inline
  void QIOTypeStringFromType(std::string& tname , 
			     const multi1d< OScalar<T> >& t) 
  { 
    tname  = "Scalar";
  }
  
  
  // Backward Compatibility
  template<typename T>
  inline
  void QIOTypeStringFromType(std::string& tname , 
			     const multi1d< OLattice<T> >& t) 
  {
    tname  = "Lattice";
  }
    
  
  // Specialisation
  // Gauge Field Type: multi1d<LatticeColorMatrix> 
  // Need specific type string to output in ILDG format with QIO
  // However I cannot inline these. They have to go into qdp_qio_strings.cc
  // If I try to inline them here, I get linkage errors with multiple
  // definitions. I wonder why?
  template<>
  void QIOTypeStringFromType(std::string& tname , 
			     const multi1d< LatticeColorMatrixF3 >& t );

  template<> 
  void QIOTypeStringFromType(std::string& tname , 
			     const multi1d< LatticeColorMatrixD3 >& t);
  
    
  char QIOSizeToStr(size_t size);
  
  
  template<typename T>
  inline
  char QIOPrecisionStringFromType(const T& t) {
    return QIOSizeToStr(sizeof(typename WordType<T>::Type_t));
  }
  
  template<typename T>
  inline
  char QIOPrecisionStringFromType(const OScalar<T>& t) {
    return QIOSizeToStr(sizeof(typename WordType<T>::Type_t));
  }
  
  template<typename T>
  inline
  char QIOPrecisionStringFromType(const multi1d<OScalar<T> >& t) {
    return QIOSizeToStr(sizeof(typename WordType<T>::Type_t));
  }
  
  template<typename T>
  inline
  char QIOPrecisionStringFromType(const OLattice<T>& t) {
    return QIOSizeToStr(sizeof(typename WordType<T>::Type_t));
  }
  
  template<typename T>
  inline
  char QIOPrecisionStringFromType(const multi1d<OLattice<T> >& t) {
    return QIOSizeToStr(sizeof(typename WordType<T>::Type_t));
  }
}

//--------------------------------------------------------------------------------
//! QIO class
/*!
 This is a QDP object wrapper around the QIO library.
 
 QIO is a C library independentof QDP. It is designed to read/write SCIDAC
 format data files, which means a mixture of binary data and XML
 metadata together in the same file according to a scheme called Lime.
 There is a seperate independent library for handling general Lime files.

 The data is assumed to be a record in a Lime file. The user metadata (both
 file and record) is also read.
 
 Data is assumed to be stored in the file in big-endian format and any
 necessary byte-swapping is taken care of.

 The status of the IO operations is monitored internally and can be queried.
*/

class QDPFileReader
{
public:
  //! Partial constructor
  QDPFileReader();


  //! Closes the last file opened.
  ~QDPFileReader();

  //! Opens a file for reading
    /*!
      Also reads the file user metadata record.
      \param xml Container for the file metadata.
      \param path The name of the file.
      \param qdp_serpar Serial or parallel IO.
    */
  QDPFileReader(XMLReader& xml, const std::string& path,
		QDP_serialparallel_t qdp_serpar);
  
    //! Opens a file for reading
    /*!
      Also reads the file user metadata record.
      \param xml Container for the file metadata.
      \param path The name of the file.
      \param qdp_serpar Serial or parallel IO.
    */
  void open(XMLReader& xml, const std::string& path,
	    QDP_serialparallel_t qdp_serpar);

  //! Closes the last file opened.
  void close();

    //! Queries whether a file is open
    /*!
      \return true if a file is open; false otherwise.
    */
  bool is_open();

  //! Read a QDP object
  template<class T, class C>
  void read(XMLReader& xml, QDPType<T,C>& s1)
    {this->read(xml,static_cast<C&>(s1));}

  //! Reads an OScalar object
  template<class T>
  void read(XMLReader& xml, OScalar<T>& s1);

  //! Reads an OLattice object
  template<class T>
  void read(XMLReader& xml, OLattice<T>& s1);

  //! Reads an array of objects all in a single record
  template<class T>
  void read(XMLReader& xml, multi1d< OScalar<T> >& s1);

  //! Reads an array of objects all in a single record
  template<class T>
  void read(XMLReader& xml, multi1d< OLattice<T> >& s1);

    //! Query whether the end-of-file has been reached.
    /*!
      \return True if  the end-of-file has been reached; false otherwise
    */
  bool eof() const;

    //! Query whether an unrecoverable error has occurred
    /*!
      \return True if an error has occured; false otherwise
    */
  bool bad() const;

    //! Sets a new value for the IO status, ignoring the existing value
    /*!
      \param state The new state (defaults to QDPIO_goodbit).
    */
  void clear(QDP_iostate_t state = QDPIO_goodbit);

protected:
  QIO_Reader *get() const {return qio_in;}

private:
  QDP_iostate_t iostate;
  bool iop;
  QIO_Reader *qio_in;
};


// Convenience functions

//! Reads an OScalar object
/*!
  \param qsw The reader
  \param rec_xml The user record metadata.
  \param sl The data
*/
template<class T>
void read(QDPFileReader& qsw, XMLReader& rec_xml, OScalar<T>& s1)
{
  qsw.read(rec_xml,s1);
}

//! Reads an array of OLattice objects
/*!
  \param qsw The reader
  \param rec_xml The user record metadata.
  \param sl The data
*/
template<class T>
void read(QDPFileReader& qsw, XMLReader& rec_xml, OLattice<T>& s1)
{
  qsw.read(rec_xml,s1);
}

//! Reads an array of OScalar object
/*!
  \param qsw The reader
  \param rec_xml The user record metadata.
  \param sl The data
*/
template<class T>
void read(QDPFileReader& qsw, XMLReader& rec_xml, multi1d< OScalar<T> >& s1)
{
  qsw.read(rec_xml,s1);
}

//! Reads an array of OLattice objects
/*!
  \param qsw The reader
  \param rec_xml The user record metadata.
  \param sl The data
*/
template<class T>
void read(QDPFileReader& qsw, XMLReader& rec_xml, multi1d< OLattice<T> >& s1)
{
  qsw.read(rec_xml,s1);
}

//! Closes a QDPFileReader.
void close(QDPFileReader& qsw);

//! Queries whether a QDPFileReader is open.
bool is_open(QDPFileReader& qsw);


//-------------------------------------------------
//! QIO writer class
/*!
 This is a QDP object wrapper around the QIO library.
 
 QIO is a C library independent of QDP. It is designed to read/write SCIDAC
 format data files, which means a mixture of binary data and XML
 metadata together in the same file according to a scheme called Lime.
 There is a seperate independent library for handling general Lime files.

 The data is written as a record in a Lime file. The user metadata (both
 file and record) is also written.
 
 Data is stored in the file in big-endian format and any
 necessary byte-swapping is taken care of.

 The status of the IO operations is monitored internally and can be queried. 
*/

class QDPFileWriter
{
public:
  //! Partial constructor
  QDPFileWriter();

  //! Closes the last file opened.
  ~QDPFileWriter();

  //! Opens a file for writing and writes the file metadata
    /*!
      Also reads the file user metadata record.
      \param xml Container for the file metadata
      \param path The name of the file
      \param qdp_volfmt The type of file to write.
      \param qdp_serpar Serial or parallel IO
    */
  QDPFileWriter(XMLBufferWriter& xml, const std::string& path,
		QDP_volfmt_t qdp_volfmt,
		QDP_serialparallel_t qdp_serpar);

  //! Opens a file for writing and writes the file metadata
    /*!
      Also reads the file user metadata record.
      \param xml Container for the file metadata
      \param path The name of the file
      \param qdp_volfmt The type of file to write.
      \param qdp_serpar Serial or parallel IO
      \param data_LFN   LFN for ILDG style output 
    */
  QDPFileWriter(XMLBufferWriter& xml, const std::string& path,
		QDP_volfmt_t qdp_volfmt,
		QDP_serialparallel_t qdp_serpar,
		const std::string& data_LFN);
  
  //! Opens a file for writing and writes the file metadata
    /*!
      Also reads the file user metadata record.
      \param xml Container for the file metadata
      \param path The name of the file
      \param qdp_volfmt The type of file to write.
      \param qdp_serpar Serial or parallel IO
    */
  void open(XMLBufferWriter& xml, const std::string& path,
	    QDP_volfmt_t qdp_volfmt,
	    QDP_serialparallel_t qdp_serpar);

  //! Opens a file for writing and writes the file metadata
    /*!
      Also reads the file user metadata record.
      \param xml Container for the file metadata
      \param path The name of the file
      \param qdp_volfmt The type of file to write.
      \param qdp_serpar Serial or parallel IO
      \param data_LFN   ILDG DATA LFN if we need to add one
    */
  void open(XMLBufferWriter& xml, const std::string& path,
	    QDP_volfmt_t qdp_volfmt,
	    QDP_serialparallel_t qdp_serpar,
	    const std::string& data_LFN);
  
  //! Opens a file for writing and writes the file metadata
    /*!
      Also reads the file user metadata record.
      \param xml Container for the file metadata
      \param path The name of the file
      \param qdp_volfmt The type of file to write.
      \param qdp_serpar Serial or parallel IO
      \param qdp_mode The file  opening mode		
    */

  QDPFileWriter(XMLBufferWriter& xml, const std::string& path,
		QDP_volfmt_t qdp_volfmt,
		QDP_serialparallel_t qdp_serpar,
		QDP_filemode_t qdp_mode);

  //! Opens a file for writing and writes the file metadata
    /*!
      Also reads the file user metadata record.
      \param xml Container for the file metadata
      \param path The name of the file
      \param qdp_volfmt The type of file to write.
      \param qdp_serpar Serial or parallel IO
      \param qdp_mode The file  opening mode		
      \param data_LFN The ILDG Data LFN
    */

  QDPFileWriter(XMLBufferWriter& xml, const std::string& path,
		QDP_volfmt_t qdp_volfmt,
		QDP_serialparallel_t qdp_serpar,
		QDP_filemode_t qdp_mode,
		const std::string& data_LFN);
  
  //! Opens a file for writing and writes the file metadata
    /*!
      Also reads the file user metadata record.
      \param xml Container for the file metadata
      \param path The name of the file
      \param qdp_volfmt The type of file to write.
      \param qdp_serpar Serial or parallel IO.
      \param qdp_mode The file opening mode	
      \param data_LFN The ILDG Data LFN	
    */

  void open(XMLBufferWriter& xml, const std::string& path,
	    QDP_volfmt_t qdp_volfmt,
	    QDP_serialparallel_t qdp_serpar,
	    QDP_filemode_t qdp_mode,
	    const std::string& data_LFN);

  //! Closes the last file opened.
  void close();

    //! Queries whether a file is open
    /*!
      \return true if a file is open; false otherwise.
    */
  bool is_open();

    //! Write a QDP object
    /*!
      \param xml The user record metadata.
      \param sl The data
    */
  template<class T, class C>
  void write(XMLBufferWriter& xml, const QDPType<T,C>& s1)
    {this->write(xml,static_cast<const C&>(s1));}

  //! Writes an OScalar object
    /*!
      \param xml The user record metadata.
      \param sl The data
    */
  template<class T>
  void write(XMLBufferWriter& xml, const OScalar<T>& s1);

  //! Writes an OLattice object
    /*!
      \param xml The user record metadata.
      \param sl The data
    */
  template<class T>
  void write(XMLBufferWriter& xml, const OLattice<T>& s1);

  //! Writes an array of objects all to a single record
    /*!
      \param xml The user record metadata.
      \param sl The data
    */
  template<class T>
  void write(XMLBufferWriter& xml, const multi1d< OScalar<T> >& s1);

  //! Writes an array of objects all to a single record
    /*!
      \param xml The user record metadata.
      \param sl The data
    */
  template<class T>
  void write(XMLBufferWriter& xml, const multi1d< OLattice<T> >& s1);

    //! Query whether an unrecoverable error has occurred
    /*!
      \return True if an error has occured; false otherwise
    */
  bool bad() const;

    //! Sets a new value for the control state ignoring the existing value
    /*!
      \param state The new state (defaults to QDPIO_goodbit).
    */
  void clear(QDP_iostate_t state = QDPIO_goodbit);

protected:
  QIO_Writer *get() const {return qio_out;}

private:
  QDP_iostate_t iostate;
  bool iop;
  QIO_Writer *qio_out;
};


// Convenience functions
//! Writes an OScalar object
/*!
  \param qsw The writer
  \param xml The user record metadata.
  \param sl The data
*/
template<class T>
void write(QDPFileWriter& qsw, XMLBufferWriter& rec_xml, const OScalar<T>& s1)
{
  qsw.write(rec_xml,s1);
}

//! Writes an OLattice object
/*!
  \param qsw The writer
  \param xml The user record metadata.
  \param sl The data
*/
template<class T>
void write(QDPFileWriter& qsw, XMLBufferWriter& rec_xml, const OLattice<T>& s1)
{
  qsw.write(rec_xml,s1);
}

//! Writes an array of OScalar objects
/*!
  \param qsw The writer
  \param xml The user record metadata.
  \param sl The data
*/
template<class T>
void write(QDPFileWriter& qsw, XMLBufferWriter& rec_xml, const multi1d< OScalar<T> >& s1)
{
  qsw.write(rec_xml,s1);
}

//! Writes an array of OLattice objects
/*!
  \param qsw The writer
  \param xml The user record metadata.
  \param sl The data
*/
template<class T>
void write(QDPFileWriter& qsw, XMLBufferWriter& rec_xml, const multi1d< OLattice<T> >& s1)
{
  qsw.write(rec_xml,s1);
}

//! Closes a QDPFileWriter.
void close(QDPFileWriter& qsw);

//! Queries whether a QDPFileWriter is open.
bool is_open(QDPFileWriter& qsw);



//-------------------------------------------------
// QIO support
//
// Scalar support

//! Function for moving data
/*!
  Data is moved from the one buffer to another with a specified offset.

  \param buf The source buffer
  \param linear The offset
  \param count The number of data to move
  \param arg The destination buffer.
*/
template<class T> void QDPOScalarFactoryPut(char *buf, size_t linear, int count, void *arg) 
{
  /* Translate arg */
  T *field = (T *)arg;

  void *dest = (void*)(field+linear);
  memcpy(dest,(const void*)buf,count*sizeof(T));
}


//! Reads an OScalar object
/*!
  This implementation is only correct for scalar ILattice

  \param rec_xml The (user) record metadata.
  \param sl The data
*/
template<class T>
void QDPFileReader::read(XMLReader& rec_xml, OScalar<T>& s1)
{

  try { 

    std::string tname;
    char tprec[2]={0,'\0'};

    QIOStrings::QIOTypeStringFromType(tname, s1);
    tprec[0] = QIOStrings::QIOPrecisionStringFromType(s1);

    QIO_RecordInfo* info = QIO_create_record_info(QIO_GLOBAL, 
						  const_cast<char *>(tname.c_str()),
						  tprec,
						  Nc, Ns, 
						  sizeof(T), 1);

    // Initialize string objects 
    QIO_String *xml_c  = QIO_string_create();
    
    
    QDPIO::cout << "xml_string->string = " << xml_c->string << endl;
    QDPIO::cout << "xml_string->length=" << xml_c->length << endl;
    QDPIO::cout << flush;
    
    if (QIO_read(get(), info, xml_c,
		 &(QDPOScalarFactoryPut<T>),
		 sizeof(T), 
		 sizeof(typename WordType<T>::Type_t), 
		 (void *)s1.elem()) != QIO_SUCCESS) {
	QDPIO::cerr << "QDPFileReader: error reading file" << endl;
	clear(QDPIO_badbit);
    }
    
    QDPIO::cout << "xml_string->string = " << xml_c->string << endl;
    QDPIO::cout << "xml_string->length=" << xml_c->length << endl;
    QDPIO::cout << flush;
    
    
    // Use string to initialize XMLReader
    istringstream ss;
    if (Layout::primaryNode()) {
      string foo = QIO_string_ptr(xml_c);
      ss.str(foo);
    }
    rec_xml.open(ss);
    
    QIO_string_destroy(xml_c);
    QIO_destroy_record_info(info);
    
  }
  catch(std::bad_alloc) { 
    QDPIO::cout << "Caught BAD ALLOC Exception" << endl;
    QDP_abort(1);
  }
  
}


//! Reads an array of OScalar objects
/*!
  This implementation is only correct for scalar ILattice

  \param rec_xml The (user) record metadata.
  \param sl The data
*/
template<class T>
void QDPFileReader::read(XMLReader& rec_xml, multi1d< OScalar<T> >& s1)
{
    std::string tname;
    //    char tprec;
    char tprec[2]={0,'\0'};

    QIOStrings::QIOTypeStringFromType(tname, s1);
    tprec[0] = QIOStrings::QIOPrecisionStringFromType(s1);

    QIO_RecordInfo* info = QIO_create_record_info(QIO_GLOBAL, 
						  const_cast<char *>(tname.c_str()),
						  tprec,
						  Nc, Ns, 
						  sizeof(T), s1.size()); // need size for now

  // Initialize string objects 
  QIO_String *xml_c  = QIO_string_create();

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


//! Function for moving data
/*!
  Data is moved from the one buffer to another with a specified offset.

  \param buf The destination buffer
  \param linear The source buffer offset
  \param count The number of data to move
  \param arg The source buffer.
*/

template<class T> void QDPOScalarFactoryGet(char *buf, size_t linear, int count, void *arg)
{
  /* Translate arg */
  T *field = (T *)arg;

  void *src = (void*)(field+linear);
  memcpy(buf,(const void*)src,count*sizeof(T));
}


//! Writes an OScalar object
/*!
  This implementation is only correct for scalar ILattice.

  \param rec_xml The (user) record metadata.
  \param sl The data
*/
template<class T>
void QDPFileWriter::write(XMLBufferWriter& rec_xml, const OScalar<T>& s1)
{
  std::string tname;
  //  char tprec;
  char tprec[2]={0,'\0'};
  
  QIOStrings::QIOTypeStringFromType(tname, s1);
  tprec[0] = QIOStrings::QIOPrecisionStringFromType(s1);
  
  QIO_RecordInfo* info = QIO_create_record_info(QIO_GLOBAL, 
						const_cast<char *>(tname.c_str()),
						tprec, 
						Nc, Ns, 
						sizeof(T), 1);

  // Copy metadata string into simple qio string container
  QIO_String* xml_c = QIO_string_create();
  
  if (Layout::primaryNode())
    QIO_string_set(xml_c, rec_xml.str().c_str());

  if (xml_c == NULL)
  {
    QDPIO::cerr << "QDPFileWriter::write - error in creating XML string" << endl;
    QDP_abort(1);
  }

  // Big call to QIO
  
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


//! Writes an array of OScalar objects
/*!
  This implementation is only correct for scalar ILattice.

  \param rec_xml The (user) record metadata.
  \param sl The data
*/
template<class T>
void QDPFileWriter::write(XMLBufferWriter& rec_xml, const multi1d< OScalar<T> >& s1)
{

  std::string tname;
  //  char tprec;
  char tprec[2]={0,'\0'};
  
  QIOStrings::QIOTypeStringFromType(tname, s1);
  tprec[0] = QIOStrings::QIOPrecisionStringFromType(s1);
  
  QIO_RecordInfo* info = QIO_create_record_info(QIO_GLOBAL, 
						const_cast<char *>(tname.c_str()),
						tprec, 
						Nc, Ns, 
						sizeof(T), s1.size());


  // Copy metadata string into simple qio string container
  QIO_String* xml_c = QIO_string_create();
  if (Layout::primaryNode())
    QIO_string_set(xml_c, rec_xml.str().c_str());

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

//! Function for moving data
/*!
  Data is moved from the one buffer to another with a specified offset.

  \param buf The source buffer
  \param linear The destination buffer offset
  \param count The number of data to move
  \param arg The destination buffer.
*/
template<class T> void QDPOLatticeFactoryPut(char *buf, size_t linear, int count, void *arg)
{
  /* Translate arg */
  T *field = (T *)arg;

  void *dest = (void*)(field+linear);
  memcpy(dest,(const void*)buf,count*sizeof(T));
}

//! Function for moving array data
/*!
  Data is moved from the one buffer to another buffer array with a
  specified offset. 
  The data is taken to be in multi1d< OLattice<T> > form.

  \param buf The source buffer
  \param linear The destination buffer offset
  \param count Ignored
  \param arg The destination buffer.
*/
template<class T> void QDPOLatticeFactoryPutArray(char *buf, size_t linear, int count, void *arg)
{
  /* Translate arg */
  multi1d< OLattice<T> >& field = *(multi1d< OLattice<T> > *)arg;

  for(int i=0; i < field.size(); ++i)
  {
    void *dest = (void*)&(field[i].elem(linear));
    memcpy(dest,(const void*)buf,sizeof(T));
    buf += sizeof(T);
  }
}


//! Reads an OLattice object
/*!
  This implementation is only correct for scalar ILattice.

  \param rec_xml The (user) record metadata.
  \param sl The data
*/
template<class T>
void QDPFileReader::read(XMLReader& rec_xml, OLattice<T>& s1)
{

  try { 


    std::string tname;
    //    char tprec;
    char tprec[2]={0,'\0'};
    
    QIOStrings::QIOTypeStringFromType(tname, s1);
    tprec[0] = QIOStrings::QIOPrecisionStringFromType(s1);
    
    QIO_RecordInfo* info = QIO_create_record_info(QIO_FIELD, 
						  const_cast<char *>(tname.c_str()), tprec,
						  Nc, Ns, 
						  sizeof(T),1);


    
    // Initialize string objects 
    QIO_String *xml_c  = QIO_string_create();
    try { 
    if (QIO_read(get(), info, xml_c,
		 &(QDPOLatticeFactoryPut<T>),
		 sizeof(T), 
		 sizeof(typename WordType<T>::Type_t), 
		 (void *)s1.getF()) != QIO_SUCCESS)
      {
	QDPIO::cerr << "QDPFileReader: error reading file" << endl;
	clear(QDPIO_badbit);
      }
    }
    catch(std::bad_alloc) { 
      QDPIO::cerr << "Bad Alloc Exception caught in QIO_read(OLattice) " <<endl << flush;
      QDP_abort(-1);
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
  catch(std::bad_alloc) { 
    QDPIO::cout << "Bad ALLOC exception caught " << endl;
    QDP_abort(-1);
  }
}


//! Reads an array of OLattice objects
/*!
  This implementation is only correct for scalar ILattice.

  \param rec_xml The (user) record metadata.
  \param sl The data
*/
template<class T>
void QDPFileReader::read(XMLReader& rec_xml, multi1d< OLattice<T> >& s1)
{

  std::string tname;
  //  char tprec;
  char tprec[2]={0,'\0'};
  
  QIOStrings::QIOTypeStringFromType(tname, s1);
  tprec[0] = QIOStrings::QIOPrecisionStringFromType(s1);
  
  QIO_RecordInfo* info = QIO_create_record_info(QIO_FIELD, 
						const_cast<char *>(tname.c_str()), tprec, 
						Nc, Ns, 
						sizeof(T),s1.size());
  

  // Initialize string objects 
  QIO_String *xml_c  = QIO_string_create();

  if (QIO_read(get(), info, xml_c,
   	       &(QDPOLatticeFactoryPutArray<T>),
               s1.size()*sizeof(T), 
	       sizeof(typename WordType<T>::Type_t), 
	       (void*)&s1) != QIO_SUCCESS)
  {
    QDPIO::cerr << "QDPFileReader: error reading file" << endl;
    clear(QDPIO_badbit);
  }
  QDPIO::cout << "QIO_read finished " << endl  << flush ;

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


//! Function for moving data
/*!
  Data is moved from the one buffer to another with a specified offset.

  \param buf The destination buffer
  \param linear The source buffer offset
  \param count The number of data to move
  \param arg The source buffer.
*/

template<class T> void QDPOLatticeFactoryGet(char *buf, size_t linear, int count, void *arg)
{
  /* Translate arg */
  T *field = (T *)arg;

  void *src = (void*)(field+linear);
  memcpy(buf,(const void*)src,count*sizeof(T));
}

//! Function for moving array data
/*!
  Data is moved from the one buffer to another with a specified offset.
  The data is taken to be in multi1d< OLattice<T> > form.

  \param buf The source buffer
  \param linear The source buffer offset
  \param count Ignored
  \param arg The destination buffer.
*/
template<class T> void QDPOLatticeFactoryGetArray(char *buf, size_t linear, int count, void *arg)
{
  /* Translate arg */
  multi1d< OLattice<T> >& field = *(multi1d< OLattice<T> > *)arg;

  for(int i=0; i < field.size(); ++i)
  {
    void *src = (void*)&(field[i].elem(linear));
    memcpy(buf,(const void*)src,sizeof(T));
    buf += sizeof(T);
  }
}



//! Writes an OLattice object
/*!
  This implementation is only correct for scalar ILattice.

  \param rec_xml The user record metadata.
  \param sl The data
*/
template<class T>
void QDPFileWriter::write(XMLBufferWriter& rec_xml, const OLattice<T>& s1)
{

  std::string tname;
  //  char tprec;
  char tprec[2]={0,'\0'};
  
  QIOStrings::QIOTypeStringFromType(tname, s1);
  tprec[0] = QIOStrings::QIOPrecisionStringFromType(s1);
  
  QIO_RecordInfo* info = QIO_create_record_info(QIO_FIELD, 
						const_cast<char *>(tname.c_str()),
						tprec,
						Nc, Ns, 
						sizeof(T),1 );
  
  // Copy metadata string into simple qio string container
  QIO_String* xml_c = QIO_string_create();
  if (Layout::primaryNode())
    QIO_string_set(xml_c, rec_xml.str().c_str());

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


//! Writes an array of OLattice objects
/*!
  This implementation is only correct for scalar ILattice.

  \param rec_xml The (user) record metadata.
  \param sl The data
*/
template<class T>
void QDPFileWriter::write(XMLBufferWriter& rec_xml, const multi1d< OLattice<T> >& s1)
{

  std::string tname;
  char tprec[2] = {0,'\0'};
  
  QIOStrings::QIOTypeStringFromType(tname, s1);
  tprec[0] = QIOStrings::QIOPrecisionStringFromType(s1);

  
  
  QIO_RecordInfo* info = QIO_create_record_info(QIO_FIELD, 
						const_cast<char *>(tname.c_str()),
						tprec,
						Nc, Ns, 
						sizeof(T), s1.size() );

  // Copy metadata string into simple qio string container
  QIO_String* xml_c = QIO_string_create();
  if (Layout::primaryNode())
    QIO_string_set(xml_c, rec_xml.str().c_str());

  if (xml_c == NULL)
  {
    QDPIO::cerr << "QDPFileWriter::write - error in creating XML string" << endl;
    QDP_abort(1);
  }

  // Big call to qio
  if (QIO_write(get(), info, xml_c,
	        &(QDPOLatticeFactoryGetArray<T>),
                s1.size()*sizeof(T), 
 	        sizeof(typename WordType<T>::Type_t), 
		(void*)&s1) != QIO_SUCCESS)
  {
    QDPIO::cerr << "QDPFileWriter: error in write" << endl;
    clear(QDPIO_badbit);
  }

  // Cleanup
  QIO_string_destroy(xml_c);
  QIO_destroy_record_info(info);
}


/*! @} */   // end of group qio
QDP_END_NAMESPACE();

#endif
