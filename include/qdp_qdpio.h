// -*- C++ -*-

/*! @file
 * @brief IO support via QIO
 */

#ifndef QDP_QDPIO_H
#define QDP_QDPIO_H

#include <qdp_config.h>
#include "qio.h"
#include <sstream>
#include "qdp_defs.h"
#include <cstring>

using namespace std;
#include "qdp_byteorder.h"

namespace QDP { 
 //-------------------------------------------------
  // QIO support
  //
  // Scalar support
  
  //! Function for moving data
  /*!
    Data is moved from the one buffer to another with a specified offset.


     \param buf The source buffer
     \param linear site index
     \param count The number of data to move
     \param arg The destination buffer.
   */
   template<class T> 
   void QDPOScalarFactoryPut(char *buf, size_t linear, int count, void *arg) 
   {


     // The destination buffer is a QDP++ field. It is an OScalar<T>

     // The source buffer is a number of sites. 
     // count is the size in bytes....
     T *field = (T *)arg;
     if ( linear != 0 ) { 
       QDPIO::cerr << "Called QDPOScalarFactory put with a nonzero linear_index" << linear << endl;
       QDP_abort(1);
     }

     memcpy(field,(const void*)buf,count*sizeof(T));
   

   }

   //! Function for moving data
   /*!
     Data is moved from the one buffer to another with a specified offset.
     This only works if the source is OScalar<...<IScalar<>>

     \param buf The destination buffer
     \param linear The source buffer offset
     \param count The number of data to move
     \param arg The source buffer.
   */

   template<class T> 
   void QDPOScalarFactoryGet(char *buf, size_t linear, int count, void *arg)
   {
     /* Translate arg */
     T *field = (T *)arg;

     if ( linear != 0 ) { 
       QDPIO::cout << " QDPOScalarFactoryGet was called with nonzero linear: " << linear << endl;
       QDP_abort(1);
     }

     void *src = (void*)field;
     memcpy(buf,(const void*)src,count*sizeof(T));

   }

   //! Function for moving data
   /*!
     Data is moved from the QIO buffer into the QDP++ buffers. QIO keeps
     data in a site based format. QDP++ may have an ILattice so we need to be carefule

     \param buf The source QIO buffer (single site)
     \param linear The linear index.
     \param count The number of data to move (this should always be just 1)
     \param arg The destination QDP++ buffer (may contain ILattice)
   */
   template<class T> 
   void QDPOLatticeFactoryPut(char *buf, size_t linear, int count, void *arg)
   {
     typedef typename UnaryReturn<T, FnGetSite>::Type_t  Site_t;
     /* Translate arg */
     T *field = (T *)arg;

     // Linear is a site index. Let us decompose it:
     //  QDPIO::cout << "QDPOLatticeFactoryPut called with linear=" << linear << " count=" << count << ",  sizeof(site(T))=" << sizeof(typename UnaryReturn<T,FnGetSite>::Type_t) << endl;

    
     int outer = linear >> INNER_LOG;
     int inner = linear & ( INNER_LEN - 1);


     Site_t res_site;
     memcpy((void *)&res_site,(const void*)buf,count*sizeof(Site_t));
     T block = field[outer];
     copy_site(block,inner,res_site);
     field[outer]=block;
   }

   //! Function for moving array data
   /*!
     Data is moved from QIO buffer to QDP++ Field
     The data is taken to be in multi1d< OLattice<T> > form.
     The T may contain inner sites whereas the QIO is always site based

     \param buf The source QIO buffer
     \param linear site index
     \param count Ignored
     \param arg QDP++ buffer
   */
   template<class T> 
   void QDPOLatticeFactoryPutArray(char *buf, size_t linear, int count, void *arg)
   {
     /* Translate arg */
     multi1d< OLattice<T> >& field = *(multi1d< OLattice<T> > *)arg;
     typedef typename UnaryReturn<T,FnGetSite>::Type_t Site_t;
     int outer = linear >> INNER_LOG;
     int inner = linear & (INNER_LEN-1 );

     for(int i=0; i < field.size(); ++i) {
       Site_t res_site ;
       void *dest = (void*)&res_site;
       memcpy(dest,(const void*)buf,sizeof(Site_t));
       buf += sizeof(Site_t);
       copy_site(field[i].elem(outer), inner, res_site);
    }

  }
  
    //! Function for moving data
  /*!
    Data is moved from QDP++ Buffer into the QIO buffer. QDP may have an ILattice<> but QIO does not.
    QIO Gives the offsets in terms of sites.

    \param buf The destination buffer QIO buffer (single site)
    \param linear The site index
    \param count The number of data to move (The way we use QIO this ought to always be 1 for OLattiec Data)
    \param arg The source buffer (QDP Field)
  */

  template<class T> void QDPOLatticeFactoryGet(char *buf, size_t linear, int count, void *arg)
  {
    typedef typename UnaryReturn<T,FnGetSite>::Type_t  Site_t;

    /* Translate arg */
    T* field = (T *)arg;

    int osite = linear >> INNER_LOG;
    int isite = linear & (INNER_LEN - 1);

    T outer_block = field[osite];
    
    Site_t the_site = getSite(outer_block, isite);
    //   QDPIO::cout << "size=" << count*sizeof(Site_t) << endl << flush;
    memcpy(buf,(const void*)&the_site,count*sizeof(Site_t));
 
#if 1  
    size_t bytes = sizeof(Site_t);
    size_t wordsize = sizeof(typename WordType<Site_t>::Type_t);
    size_t numnums = bytes/wordsize;
    QDPUtil::byte_swap(buf, wordsize, numnums);
#endif

  }

  //! Function for moving array data
  /*!
    Data is moved from the QDP++ buffer to the QIO one, with a specified offset.
    The data is taken to be in multi1d< OLattice<T> > form.

    \param buf The source (QDP++) buffer;
    \param linear The linear site
    \param count Ignored (set by datum size, and array size)
    \param arg The destination buffer QIO buffer.
  */
  template<class T>
  void QDPOLatticeFactoryGetArray(char *buf, size_t linear, int count, void *arg)
  {
    multi1d< OLattice<T> >& field = *(multi1d< OLattice<T> > *)arg;
    typedef typename UnaryReturn<T,FnGetSite>::Type_t Site_t;
    int osite = linear >> INNER_LOG;
    int isite = linear & (INNER_LEN - 1);
    size_t bytes = sizeof(Site_t);
    size_t wordsize = sizeof(typename WordType<Site_t>::Type_t);
    size_t numnums = bytes/wordsize;

    
    for(int i=0; i < field.size(); ++i)
    {
      Site_t the_site = getSite(field[i].elem(osite), isite);
      memcpy(buf,(const void*)&the_site,sizeof(Site_t));
      QDPUtil::byte_swap(buf, wordsize, numnums);
      buf += sizeof(Site_t);
    }

  }

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

#if 0
  // This code was to support better partfile IO on the QCDOC 
  // However I have commented it out because it is not clear
  // at this time in the API how to pass this information 
  // town to the QIO. A straightforward hack is to modify 
  // the QIO_Layout structure, but we have not actually agreed
  // with Carleton that that is what I should do. The placement
  // of the choice for a particular kind of partitioning scheme
  // is not yet present -- will it be in QIO, will it be in QMP?
  // will it be here? WIll it be configure/runtime? We just don't
  // know.
  namespace SingleFileIONode { 
    int IONode(int node);
    int masterIONode(void);
  };

  namespace MultiFileIONode {
    int IONode(int node);
    int masterIONode(void);
  };

  namespace PartFileIONode { 
    int IONode(int node);
    int masterIONode(void);
  };
#endif

  //! QIO Type and precision strings. 
  //  Using the magic of C++ I can define the right type and precision
  //  strings i Need to pass to QIO using templates. To do this I need
  //  templated structures with static members.

  //! Catch all case
  template<typename T>
  struct QIOStringTraits 
  {
    static char* tname;
    static char* tprec;
  };

  //! Partial(?) Specialisation for OLattice Objects
  template<typename T>
  struct QIOStringTraits<OLattice<T> >
  {
    static char* tname;
    static char* tprec;
  };

  //! Partial(?) Specialisation for OScalar Objects
  template<typename T>
  struct QIOStringTraits<OScalar<T> >
  {
    static char* tname;
    static char* tprec;
  };

  //! Partial(?) Specialisation for OLattice Objects
  template<typename T>
  struct QIOStringTraits<multi1d< OLattice<T> > >
  {
    static char* tname;
    static char* tprec;
  };

  //! Partial(?) Specialisation for OScalar Objects
  template<typename T>
  struct QIOStringTraits<multi1d< OScalar<T> > >
  {
    static char* tname;
    static char* tprec;
  };

  //! Generic type
  template<typename T>
  char* QIOStringTraits<T>::tname = (char *)"QDP_GenericType";

  //! Lattice Type
  template<typename T> 
  char* QIOStringTraits< OLattice<T> >::tname = (char *)"Lattice";

  //! Scalar Type
  template<typename T> 
  char* QIOStringTraits< OScalar<T> >::tname = (char *)"Scalar";

  //! multi1d<LatticeType>
  template<typename T> 
  char* QIOStringTraits< multi1d<OLattice<T> > >::tname = (char *)"Lattice";

  //! multi1d<ScalarType>
  template<typename T> 
  char* QIOStringTraits< multi1d<OScalar<T> > >::tname = (char *)"Scalar";

  //! Unknown precision string
  template<typename T>
  char*  QIOStringTraits<T>::tprec = (char *)"U"; 

  // Full specialisations deferred to the qdp_qio_strings.cc file
  template<>
  char* QIOStringTraits<float>::tprec;

  template<>
  char* QIOStringTraits<double>::tprec;

  template<>
  char* QIOStringTraits<int>::tprec;
  
  template<>
  char* QIOStringTraits< multi1d<LatticeColorMatrixF3> >::tname;
  
  template<>
  char* QIOStringTraits< multi1d<LatticeColorMatrixD3> >::tname;

  template<>
  char* QIOStringTraits< multi1d<LatticeDiracFermionF3> >::tname;
  
  template<>
  char* QIOStringTraits< multi1d<LatticeDiracFermionD3> >::tname;


  // Most types have no colors
  template<typename T>
  struct NumColors {
    static const int value=0;
  };
  
  // Recursion base cases...
  template<typename T, const int N>
  struct NumColors< PColorVector<T,N> > {
    static const int value = N;
  };
  
  template<typename T, const int N>
  struct NumColors< PColorMatrix<T,N> > {
    static const int value = N;
  };
  
  // Top level recursion
  template<typename T, template<typename> class C>
  struct NumColors< C< T > > {
    static const int value = NumColors<T>::value;
  };

  // Most types have no colors
  template<typename T>
  struct NumSpins {
    static const int value=0;
  };
  
  // Recursion base cases...
  template<typename T, const int N>
  struct NumSpins< PSpinVector<T,N> > {
    static const int value = N;
  };
  
  template<typename T, const int N>
  struct NumSpins< PSpinMatrix<T,N> > {
    static const int value = N;
  };
  
  // Top level recursion
  template<typename T, template<typename> class C>
  struct NumSpins< C< T > > {
    static const int value = NumSpins<T>::value;
  };
  

    
    

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

    //! Reads a record header only 
    void read(XMLReader& xml);

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

    //! Reads an XMLReader plus BinaryBufferReader pair
    void read(XMLReader& xml, BinaryBufferReader& s1);

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

  //! Reads a BinaryBufferReader object
  /*!
    \param qsw The reader
    \param rec_xml The user record metadata.
    \param sl The data
  */
  inline
  void read(QDPFileReader& qsw, XMLReader& rec_xml, BinaryBufferReader& s1)
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

    //! Writes a hypercube from an OLattice object
    /*!
      \param xml The user record metadata.
      \param sl The data
      \param lower_left a multi1d<int> holding coordinates of lower left corner of the hypercube
      \param upper_right a multi1d<int> holding coordinates of the upper right corner of the hypercube
    */
    template<class T>
    void write(XMLBufferWriter& xml, const OLattice<T>& s1,
	       const multi1d<int>& lower_left, 
	       const multi1d<int>& upper_right);

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


    //! Writes a hypercube of an array of objects all to a single record
    /*!
      \param xml The user record metadata.
      \param sl The data
      \param lower_left a multi1d<int> holding coordinates of lower left corner of the hypercube
      \param upper_right a multi1d<int> holding coordinates of the upper right corner of the hypercube
    */
    template<class T>
    void write(XMLBufferWriter& xml, const multi1d< OLattice<T> >& s1,
	       const multi1d<int>& lower_left, 
	       const multi1d<int>& upper_right);


    //! Writes an XML plus Binary writer pair
    /*!
      \param xml The user record metadata.
      \param sl The data
    */
    void write(XMLBufferWriter& xml, BinaryBufferWriter& s1);

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

  //! Writes a hypercube part of an OLattice object
  /*!
    \param qsw The writer
    \param xml The user record metadata.
    \param sl The data
    \param lower_left A multi1d of integers containing the lower left lower left corner of the hypercube 
    \param upper_right A multi1d of integers containing the coordinates of the upper right corner of the hypercube
  */
  template<class T>
  void write(QDPFileWriter& qsw, XMLBufferWriter& rec_xml, const OLattice<T>& s1, const multi1d<int>& lower_left, const multi1d<int>& upper_right)
  {
    qsw.write(rec_xml,s1, lower_left, upper_right);
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

  //! Writes an array of OLattice objects
  /*!
    \param qsw The writer
    \param xml The user record metadata.
    \param sl The data
    \param lower_left A multi1d of integers containing the lower left lower left corner of the hypercube 
    \param upper_right A multi1d of integers containing the coordinates of the upper right corner of the hypercube
  */
  template<class T>
  void write(QDPFileWriter& qsw, XMLBufferWriter& rec_xml, const multi1d< OLattice<T> >& s1, const multi1d<int>& lower_left, const multi1d<int>& upper_right)
  {
    qsw.write(rec_xml,s1,lower_left, upper_right);
  }

  //! Writes a XML plus BinaryBufferWriter pair
  /*!
    \param qsw The writer
    \param xml The user record metadata.
    \param sl The data
  */
  inline
  void write(QDPFileWriter& qsw, XMLBufferWriter& rec_xml, BinaryBufferWriter& s1)
  {
    qsw.write(rec_xml,s1);
  }

  //! Closes a QDPFileWriter.
  void close(QDPFileWriter& qsw);

  //! Queries whether a QDPFileWriter is open.
  bool is_open(QDPFileWriter& qsw);





  //! Reads an OScalar object
  /*!
    This implementation is only correct for IScalar
    \param rec_xml The (user) record metadata.
    \param sl The data
  */

  template<typename T>
  void QDPFileReader::read(XMLReader& rec_xml, OScalar<T>& s1)
  {
    /* For now I may not be able to read Dirk's stuff, but I should
     * be able to read what I wrote */
    QIO_RecordInfo rec_info;
    QIO_String* xml_c = QIO_string_create();
    int status;
  
  
    status=QIO_read_record_info(qio_in, &rec_info, xml_c);
    if( status != QIO_SUCCESS) { 
      QDPIO::cerr << "Failed to read the Record Info" << endl;
      QDP_abort(1);
    }
  
    switch( (QIO_get_precision(&rec_info))[0] ) { 
    case 'F' :
    {
      QDPIO::cout << "Single Precision Read" << endl;
      OScalar< typename SinglePrecType<T>::Type_t > from_disk;
      status = QIO_read_record_data(qio_in,
				    &(QDPOScalarFactoryPut<typename SinglePrecType<T>::Type_t> ),
				    sizeof(typename SinglePrecType<T>::Type_t),
				    sizeof(typename WordType< typename SinglePrecType<T>::Type_t >::Type_t),
				    (void *)(&(from_disk.elem())));
      if (status != QIO_SUCCESS) { 
	QDPIO::cerr << "Failed to read data" << endl;
	clear(QDPIO_badbit);
	QDP_abort(1);
      }
      QDPIO::cout << "QIO_read_finished" << endl;
      s1 = from_disk;
    }
    break;
    case 'D' :
    {
      QDPIO::cout << "Reading Double Precision" << endl;
      OScalar< typename DoublePrecType<T>::Type_t > from_disk;
      status = QIO_read_record_data(qio_in,
				    &(QDPOScalarFactoryPut< typename DoublePrecType<T>::Type_t > ),
				    sizeof(typename DoublePrecType<T>::Type_t),
				    sizeof(typename WordType< typename DoublePrecType<T>::Type_t >::Type_t),
				    (void *)(&(from_disk.elem())));
      if (status != QIO_SUCCESS) { 
	QDPIO::cerr << "Failed to read data" << endl;
	clear(QDPIO_badbit);
	QDP_abort(1);
      }
      QDPIO::cout << "QIO_read_finished" << endl;
      
      s1 = from_disk;
    }
    break;
    default:
    {
      QDPIO::cout << "Reading I or U Precision" << endl;
      status = QIO_read_record_data(qio_in,
				    &(QDPOScalarFactoryPut<T> ),
				    sizeof(T),
				    sizeof(typename WordType<T>::Type_t),
				    (void *)(&(s1.elem())));
      if (status != QIO_SUCCESS) { 
	QDPIO::cerr << "Failed to read data" << endl;
	clear(QDPIO_badbit);
	QDP_abort(1);
      }
      QDPIO::cout << "QIO_read_finished" << endl;
    }
    break;
    }
  
    istringstream ss;
    if (Layout::primaryNode()) {
      string foo = QIO_string_ptr(xml_c);
      ss.str(foo);
    }
    rec_xml.open(ss);
  
    QIO_string_destroy(xml_c);
  }
  

  //! Reads an array of OScalar objects
  /*!
    This implementation is only correct for scalar ILattice

    \param rec_xml The (user) record metadata.
    \param sl The data
  */

  template<typename T>
  void QDPFileReader::read(XMLReader& rec_xml, multi1d< OScalar<T> >& s1)
  {
    /* For now I may not be able to read Dirk's stuff, but I should
     * be able to read what I wrote */
    QIO_RecordInfo rec_info;
    QIO_String* xml_c = QIO_string_create();
    int status;
  
  
    status=QIO_read_record_info(qio_in, &rec_info, xml_c);
    if( status != QIO_SUCCESS) { 
      QDPIO::cerr << "Failed to read the Record Info" << endl;
      QDP_abort(1);
    }
  
    switch( (QIO_get_precision(&rec_info))[0] ) { 
    case 'F' :
    {
      QDPIO::cout << "Single Precision Read" << endl;
      multi1d< OScalar< typename SinglePrecType<T>::Type_t > > from_disk(s1.size());
      status = QIO_read_record_data(qio_in,
				    &(QDPOScalarFactoryPut<typename SinglePrecType<T>::Type_t> ),
				    s1.size()*sizeof(typename SinglePrecType<T>::Type_t),
				    sizeof(typename WordType< typename SinglePrecType<T>::Type_t >::Type_t),
				    (void *)from_disk.slice());
      if (status != QIO_SUCCESS) { 
	QDPIO::cerr << "Failed to read data" << endl;
	clear(QDPIO_badbit);
	QDP_abort(1);
      }
      QDPIO::cout << "QIO_read_finished" << endl;
      
      // Cast appropriately
      for(int i=0; i < from_disk.size(); i++) { 
	s1[i] = from_disk[i];
      }
      
    }
    break;
    case 'D' :
    {
      QDPIO::cout << "Reading Double Precision" << endl;
      multi1d< typename DoublePrecType< OScalar<T> >::Type_t > from_disk(s1.size());
      status = QIO_read_record_data(qio_in,
				    &(QDPOScalarFactoryPut< typename DoublePrecType<T>::Type_t > ),
				    s1.size()*sizeof(typename DoublePrecType<T>::Type_t),
				    sizeof(typename WordType< typename DoublePrecType<T>::Type_t >::Type_t),
				    (void *)from_disk.slice());
      if (status != QIO_SUCCESS) { 
	QDPIO::cerr << "Failed to read data" << endl;
	clear(QDPIO_badbit);
	QDP_abort(1);
      }
      QDPIO::cout << "QIO_read_finished" << endl;
      
      // Cast appropriately
      for(int i=0; i < from_disk.size(); i++) { 
	s1[i] = from_disk[i];
      }
    }
    break;
    default:
    {
      QDPIO::cout << "Reading I or U Precision" << endl;
      status = QIO_read_record_data(qio_in,
				    &(QDPOScalarFactoryPut<T> ),
				    s1.size()*sizeof(T),
				    sizeof(typename WordType<T>::Type_t),
				    (void *)s1.slice());
      if (status != QIO_SUCCESS) { 
	QDPIO::cerr << "Failed to read data" << endl;
	clear(QDPIO_badbit);
	QDP_abort(1);
      }
      QDPIO::cout << "QIO_read_finished" << endl;
    }
    break;
    }
  
    istringstream ss;
    if (Layout::primaryNode()) {
      string foo = QIO_string_ptr(xml_c);
      ss.str(foo);
    }
    rec_xml.open(ss);
  
    QIO_string_destroy(xml_c);
  }


  // Reads a BinaryBufferReader object
  /*!
    \param rec_xml The (user) record metadata.
    \param sl The data
  */
//  void QDPFileReader::read(XMLReader& rec_xml, BinaryBufferReader& s1);




  //! Writes an OScalar object
  /*!
    This implementation is only correct for scalar ILattice.

    \param rec_xml The (user) record metadata.
    \param sl The data
  */
  template<class T>
  void QDPFileWriter::write(XMLBufferWriter& rec_xml, const OScalar<T>& s1)
  {
    QIO_RecordInfo* info = QIO_create_record_info(QIO_GLOBAL, NULL, NULL, 0,
						  QIOStringTraits< OScalar<T> >::tname,
						  QIOStringTraits< typename WordType<T>::Type_t >::tprec,
						  NumColors<T>::value, NumSpins<T>::value, 
						  sizeof(typename UnaryReturn<T,FnGetSite>::Type_t), 1);

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
		  (void *)(&(s1.elem()))) != QIO_SUCCESS)
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

  //! Reads an array of OLattice objects
  /*!
    This implementation is only correct for scalar ILattice.

    \param rec_xml The (user) record metadata.
    \param sl The data
  */
  template<class T>
  void QDPFileWriter::write(XMLBufferWriter& rec_xml, const multi1d< OScalar<T> >& s1)
  {
    QIO_RecordInfo* info = QIO_create_record_info(QIO_GLOBAL, NULL, NULL, 0,
						  QIOStringTraits<multi1d< OScalar<T> > >::tname,
						  QIOStringTraits<typename WordType<T>::Type_t>::tprec, 
						  NumColors<T>::value, NumSpins<T>::value, 
						  sizeof(typename UnaryReturn<T,FnGetSite>::Type_t), s1.size());


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


  //! Writes a BinaryBufferWriter object
  /*!
    \param rec_xml The (user) record metadata.
    \param sl The data
  */
//  void QDPFileWriter::write(XMLBufferWriter& rec_xml, BinaryBufferWriter& s1);


  //-------------------------------------------------
  // QIO support
  // NOTE: this is exactly the same bit of code as in scalar_specific.h 
  //       need to make common only on scalarsite.h  like architectures




  //! Reads an OLattice object
  /*!
    This implementation is only correct for scalar ILattice.

    \param rec_xml The (user) record metadata.
    \param sl The data
  */
  template<class T>
  void QDPFileReader::read(XMLReader& rec_xml, OLattice<T>& s1)
  {
    /* For now I may not be able to read Dirk's stuff, but I should
     * be able to read what I wrote */
    QIO_RecordInfo rec_info;
    QIO_String* xml_c = QIO_string_create();
    int status;
    status=QIO_read_record_info(qio_in, &rec_info, xml_c);
    if( status != QIO_SUCCESS) { 
      QDPIO::cerr << "Failed to read the Record Info" << endl;
      QDP_abort(1);
    }
    
    
    switch( (QIO_get_precision(&rec_info))[0] ) { 
    case 'F' :
    {
      QDPIO::cout << "Single Precision Read" << endl;
      OLattice< typename SinglePrecType<T>::Type_t > from_disk;
      zero_rep(from_disk);

      
      status = QIO_read_record_data(qio_in,
				    &(QDPOLatticeFactoryPut<typename SinglePrecType<T>::Type_t> ),
				    sizeof(typename UnaryReturn<typename SinglePrecType<T>::Type_t, FnGetSite>::Type_t ),
				    sizeof(typename WordType<typename UnaryReturn<typename SinglePrecType<T>::Type_t, FnGetSite>::Type_t>::Type_t ),
				    (void *)from_disk.getF());
      
      if (status != QIO_SUCCESS) { 
	QDPIO::cerr << "Failed to read data" << endl;
	clear(QDPIO_badbit);
	QDP_abort(1);
      }
      QDPIO::cout << "QIO_read_finished" << endl;
      s1 = from_disk; // Cast
    }
    break;
    case 'D' :
    {
      QDPIO::cout << "Reading Double Precision" << endl;
      OLattice< typename DoublePrecType<T>::Type_t > from_disk;
      zero_rep(from_disk);

      /* Disagnostics */
      status = QIO_read_record_data(qio_in,
				    &(QDPOLatticeFactoryPut< typename DoublePrecType<T>::Type_t>),
				    sizeof(typename UnaryReturn<typename DoublePrecType<T>::Type_t, FnGetSite>::Type_t ),
				    sizeof(typename WordType<typename UnaryReturn<typename DoublePrecType<T>::Type_t, FnGetSite>::Type_t>::Type_t ),
				    (void *)from_disk.getF());
      
      if (status != QIO_SUCCESS) { 
	QDPIO::cerr << "Failed to read data" << endl;
	clear(QDPIO_badbit);
	QDP_abort(1);
      }
      QDPIO::cout << "QIO_read_finished" << endl;
      
      // Cast appropriately
      s1= from_disk;
	
    }
    break;
    default:
    {
      QDPIO::cout << "Reading I or U precisions" << endl;
      status = QIO_read_record_data(qio_in,
				    &(QDPOLatticeFactoryPut<T>),
				    sizeof(typename UnaryReturn<T,FnGetSite>::Type_t),
				    sizeof(typename WordType<typename UnaryReturn<T,FnGetSite>::Type_t>::Type_t),
				    (void *)s1.getF());
     
      if (status != QIO_SUCCESS) { 
	QDPIO::cerr << "Failed to read data" << endl;
	clear(QDPIO_badbit);
	QDP_abort(1);
      }
      QDPIO::cout << "QIO_read_finished" << endl;
    }
    break;
    };
        
    istringstream ss;
    if (Layout::primaryNode()) {
      string foo = QIO_string_ptr(xml_c);
      ss.str(foo);
    }
    rec_xml.open(ss);
  
    QIO_string_destroy(xml_c);
  }

  //! Reads an array of OLattice objects
  /*!
    This implementation is only correct for scalar ILattice.

    \param rec_xml The (user) record metadata.
    \param sl The data
  */
  template<typename T>
  void QDPFileReader::read(XMLReader& rec_xml, multi1d< OLattice<T> >& s1)
  {
    /* For now I may not be able to read Dirk's stuff, but I should
     * be able to read what I wrote */
    QIO_RecordInfo rec_info;
    QIO_String* xml_c = QIO_string_create();
    int status;
  
  
    status=QIO_read_record_info(qio_in, &rec_info, xml_c);
    if( status != QIO_SUCCESS) { 
      QDPIO::cerr << "Failed to read the Record Info" << endl;
      QDP_abort(1);
    }
  
    switch( (QIO_get_precision(&rec_info))[0] ) { 
    case 'F' :
    {
      QDPIO::cout << "Single Precision Read" << endl;
      multi1d< OLattice< typename SinglePrecType<T>::Type_t > > from_disk(s1.size());
      for(int i=0; i < s1.size(); i++) { zero_rep(from_disk[i]); }

      status = QIO_read_record_data(qio_in,
				    &(QDPOLatticeFactoryPutArray<typename SinglePrecType<T>::Type_t> ),
				    s1.size()*sizeof(typename UnaryReturn<typename SinglePrecType<T>::Type_t, FnGetSite>::Type_t ),
				    sizeof(typename WordType<typename UnaryReturn<typename SinglePrecType<T>::Type_t, FnGetSite>::Type_t>::Type_t ),
				    (void *)&from_disk);
      if (status != QIO_SUCCESS) { 
	QDPIO::cerr << "Failed to read data" << endl;
	clear(QDPIO_badbit);
	QDP_abort(1);
      }
      QDPIO::cout << "QIO_read_finished" << endl;
      
      // Cast appropriately
      for(int i=0; i < from_disk.size(); i++) { 
	s1[i] = from_disk[i];
      }
      
    }
    break;
    case 'D' :
    {
      QDPIO::cout << "Reading Double Precision" << endl;
      multi1d< typename DoublePrecType< OLattice<T> >::Type_t > from_disk(s1.size());
      for(int i=0; i < s1.size(); i++) { zero_rep(from_disk[i]); }

      status = QIO_read_record_data(qio_in,
				    &(QDPOLatticeFactoryPutArray< typename DoublePrecType<T>::Type_t > ),
				    s1.size()*sizeof(typename UnaryReturn<typename DoublePrecType<T>::Type_t, FnGetSite>::Type_t ),
				    sizeof(typename WordType<typename UnaryReturn<typename DoublePrecType<T>::Type_t, FnGetSite>::Type_t>::Type_t ),
				    (void *)&from_disk);
      if (status != QIO_SUCCESS) { 
	QDPIO::cerr << "Failed to read data" << endl;
	clear(QDPIO_badbit);
	QDP_abort(1);
      }
      QDPIO::cout << "QIO_read_finished" << endl;
      
      // Cast appropriately
      for(int i=0; i < from_disk.size(); i++) { 
	s1[i] = from_disk[i];
      }
    }
    break;
    default:
    {
      QDPIO::cout << "Reading I or U Precision" << endl;
      status = QIO_read_record_data(qio_in,
				    &(QDPOLatticeFactoryPutArray<T> ),
				    s1.size()*sizeof(typename UnaryReturn<T,FnGetSite>::Type_t),
				    sizeof(typename WordType<typename UnaryReturn<T,FnGetSite>::Type_t>::Type_t   )  ,
				    (void *)&s1);
      if (status != QIO_SUCCESS) { 
	QDPIO::cerr << "Failed to read data" << endl;
	clear(QDPIO_badbit);
	QDP_abort(1);
      }
      QDPIO::cout << "QIO_read_finished" << endl;
    }
    break;
    }
  
    istringstream ss;
    if (Layout::primaryNode()) {
      string foo = QIO_string_ptr(xml_c);
      ss.str(foo);
    }

    try { 
      rec_xml.open(ss);
    }
    catch(const std::string& e) { 
      QDPIO::cout << "Handling exception" << endl;
    }
  
    QIO_string_destroy(xml_c);
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


    QDPIO::cout << "Typename is " <<  QIOStringTraits< OLattice<T> >::tname <<endl;
    QDPIO::cout << "Precision is " <<  QIOStringTraits<typename WordType<T>::Type_t >::tprec << endl;
    QDPIO::cout << "N Colors is " << NumColors<T>::value << endl;
    QDPIO::cout << "N Spins is " << NumSpins<T>::value << endl;
    QDPIO::cout << "Datum is " << sizeof(typename UnaryReturn<T,FnGetSite>::Type_t ) << endl;

    QIO_RecordInfo* info = QIO_create_record_info(QIO_FIELD, NULL, NULL,0,
						  QIOStringTraits< OLattice<T> >::tname,
						  QIOStringTraits<typename WordType<T>::Type_t >::tprec,
						  Nc,Ns, 
						  sizeof(typename UnaryReturn<T,FnGetSite>::Type_t ),1 );
  
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
		  sizeof(typename UnaryReturn<T,FnGetSite>::Type_t), 
		  sizeof(typename WordType< UnaryReturn<T,FnGetSite>   >::Type_t), 
		  (void *)s1.getF()) != QIO_SUCCESS)
    {
      QDPIO::cerr << "QDPFileWriter: error in write" << endl;
      clear(QDPIO_badbit);
    }

    // Cleanup
    QIO_string_destroy(xml_c);
    QIO_destroy_record_info(info);
  }


  //! Writes a hypercube from an  OLattice object
  /*!
    This implementation is only correct for scalar ILattice.

    \param rec_xml The user record metadata.
    \param sl The data
    \param lower_left A multi1d of integers containing the lower left lower left corner of the hypercube 
    \param upper_right A multi1d of integers containing the coordinates of the upper right corner of the hypercube
  */
  template<class T>
  void QDPFileWriter::write(XMLBufferWriter& rec_xml, const OLattice<T>& s1, const multi1d<int>& lower_left, const multi1d<int>& upper_right)
  {

    // Sanity check...
    if( lower_left.size() != upper_right.size()) {
	QDPIO::cerr << "QDPFileWriter: Error! Lower left and upper right corner of hypercube to write have different dimensions" << endl;
	QDP_abort(1);
    }

    QIO_RecordInfo* info = QIO_create_record_info(QIO_HYPER, 
						  (int *)lower_left.slice(), 
						  (int *)upper_right.slice(),
						  lower_left.size(),
						  QIOStringTraits< OLattice<T> >::tname,
						  QIOStringTraits<typename WordType<T>::Type_t >::tprec,
						  NumColors<T>::value, NumSpins<T>::value, 
						  sizeof(typename UnaryReturn<T,FnGetSite>::Type_t),1 );
  
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
		  sizeof(typename UnaryReturn<T,FnGetSite>::Type_t), 
		  sizeof(typename WordType< UnaryReturn<T,FnGetSite>   >::Type_t), 
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

QDPIO::cout << "Typename is " <<  QIOStringTraits< OLattice<T> >::tname <<endl;
    QDPIO::cout << "Precision is " <<  QIOStringTraits<typename WordType<T>::Type_t >::tprec << endl;
    QDPIO::cout << "N Colors is " << NumColors<T>::value << endl;
    QDPIO::cout << "N Spins is " << NumSpins<T>::value << endl;
    QDPIO::cout << "Datum is " << sizeof(typename UnaryReturn<T,FnGetSite>::Type_t ) << endl;
    QDPIO::cout << "Num Dat is " << s1.size() << endl;

    QIO_RecordInfo* info = QIO_create_record_info(QIO_FIELD, 
						  NULL, NULL, 0,
						  QIOStringTraits<multi1d< OLattice<T> > >::tname,
						  QIOStringTraits<typename WordType<T>::Type_t>::tprec,
						  NumColors<T>::value, NumSpins<T>::value, 
						  sizeof(typename UnaryReturn<T,FnGetSite>::Type_t), s1.size() );

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

		  s1.size()*sizeof(typename UnaryReturn<T,FnGetSite>::Type_t), 
		  sizeof(typename WordType< UnaryReturn<T,FnGetSite>   >::Type_t), 

		  (void*)&s1) != QIO_SUCCESS)
    {
      QDPIO::cerr << "QDPFileWriter: error in write" << endl;
      clear(QDPIO_badbit);
    }

    // Cleanup
    QIO_string_destroy(xml_c);
    QIO_destroy_record_info(info);
  }



  //! Writes a hypercube from an array of OLattice objects
  /*!
    This implementation is only correct for scalar ILattice.

    \param rec_xml The (user) record metadata.
    \param sl The data
    \param lower_left A multi1d of integers containing the lower left lower left corner of the hypercube 
    \param upper_right A multi1d of integers containing the coordinates of the upper right corner of the hypercube

  */
  template<class T>
  void QDPFileWriter::write(XMLBufferWriter& rec_xml, const multi1d< OLattice<T> >& s1, const multi1d<int>& lower_left, const multi1d<int>& upper_right)
  {

    // Sanity check...
    if( lower_left.size() != upper_right.size()) {
	QDPIO::cerr << "QDPFileWriter: Error! Lower left and upper right corner of hypercube to write have different dimensions" << endl;
	QDP_abort(1);
    }

    QIO_RecordInfo* info = QIO_create_record_info(QIO_HYPER, 
						  (int *)(lower_left.slice()),
						  (int *)(upper_right.slice()), 
						  lower_left.size(),
						  QIOStringTraits<multi1d< OLattice<T> > >::tname,
						  QIOStringTraits<typename WordType<T>::Type_t>::tprec,
						  NumColors<T>::value, NumSpins<T>::value, 
						  sizeof(typename UnaryReturn<T,FnGetSite>::Type_t), s1.size() );

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
		  s1.size()*sizeof(typename UnaryReturn<T,FnGetSite>::Type_t), 
		  sizeof(typename WordType< UnaryReturn<T,FnGetSite>   >::Type_t), 
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
} // namespace QDP

#endif
