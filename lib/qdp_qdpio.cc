// $Id: qdp_qdpio.cc,v 1.28 2007-08-14 03:08:50 edwards Exp $
//
/*! @file
 * @brief IO support via QIO
 */

#include "qdp.h"

namespace QDP 
{

  //-----------------------------------------
  static int get_node_number(const int coord[])
  {
    multi1d<int> crd(Nd);
    crd = coord;   // an array copy
    int node = Layout::nodeNumber(crd);
    return node;
  }

  static int get_node_index(const int coord[])
  {
    multi1d<int> crd(Nd);
    crd = coord;   // an array copy
    int linear = Layout::linearSiteIndex(crd);
    return linear;
  }

  static void get_coords(int coord[], int node, int linear)
  {
    multi1d<int> crd = Layout::siteCoords(node, linear);
    for(int i=0; i < Nd; ++i)
      coord[i] = crd[i];
  }

  static int get_sites_on_node(int node) 
  {
    return Layout::sitesOnNode();
  }


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

  //! A little namespace to mark I/O nodes
  // This was originally so that we could use part file better
  // but now is probably unused.
  namespace SingleFileIONode { 
    int IONode(int node) {
      return 0;
    }

    int masterIONode(void) {
      return 0;
    }
  }

  namespace MultiFileIONode {
    int IONode(int node) { 
      return node; 
    }

    int masterIONode(void) { 
      return DML_master_io_node();
    }
  }

  namespace PartFileIONode { 
    int IONode(int node) {
      // This code supports
      multi1d<int> my_coords = Layout::getLogicalCoordFrom(node);
      multi1d<int> io_node_coords(my_coords.size());
      for(int i=0; i < my_coords.size(); i++) { 
	io_node_coords[i] = 2*(my_coords[i]/2);
      }
      return Layout::getNodeNumberFrom(io_node_coords); 
      return DML_io_node(node);
    }
    int masterIONode(void) { 
      return DML_master_io_node();
    }
  }
#endif

  //-----------------------------------------------------------------------------
  // QDP QIO support
  QDPFileReader::QDPFileReader() {iop=false;}

  QDPFileReader::QDPFileReader(XMLReader& xml, 
			       const std::string& path,
			       QDP_serialparallel_t serpar)
  {open(xml,path,serpar);}

  void QDPFileReader::open(XMLReader& file_xml, 
			   const std::string& path, 
			   QDP_serialparallel_t serpar)
  {
    QIO_Layout layout;

    int latsize[Nd];

    for(int m=0; m < Nd; ++m)
      latsize[m] = Layout::lattSize()[m];

    layout.node_number = &get_node_number;
    layout.node_index  = &get_node_index;
    layout.get_coords  = &get_coords;
    layout.num_sites = &get_sites_on_node;
    layout.latsize = latsize;
    layout.latdim = Nd; 
    layout.volume = Layout::vol(); 
    layout.sites_on_node = Layout::sitesOnNode(); 
    layout.this_node = Layout::nodeNumber(); 
    layout.number_of_nodes = Layout::numNodes(); 

    // Initialize string objects 
    QIO_String *xml_c  = QIO_string_create();


    // Call QIO read
    // At this moment, serpar (which is an enum in QDP++) is ignored here.
    if ((qio_in = QIO_open_read(xml_c, path.c_str(), &layout, NULL, NULL)) == NULL)
    {
      iostate = QDPIO_badbit;  // not helpful

      QDPIO::cerr << "QDPFileReader: failed to open file " << path << endl;
      QDP_abort(1);  // just bail, otherwise xml stuff below fails.
    }
    else
    {
      iostate = QDPIO_goodbit;
    }

    // Use string to initialize XMLReader
    istringstream ss;
    if (Layout::primaryNode())
    {
      string foo = QIO_string_ptr(xml_c);
      ss.str(foo);
    }
    file_xml.open(ss);

    QIO_string_destroy(xml_c);

    iop=true;
  }


  void QDPFileReader::close()
  {
    if (is_open()) 
    {
      //int status = QIO_close_read(qio_in);
      QIO_close_read(qio_in);
    }

    iop = false;
    iostate = QDPIO_badbit;
  }

  bool QDPFileReader::is_open() {return iop;}

  bool QDPFileReader::eof() const {return false;}

  bool QDPFileReader::bad() const {return iostate;}

  void QDPFileReader::clear(QDP_iostate_t state)
  {
    iostate = state;
  }

  QDPFileReader::~QDPFileReader() {close();}

  //! Close a QDPFileReader
  void close(QDPFileReader& qsw)
  {
    qsw.close();
  }

  //! Is a QDPFileReader open
  bool is_open(QDPFileReader& qsw)
  {
    return qsw.is_open();
  }


  // Reads a BinaryBufferReader object
  /*!
    \param rec_xml The (user) record metadata.
    \param sl The data
  */
  void QDPFileReader::read(XMLReader& rec_xml, BinaryBufferReader& s1)
  {
    QIO_RecordInfo rec_info;
    QIO_String* xml_c = QIO_string_create();
    int status;
  
    status = QIO_read_record_info(qio_in, &rec_info, xml_c);
    if( status != QIO_SUCCESS) { 
      QDPIO::cerr << "Failed to read the Record Info" << endl;
      QDP_abort(1);
    }
  
    QDPIO::cout << "BinaryBufferRead" << endl;
    std::string from_disk;
    from_disk.resize(QIO_get_datacount(&rec_info));
    status = QIO_read_record_data(qio_in,
				  &(QDPOScalarFactoryPut<char> ),
				  from_disk.size()*sizeof(char),
				  sizeof(char),
				  (void *)&(from_disk[0]));
    if (status != QIO_SUCCESS) { 
      QDPIO::cerr << "Failed to read data" << endl;
      clear(QDPIO_badbit);
      QDP_abort(1);
    }
    QDPIO::cout << "QIO_read_finished" << endl;
      
    // Cast appropriately
//    for(int i=0; i < from_disk.size(); i++) { 
    s1.open(from_disk);
//    }
  
    istringstream ss;
    if (Layout::primaryNode()) {
      string foo = QIO_string_ptr(xml_c);
      ss.str(foo);
    }
    rec_xml.open(ss);
  
    QIO_string_destroy(xml_c);
  }


  //-----------------------------------------------------------------------------
  // QDP QIO support (writers)
  QDPFileWriter::QDPFileWriter() {iop=false;}

  QDPFileWriter::QDPFileWriter(XMLBufferWriter& xml, 
			       const std::string& path,
			       QDP_volfmt_t qdp_volfmt,
			       QDP_serialparallel_t qdp_serpar,
			       QDP_filemode_t qdp_mode) 
  {
    open(xml,path,qdp_volfmt,qdp_serpar,qdp_mode, std::string());
  }

  QDPFileWriter::QDPFileWriter(XMLBufferWriter& xml, 
			       const std::string& path,
			       QDP_volfmt_t qdp_volfmt,
			       QDP_serialparallel_t qdp_serpar,
			       QDP_filemode_t qdp_mode,
			       const std::string& data_LFN) 
  {
    open(xml,path,qdp_volfmt,qdp_serpar,qdp_mode, data_LFN);
  }

  // filemode not specified
  void QDPFileWriter::open(XMLBufferWriter& file_xml, 
			   const std::string& path,
			   QDP_volfmt_t qdp_volfmt,
			   QDP_serialparallel_t qdp_serpar)
  {
    open(file_xml,path,qdp_volfmt,qdp_serpar,QDPIO_OPEN, std::string());
  }

  void QDPFileWriter::open(XMLBufferWriter& file_xml, 
			   const std::string& path,
			   QDP_volfmt_t qdp_volfmt,
			   QDP_serialparallel_t qdp_serpar,
			   const std::string& data_LFN) 
  {
    open(file_xml,path,qdp_volfmt,qdp_serpar,QDPIO_OPEN, data_LFN);
  }


  // filemode not specified
  QDPFileWriter::QDPFileWriter(XMLBufferWriter& xml, 
			       const std::string& path,
			       QDP_volfmt_t qdp_volfmt,
			       QDP_serialparallel_t qdp_serpar) 
  {
    open(xml,path,qdp_volfmt,qdp_serpar,QDPIO_OPEN, std::string());
  }

  // filemode not specified
  QDPFileWriter::QDPFileWriter(XMLBufferWriter& xml, 
			       const std::string& path,
			       QDP_volfmt_t qdp_volfmt,
			       QDP_serialparallel_t qdp_serpar,
			       const std::string& data_LFN) 
  {
    open(xml,path,qdp_volfmt,qdp_serpar,QDPIO_OPEN, data_LFN);
  }

  void QDPFileWriter::open(XMLBufferWriter& file_xml, 
			   const std::string& path,
			   QDP_volfmt_t qdp_volfmt,
			   QDP_serialparallel_t qdp_serpar,
			   QDP_filemode_t qdp_mode, 
			   const std::string& data_LFN) 
  {

    QIO_Layout layout;
    int latsize[Nd];

    for(int m=0; m < Nd; ++m)
      latsize[m] = Layout::lattSize()[m];

    layout.node_number = &get_node_number;
    layout.node_index  = &get_node_index;
    layout.get_coords  = &get_coords;
    layout.num_sites = &get_sites_on_node;
    layout.latsize = latsize;
    layout.latdim = Nd; 
    layout.volume = Layout::vol(); 
    layout.sites_on_node = Layout::sitesOnNode(); 
    layout.this_node = Layout::nodeNumber(); 
    layout.number_of_nodes = Layout::numNodes(); 

    // Copy metadata string into simple qio string container
    QIO_String* xml_c = QIO_string_create();
    QIO_string_set(xml_c, file_xml.str().c_str());

    if (xml_c == NULL)
    {
      QDPIO::cerr << "QDPFileWriter - error in creating QIO string" << endl;
      iostate = QDPIO_badbit;
    }
    else
    {
      iostate = QDPIO_goodbit;
    }

    // Wrappers over simple ints
    int volfmt;
    switch(qdp_volfmt)
    {
    case QDPIO_SINGLEFILE:
      volfmt = QIO_SINGLEFILE;
      //    ionodefunc = &(SingleFileIONode::IONode);
      //    master_io_nodefunc = &(SingleFileIONode::masterIONode);
      break;
    
    case QDPIO_MULTIFILE:
      volfmt = QIO_MULTIFILE;
      // ionodefunc = &(MultiFileIONode::IONode);
      // master_io_nodefunc = &(MultiFileIONode::masterIONode);

      break;

    case QDPIO_PARTFILE:
      volfmt = QIO_PARTFILE;
      //ionodefunc = &(PartFileIONode::IONode);
      // master_io_nodefunc = &(PartFileIONode::masterIONode);
    
      break;

    default: 
      QDPIO::cerr << "Unknown value for qdp_volfmt " << qdp_volfmt << endl;
      QDP_abort(1);
      return;
    }
  
    // Wrappers over simple ints
    int mode;
    switch(qdp_mode)
    {
    case QDPIO_CREATE:
      mode = QIO_CREAT;
      break;
    
    case QDPIO_OPEN:
      mode = QIO_TRUNC;
      break;

    case QDPIO_APPEND:
      mode = QIO_APPEND;
      break;

    default: 
      QDPIO::cerr << "Unknown value for qdp_mode " << qdp_mode << endl;
      QDP_abort(1);
      return;
    }
  
    // QIO write
    // For now, serpar (which is an enum in QDP) is ignored here
    QIO_Oflag oflag;
    oflag.serpar = QIO_SERIAL;
    oflag.mode   = mode;
    oflag.ildgstyle = QIO_ILDGLAT;
    if( data_LFN.length() == 0 ) { 
      oflag.ildgLFN = NULL;
    }
    else {
      oflag.ildgLFN = QIO_string_create();
      QIO_string_set(oflag.ildgLFN, data_LFN.c_str());
    }

#if 1
    // This is the QIO Way - older way 
    if ((qio_out = QIO_open_write(xml_c, path.c_str(), 
				  volfmt, 
				  &layout, 
				  NULL, &oflag)) == NULL )
    {
      iostate = QDPIO_badbit;  // not helpful

      QDPIO::cerr << "QDPFileWriter: failed to open file " << path << endl;
      QDP_abort(1);  // just bail. Not sure I want this. This is not stream semantics
    }
    else
    {
      iostate = QDPIO_goodbit;
    }

#else 
    /*! This was an attempt to control the choking to I/O nodes
     *  I have deactivated it because I am not sure how to choke
     *  the readers to do the same thing
     */
    if ((qio_out = QIO_generic_open_write(path.c_str(), 
					  volfmt, 
					  &layout, 
					  &oflag,
					  ionodefunc,
					  master_io_nodefunc)) == NULL)
    {
      iostate = QDPIO_badbit;  // not helpful

      QDPIO::cerr << "QDPFileWriter: failed to open file " << path << endl;
      QDP_abort(1);  // just bail. Not sure I want this. This is not stream semantics
    }
    else
    {
      iostate = QDPIO_goodbit;
    }

    DML_sync();

    int status = QIO_write_file_header(qio_out, xml_c);
    if( status != QIO_SUCCESS ) { 
      iostate = QDPIO_badbit;
      QDPIO::cerr << "QDPFileWriter: failed to write File XML " << endl;
      QDP_abort(1);  // just bail. Not sure I want this. This is not stream semantics
    }
    else {
      iostate = QDPIO_goodbit;
    }
#endif 

    // Free memory -- this is OK< as it should'of been copied
    QIO_string_destroy(oflag.ildgLFN);
    // Cleanup
    QIO_string_destroy(xml_c);

    iop=true;
  }

  void QDPFileWriter::close()
  {
    if (is_open()) 
    {
      // int status = QIO_close_write(qio_out);
      QIO_close_write(qio_out);
    }

    iop = false;
    iostate = QDPIO_badbit;
  }

  bool QDPFileWriter::is_open() {return iop;}

  bool QDPFileWriter::bad() const {return iostate;}

  void QDPFileWriter::clear(QDP_iostate_t state)
  {
    iostate = state;
  }

  QDPFileWriter::~QDPFileWriter() {close();}


  void close(QDPFileWriter& qsw)
  {
    qsw.close();
  }


  bool is_open(QDPFileWriter& qsw)
  {
    return qsw.is_open();
  }


  //! Writes a BinaryBufferWriter object
  /*!
    \param rec_xml The (user) record metadata.
    \param sl The data
  */
  void QDPFileWriter::write(XMLBufferWriter& rec_xml, BinaryBufferWriter& s1)
  {
    std::string ss = s1.str();
    QIO_RecordInfo* info = QIO_create_record_info(QIO_GLOBAL, NULL, NULL, 0,
						  "char",
						  "U",
						  0, 0, 
						  sizeof(char), ss.size());


    // Copy metadata string into simple qio string container
    QIO_String* xml_c = QIO_string_create();
    if (xml_c == NULL)
    {
      QDPIO::cerr << "QDPFileWriter::write - error in creating XML string" << endl;
      QDP_abort(1);
    }

    if (Layout::primaryNode())
      QIO_string_set(xml_c, rec_xml.str().c_str());

    // Big call to qio
    if (QIO_write(get(), info, xml_c,
		  &(QDPOScalarFactoryGet<char>),
		  ss.size()*sizeof(char), 
		  sizeof(char), 
		  (void *)ss.c_str()) != QIO_SUCCESS)
    {
      QDPIO::cerr << "QDPFileWriter: error in write" << endl;
      clear(QDPIO_badbit);
      QDP_abort(1);
    }

    // Cleanup
    QIO_string_destroy(xml_c);
    QIO_destroy_record_info(info);
  }


} // namespace QDP;
