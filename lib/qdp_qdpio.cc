// $Id: qdp_qdpio.cc,v 1.23 2005-12-01 17:48:32 bjoo Exp $
//
/*! @file
 * @brief IO support via QIO
 */

#include "qdp.h"

QDP_BEGIN_NAMESPACE(QDP);

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
    multi1d<int> my_coords = Layout::getLogicalCoordFrom(node);
    multi1d<int> io_node_coords(my_coords.size());
    for(int i=0; i < my_coords.size(); i++) { 
      io_node_coords[i] = 2*(my_coords[i]/2);
    }
    return Layout::getNodeNumberFrom(io_node_coords);
  }
  int masterIONode(void) { 
    return DML_master_io_node();
  }
}


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
  if ((qio_in = QIO_open_read(xml_c, path.c_str(), &layout, NULL)) == NULL)
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

  // Yucky C style callback functions
  int (*ionodefunc)(int);
  int (*master_io_nodefunc)(void);

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
    ionodefunc = &(SingleFileIONode::IONode);
    master_io_nodefunc = &(SingleFileIONode::masterIONode);
    break;
    
  case QDPIO_MULTIFILE:
    volfmt = QIO_MULTIFILE;
    ionodefunc = &(MultiFileIONode::IONode);
    master_io_nodefunc = &(MultiFileIONode::masterIONode);

    break;

  case QDPIO_PARTFILE:
    ionodefunc = &(PartFileIONode::IONode);
    master_io_nodefunc = &(PartFileIONode::masterIONode);
    volfmt = QIO_PARTFILE;
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
					&oflag)) == NULL )
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

QDP_END_NAMESPACE();
