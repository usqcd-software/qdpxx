// $Id: qdp_qdpio.cc,v 1.6 2004-02-02 04:59:54 edwards Exp $
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


//-----------------------------------------------------------------------------
// QDP QIO support
QDPFileReader::QDPFileReader() {iop=false;}

QDPFileReader::QDPFileReader(XMLReader& xml, 
			     const std::string& path,
			     QDP_serialparallel_t qdp_serpar)
  {open(xml,path,qdp_serpar);}

void QDPFileReader::open(XMLReader& file_xml, 
			 const std::string& path, 
			 QDP_serialparallel_t qdp_serpar)
{
  QIO_Layout layout;
  int latsize[Nd];

  for(int m=0; m < Nd; ++m)
    latsize[m] = Layout::lattSize()[m];

  layout.node_number = &get_node_number;
  layout.node_index  = &get_node_index;
  layout.get_coords  = &get_coords;
  layout.latsize = latsize;
  layout.latdim = Nd; 
  layout.volume = Layout::vol(); 
  layout.sites_on_node = Layout::sitesOnNode(); 
  layout.this_node = Layout::nodeNumber(); 
  layout.number_of_nodes = Layout::numNodes(); 

  // Initialize string objects 
  XML_String *xml_c  = XML_string_create(0);

  // Wrappers over simple ints
  int serpar;
  switch(qdp_serpar)
  {
  case QDPIO_SERIAL:
    serpar = QIO_SERIAL;
    break;
    
  case QDPIO_PARALLEL:
    serpar = QIO_PARALLEL;
    break;

  default:
    QDPIO::cerr << "QDPFileReader: invalid serial mode" << endl;
  }

  // Call QIO read
  if ((qio_in = QIO_open_read(xml_c, path.c_str(), serpar, &layout)) == NULL)
  {
    QDPIO::cerr << "QDPFileReader: failed to open file " << path << endl;
    QDP_abort(1);
  }

  // Use string to initialize XMLReader
  istringstream ss;
  if (Layout::primaryNode())
    ss.str((const string)(XML_string_ptr(xml_c)));
  file_xml.open(ss);

  XML_string_destroy(xml_c);

  iop=true;
}

void QDPFileReader::close()
{
  if (is_open()) 
  {
    int status = QIO_close_read(qio_in);
  }

  iop = false;
}

bool QDPFileReader::is_open() {return iop;}

bool QDPFileReader::eof() const {return false;}

bool QDPFileReader::bad() const {return false;}

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
  open(xml,path,qdp_volfmt,qdp_serpar,qdp_mode);
}

void QDPFileWriter::open(XMLBufferWriter& file_xml, 
			 const std::string& path,
			 QDP_volfmt_t qdp_volfmt,
			 QDP_serialparallel_t qdp_serpar,
			 QDP_filemode_t qdp_mode) 
{
  QIO_Layout layout;
  int latsize[Nd];

  for(int m=0; m < Nd; ++m)
    latsize[m] = Layout::lattSize()[m];

  layout.node_number = &get_node_number;
  layout.node_index  = &get_node_index;
  layout.get_coords  = &get_coords;
  layout.latsize = latsize;
  layout.latdim = Nd; 
  layout.volume = Layout::vol(); 
  layout.sites_on_node = Layout::sitesOnNode(); 
  layout.this_node = Layout::nodeNumber(); 
  layout.number_of_nodes = Layout::numNodes(); 

  // Copy metadata string into simple qio string container
  XML_String* xml_c = XML_string_create(file_xml.str().length()+1);  // check if +1 is needed
  XML_string_set(xml_c, file_xml.str().c_str());

  // Wrappers over simple ints
  int serpar;
  switch(qdp_serpar)
  {
  case QDPIO_SERIAL:
    serpar = QIO_SERIAL;
    break;
    
  case QDPIO_PARALLEL:
    serpar = QIO_PARALLEL;
    break;
  }

  // Wrappers over simple ints
  int volfmt;
  switch(qdp_volfmt)
  {
  case QDPIO_SINGLEFILE:
    volfmt = QIO_SINGLEFILE;
    break;
    
  case QDPIO_MULTIFILE:
    volfmt = QIO_MULTIFILE;
    break;
  }

  // Wrappers over simple ints
  int mode;
  switch(qdp_mode)
  {
  case QDPIO_CREATE:
    mode = QIO_CREATE;
    break;
    
  case QDPIO_OPEN:
    mode = QIO_TRUNCATE;
    break;
    
  case QDPIO_APPEND:
    mode = QIO_APPEND;
    break;
  }

  // QIO write
  if ((qio_out = QIO_open_write(xml_c, path.c_str(), 
				serpar, volfmt, mode,
				&layout)) == NULL)
  {
    QDPIO::cerr << "QDPFileWriter: failed to open file " << path << endl;
    QDP_abort(1);
  }

  // Cleanup
  XML_string_destroy(xml_c);

  iop=true;
}

void QDPFileWriter::close()
{
  if (is_open()) 
  {
    int status = QIO_close_write(qio_out);
  }

  iop = false;
}

bool QDPFileWriter::is_open() {return iop;}

bool QDPFileWriter::bad() const {return false;}

QDPFileWriter::~QDPFileWriter() {close();}

//! Close a QDPFileWriter
void close(QDPFileWriter& qsw)
{
  qsw.close();
}

//! Is a QDPFileWriter open
bool is_open(QDPFileWriter& qsw)
{
  return qsw.is_open();
}

QDP_END_NAMESPACE();
