// $Id: qdp_qdpio.cc,v 1.5 2003-12-06 23:05:26 edwards Exp $
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


//-----------------------------------------
// QDP QIO support
QDPSerialFileReader::QDPSerialFileReader() {iop=false;}

QDPSerialFileReader::QDPSerialFileReader(XMLReader& xml, const std::string& p) {open(xml,p);}

void QDPSerialFileReader::open(XMLReader& file_xml, const std::string& path) 
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

  if ((qio_in = QIO_open_read(xml_c, path.c_str(), QIO_SERIAL, &layout)) == NULL)
  {
    QDPIO::cerr << "QDPSerialFile::Reader: failed to open file " << path << endl;
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

void QDPSerialFileReader::close()
{
  if (is_open()) 
  {
    int status = QIO_close_read(qio_in);
  }

  iop = false;
}

bool QDPSerialFileReader::is_open() {return iop;}

bool QDPSerialFileReader::eof() const {return false;}

bool QDPSerialFileReader::bad() const {return false;}

QDPSerialFileReader::~QDPSerialFileReader() {close();}


//-----------------------------------------
//! text writer support
QDPSerialFileWriter::QDPSerialFileWriter() {iop=false;}

QDPSerialFileWriter::QDPSerialFileWriter(XMLBufferWriter& xml, const std::string& p) 
{
  open(xml,p);
}

void QDPSerialFileWriter::open(XMLBufferWriter& file_xml, const std::string& path) 
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
//  XMLBufferWriter& foo_xml = const_cast<XMLBufferWriter&>(file_xml);
  XML_String* xml_c = XML_string_create(file_xml.str().length()+1);  // check if +1 is needed
  XML_string_set(xml_c, file_xml.str().c_str());

  // Big call to qio
  if ((qio_out = QIO_open_write(xml_c, path.c_str(), 
				QIO_SERIAL, QIO_LEX_ORDER, QIO_CREATE, 
				&layout)) == NULL)
  {
    QDPIO::cerr << "QDPSerialFile::Writer: failed to open file " << path << endl;
    QDP_abort(1);
  }

  // Cleanup
  XML_string_destroy(xml_c);

  iop=true;
}

void QDPSerialFileWriter::close()
{
  if (is_open()) 
  {
    int status = QIO_close_write(qio_out);
  }

  iop = false;
}

bool QDPSerialFileWriter::is_open() {return iop;}

bool QDPSerialFileWriter::bad() const {return false;}

QDPSerialFileWriter::~QDPSerialFileWriter() {close();}


QDP_END_NAMESPACE();
