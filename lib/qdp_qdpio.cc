// $Id: qdp_qdpio.cc,v 1.14 2004-09-10 21:23:41 edwards Exp $
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
			     int iflag)
  {open(xml,path,iflag);}

void QDPFileReader::open(XMLReader& file_xml, 
			 const std::string& path, 
			 int iflag)
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
  QIO_String *xml_c  = QIO_string_create();

  // Call QIO read
  if ((qio_in = QIO_open_read(xml_c, path.c_str(), &layout, iflag)) == NULL)
  {
    QDPIO::cerr << "QDPFileReader: failed to open file " << path << endl;
    iostate = QDPIO_badbit;
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

// OBSOLETE
QDPFileReader::QDPFileReader(XMLReader& xml, 
			     const std::string& path,
			     QDP_serialparallel_t qdp_serpar)
  {open(xml,path,0);}

// OBSOLETE
void QDPFileReader::open(XMLReader& file_xml, 
			 const std::string& path, 
			 QDP_serialparallel_t qdp_serpar)
  {open(file_xml,path,0);}



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
			     int oflag) 
{
  open(xml,path,qdp_volfmt,oflag);
}

void QDPFileWriter::open(XMLBufferWriter& file_xml, 
			 const std::string& path,
			 QDP_volfmt_t qdp_volfmt,
			 int oflag) 
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
    break;
    
  case QDPIO_MULTIFILE:
    volfmt = QIO_MULTIFILE;
    break;

  case QDPIO_PARTFILE:
    volfmt = QIO_PARTFILE;
    break;

  default: 
    QDPIO::cerr << "Unknown value for qdp_volfmt " << qdp_volfmt << endl;
    QDP_abort(1);
    return;
  }
  
  // QIO write
  if ((qio_out = QIO_open_write(xml_c, path.c_str(), 
				volfmt, &layout, oflag)) == NULL)
  {
    QDPIO::cerr << "QDPFileWriter: failed to open file " << path << endl;
    iostate = QDPIO_badbit;
  }
  else
  {
    iostate = QDPIO_goodbit;
  }

  // Cleanup
  QIO_string_destroy(xml_c);

  iop=true;
}

// OBSOLETE
QDPFileWriter::QDPFileWriter(XMLBufferWriter& xml, 
			     const std::string& path,
			     QDP_volfmt_t qdp_volfmt,
			     QDP_serialparallel_t qdp_serpar,
			     QDP_filemode_t qdp_mode) 
{
  open(xml,path,qdp_volfmt,0);
}

// OBSOLETE
void QDPFileWriter::open(XMLBufferWriter& file_xml, 
			 const std::string& path,
			 QDP_volfmt_t qdp_volfmt,
			 QDP_serialparallel_t qdp_serpar,
			 QDP_filemode_t qdp_mode) 
{
  open(file_xml,path,qdp_volfmt,0);
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
