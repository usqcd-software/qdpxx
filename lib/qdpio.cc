// $Id: qdpio.cc,v 1.6 2003-04-30 21:09:48 edwards Exp $
//
/*! @file
 * @brief IO support via QIO
 */

#include "qdp.h"

QDP_BEGIN_NAMESPACE(QDP);

//--------------------------------------------------------------------------------
// XML classes
//! XML reader class
XMLMetaReader::XMLMetaReader()
{
  // HACK - set string to something trivial
  xml = XML_create(MAX_XML);
}

XMLMetaReader::~XMLMetaReader()
{
  XML_destroy(xml);
}

//! XML writer class
XMLMetaWriter::XMLMetaWriter()
{
  // HACK - set string to something trivial
  xml = XML_create(MAX_XML);
}

XMLMetaWriter::~XMLMetaWriter()
{
  XML_destroy(xml);
}



//-----------------------------------------
static int get_node_number(const int coord[])
{
  multi1d<int> crd(Nd);
  crd = coord;   // an array copy
  int node = Layout::nodeNumber(crd);
  return node;
}


//-----------------------------------------
// QDP QIO support
QDPSerialReader::QDPSerialReader() {iop=false;}

QDPSerialReader::QDPSerialReader(XMLMetaReader& xml, const std::string& p) {open(xml,p);}

void QDPSerialReader::open(XMLMetaReader& xml, const std::string& p) 
{
  QIO_Layout *layout = new QIO_Layout;
  int latsize[Nd];

  for(int m=0; m < Nd; ++m)
    latsize[m] = Layout::lattSize()[m];

  layout->node_number = &get_node_number;
  layout->latsize = latsize; // local copy
  layout->latdim = Nd; 
  layout->volume = Layout::vol(); 
  layout->this_node = Layout::nodeNumber(); 

  if ((qio_in = QIO_open_read(xml.get(), p.c_str(), QIO_SERIAL, layout)) == NULL)
    QDP_error_exit("QDPSerial::Reader: failed to open file %s",p);

  delete layout;

  iop=true;
}

void QDPSerialReader::close()
{
  if (is_open()) 
  {
    int status = QIO_close_read(qio_in);
  }

  iop = false;
}

bool QDPSerialReader::is_open() {return iop;}

bool QDPSerialReader::eof() const {return false;}

bool QDPSerialReader::bad() const {return false;}

QDPSerialReader::~QDPSerialReader() {close();}


//-----------------------------------------
//! text writer support
QDPSerialWriter::QDPSerialWriter() {iop=false;}

QDPSerialWriter::QDPSerialWriter(const XMLMetaWriter& xml, const std::string& p) {open(xml,p);}

void QDPSerialWriter::open(const XMLMetaWriter& xml, const std::string& p) 
{
  QIO_Layout *layout = new QIO_Layout;
  int latsize[Nd];

  for(int m=0; m < Nd; ++m)
    latsize[m] = Layout::lattSize()[m];

  layout->node_number = &get_node_number;
  layout->latsize = latsize; // local copy
  layout->latdim = Nd; 
  layout->volume = Layout::vol(); 
  layout->this_node = Layout::nodeNumber(); 

  if ((qio_out = QIO_open_write(xml.get(), p.c_str(), QIO_SERIAL, QIO_LEX_ORDER, QIO_CREATE, 
				layout)) == NULL)
    QDP_error_exit("QDPSerial::Writer: failed to open file %s",p);

  delete layout;

  iop=true;
}

void QDPSerialWriter::close()
{
  if (is_open()) 
  {
    int status = QIO_close_write(qio_out);
  }

  iop = false;
}

bool QDPSerialWriter::is_open() {return iop;}

bool QDPSerialWriter::bad() const {return false;}

QDPSerialWriter::~QDPSerialWriter() {close();}


QDP_END_NAMESPACE();
