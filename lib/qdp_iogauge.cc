// $Id: qdp_iogauge.cc,v 1.18 2005-03-18 13:56:23 zbigniew Exp $
//
// QDP data parallel interface
/*!
 * @file
 * @brief  Various gauge readers/writers and propagator readers/writers.
 */

#include "qdp.h"
#include "qdp_iogauge.h"

#include "time.h"

#include <string>
using std::string;

QDP_BEGIN_NAMESPACE(QDP);


//! Write a multi1d array
template<class T>
ostream& operator<<(ostream& s, const multi1d<T>& d)
{
  s << d[0];
  for(int i=1; i < d.size(); ++i)
    s << " " << d[i];

  return s;
}


void archivGaugeInit(ArchivGauge_t& header)
{
  header.mat_size = 12;
  header.float_size = 4; // 32 bits
  header.nrow = Layout::lattSize();
  header.boundary.resize(Nd);
  header.boundary = 1;   // periodic
  header.sequence_number = 0;
  header.ensemble_label = "NERSC archive";
  header.creator = "QDP++";
  header.creator_hardware = "QDP++";

  time_t now = time(NULL);
  {
    char *tmp = ctime(&now);
    int date_size = strlen(tmp);
    char *datetime = new(nothrow) char[date_size+1];
    if( datetime == 0x0 ) { 
      QDP_error_exit("Unable to allocate datetime in qdp_iogauge.cc\n");
    }

    strcpy(datetime,ctime(&now));

    for(int i=0; i < date_size; ++i)
      if ( datetime[i] == '\n' )
      {
	datetime[i] = '\0';
	date_size = i;
	break;
     }   

    header.creation_date = datetime;
    delete[] datetime;
  }
  header.archive_date  = header.creation_date;

  {
    ostringstream s;
    s << "X" << now;
    header.ensemble_id = s.str();
  }

  header.w_plaq = 0;   // WARNING: bogus
  header.link = 0;     // WARNING: bogus
}


//! Source header read
void read(XMLReader& xml, const string& path, ArchivGauge_t& header)
{
  XMLReader paramtop(xml, path);

  read(paramtop, "mat_size", header.mat_size);
  read(paramtop, "float_size", header.float_size);
  read(paramtop, "nrow", header.nrow);
  read(paramtop, "boundary", header.boundary);
  read(paramtop, "ensemble_id", header.ensemble_id);
  read(paramtop, "ensemble_label", header.ensemble_label);
  read(paramtop, "creator", header.creator);
  read(paramtop, "creator_hardware", header.creator_hardware);
  read(paramtop, "creation_date", header.creation_date);
  read(paramtop, "archive_date", header.archive_date);
}


//! Source header writer
void write(XMLWriter& xml, const string& path, const ArchivGauge_t& header)
{
  push(xml, path);

  write(xml, "mat_size", header.mat_size);
  write(xml, "float_size", header.float_size);
  write(xml, "nrow", header.nrow);
  write(xml, "boundary", header.boundary);
  write(xml, "ensemble_id", header.ensemble_id);
  write(xml, "ensemble_label", header.ensemble_label);
  write(xml, "creator", header.creator);
  write(xml, "creator_hardware", header.creator_hardware);
  write(xml, "creation_date", header.creation_date);
  write(xml, "archive_date", header.archive_date);

  pop(xml);
}



//-----------------------------------------------------------------------
// Read a QCD archive file header
//! Read a QCD (NERSC) Archive format gauge field header
/*!
 * \ingroup io
 *
 * \param header     structure holding config info ( Modify )
 * \param cfg_in     binary writer object ( Modify )

 \note This can handle three-row format link matrices if the
 \c DATATYPE key has the value \c 4D_SU3_GAUGE_3x3
 or two-row format matrices if it has the value \c 4D_SU3_GAUGE

 The plaquette, link and checksum values are ignored.
*/    

static void readArchivHeader(BinaryReader& cfg_in, ArchivGauge_t& header)
{
  if (Nd != 4)
    QDP_error_exit("Expecting Nd == 4");


  archivGaugeInit(header);

  const size_t max_line_length = 128;

  // The expected lattice size of the gauge field
  header.nrow.resize(Nd);

  /* For now, read and throw away the header */
  string line;

  QDPIO::cout << "Start of header" << endl;

  cfg_in.read(line, max_line_length);
  QDPIO::cout << line << endl;
  
  if (line != string("BEGIN_HEADER"))
    QDP_error_exit("Missing BEGIN_HEADER");

  /* assume matrix size is 2*Nc*Nc (matrix is UNcompressed) 
     and change if we find out otherwise */
  header.mat_size=2*Nc*Nc ;

  /* Begin loop on lines */
  int  lat_size_cnt = 0;

  while (1)
  {
    cfg_in.read(line, max_line_length);
    QDPIO::cout << line << endl;

    char linetype[max_line_length];
    int itmp, dd;

    // Scan for the datatype then scan for it
    if ( sscanf(line.c_str(), "DATATYPE = %s", linetype) == 1 ) 
    {
      /* Check if it is uncompressed */
      if (strcmp(linetype, "4D_SU3_GAUGE_3x3") == 0) 
      {
	header.mat_size=18;   /* Uncompressed matrix */
	if (Nc != 3)
	  QDP_error_exit("Expecting Nc == 3");
      }
      else if (strcmp(linetype, "4D_SU3_GAUGE") == 0) 
      {
	header.mat_size=12;   /* Compressed matrix */
	if (Nc != 3)
	  QDP_error_exit("Expecting Nc == 3");
      }
      else if (strcmp(linetype, "4D_SU4_GAUGE") == 0) 
      {
	if (Nc != 4)
	  QDP_error_exit("Expecting Nc == 4");
      }
    }

    // Scan for the sequence number
    if ( sscanf(line.c_str(), "SEQUENCE_NUMBER = %d", &itmp) == 1 ) 
    {
      header.sequence_number = itmp;
    }

    // Scan for the ensemble label
    if ( sscanf(line.c_str(), "ENSEMBLE_LABEL = %s", linetype) == 1 ) 
    {
      header.ensemble_label = linetype;
    }

    // Scan for the creator
    if ( sscanf(line.c_str(), "CREATOR = %s", linetype) == 1 ) 
    {
      header.creator = linetype;
    }

    // Scan for the creator
    if ( sscanf(line.c_str(), "CREATOR_HARDWARE = %s", linetype) == 1 ) 
    {
      header.creator_hardware = linetype;
    }

    // Scan for the creation date
    if ( sscanf(line.c_str(), "CREATION_DATE = %s", linetype) == 1 ) 
    {
      header.creation_date = linetype;
    }

    // Scan for the archive date
    if ( sscanf(line.c_str(), "ARCHIVE_DATE = %s", linetype) == 1 ) 
    {
      header.archive_date = linetype;
    }

    // Find the lattice size of the gauge field
    if ( sscanf(line.c_str(), "DIMENSION_%d = %d", &dd, &itmp) == 2 ) 
    {
      /* Found a lat size */
      if (dd < 1 || dd > Nd)
	QDP_error_exit("oops, dimension number out of bounds");

      header.nrow[dd-1] = itmp;
      ++lat_size_cnt;
      }
    
    // Find the boundary conditions
    if ( sscanf(line.c_str(), "BOUNDARY_%d = %d", &dd, &itmp) == 2 ) 
    {
      /* Found a lat size */
      if (dd < 1 || dd > Nd)
	QDP_error_exit("oops, dimension number out of bounds");
      
      header.boundary[dd-1] = itmp;
    }

    char fpstring[12];
    if( sscanf(line.c_str(), "FLOATING_POINT = %s", fpstring) == 1 ) {
      if( (strcmp(fpstring, "IEEE32BIG") == 0)
         || (strcmp(fpstring, "IEEE32") == 0)  ) { 
	QDPIO::cout << "Floating type: IEEE32BIG" << endl; 
	header.float_size=4;
      }
      else if ( strcmp(fpstring, "IEEE64BIG") == 0 ) { 
	header.float_size=8;
      }
      else { 
	QDP_error_exit("oops unknown floating point type\n");
      }
    }
 
    if (line == string("END_HEADER")) break;
  }

  QDPIO::cout << "End of header" << endl;

  // Sanity check
  if (lat_size_cnt != Nd)
    QDP_error_exit("did not find all the lattice sizes");

  for(int dd=0; dd < Nd; ++dd)
    if (header.nrow[dd] != Layout::lattSize()[dd])
      QDP_error_exit("readArchiv: archive lattice size does not agree with current size");

}


//-----------------------------------------------------------------------
//! Read a NERSC Gauge Connection  Archive file
// See the corresponding  qdp_*_specific.cc files
//! Writes a NERSC Gauge Connection Archive gauge configuration file
/*!
 * \ingroup io
 An architecture-specific version of this routine is called by the generic
 readArchiv functions.

 The data is written in big-endian IEEE format to the file.
 If the host nachine is little-endian, the data is byte-swapped.

  \param cfg_in    A binary reader
  \param u          The gauge configuration 
  \param mat_size   The number of floating-point numbers per link matrix in
  the file. This should be 12 to write two-row format or 18 for three-row
  format. 
  \param float_size
  
  \pre The binary writer should have already opened the file, and should be
  pointing to the beginning of the binary data.
*/
void readArchiv(BinaryReader& cfg_in, multi1d<LatticeColorMatrix>& u, int mat_size, int float_size);



// Read a QCD (NERSC) Archive format gauge field
/*
 * \ingroup io
 *
 * \param header     structure holding config info ( Modify )
 * \param u          gauge configuration ( Modify )
 * \param file       path ( Read )
 */    
void readArchiv(ArchivGauge_t& header, multi1d<LatticeColorMatrix>& u, const string& file)
{
  BinaryReader cfg_in(file);

  readArchivHeader(cfg_in, header);   // read header
  readArchiv(cfg_in, u, header.mat_size, header.float_size);  // expects to be positioned at the beginning of the binary payload

  cfg_in.close();
}


//-----------------------------------------------------------------------
// Read a Archive configuration file
/*
 * \ingroup io
 *
 * \param xml        xml reader holding config info ( Modify )
 * \param u          gauge configuration ( Modify )
 * \param cfg_file   path ( Read )
 */    

void readArchiv(XMLReader& xml, multi1d<LatticeColorMatrix>& u, const string& cfg_file)
{
  ArchivGauge_t header;

  // Read the config and its binary header
  readArchiv(header, u, cfg_file);

  // Now, set up the XML header. Do this by first making a buffer
  // writer that is then used to make the reader
  XMLBufferWriter  xml_buf;
  write(xml_buf, "NERSC", header);

  try 
  {
    xml.open(xml_buf);
  }
  catch(const string& e)
  { 
    QDP_error_exit("Error in readArchiv: %s",e.c_str());
  }
}



//-----------------------------------------------------------------------
// Read a QCD (NERSC) Archive format gauge field
/*
 * \ingroup io
 *
 * \param u          gauge configuration ( Modify )
 * \param cfg_file   path ( Read )
 */    
void readArchiv(multi1d<LatticeColorMatrix>& u, const string& cfg_file)
{
  ArchivGauge_t header;
  readArchiv(header, u, cfg_file); // throw away the header
}




//-----------------------------------------------------------------------
// Write a QCD archive file
//! Write a QCD (NERSC) Archive format gauge field
/*!
 * \ingroup io
 *
 * \param header     structure holding config info ( Modify )
 * \param cfg_out     binary writer object ( Modify )

 \pre The information in the header should be filled in.
 
 \note The value 0 is written as checksum.
 \note The token \c FLOATING_POINT is always given the value \c IEEE32BIG
 */    
static void writeArchivHeader(BinaryWriter& cfg_out, const ArchivGauge_t& header)
{
  if (Nd != 4)
  {
    QDPIO::cerr << "Expecting Nd == 4" << endl;
    QDP_abort(1);
  }

  if (Nc != 3)
  {
    QDPIO::cerr << "Expecting Nc == 3" << endl;
    QDP_abort(1);
  }

  ostringstream head;

  head << "BEGIN_HEADER\n";

  head << "CHECKSUM = 0\n";     // WARNING BOGUS!!!
  head << "LINK_TRACE = " << header.link << "\n";
  head << "PLAQUETTE = " << header.w_plaq << "\n";

  head << "DATATYPE = 4D_SU3_GAUGE\n"
       << "HDR_VERSION = 1.0\n"
       << "STORAGE_FORMAT = 1.0\n";

  for(int i=1; i <= Nd; ++i)
    head << "DIMENSION_" << i << " = " << Layout::lattSize()[i-1] << "\n";

  for(int i=0; i < Nd; ++i)
    if (header.boundary[i] == 1)
      head << "BOUNDARY_" << (i+1) << " = PERIODIC\n";
    else if (header.boundary[i] == -1)
      head << "BOUNDARY_" << (i+1) << " = ANTIPERIODIC\n";
    else
    {
      QDPIO::cerr << "writeArchiv: unknown boundary type";
      QDP_abort(1);
    }

  head << "ENSEMBLE_ID = " << header.ensemble_id << "\n"
       << "ENSEMBLE_LABEL = " << header.ensemble_label << "\n"
       << "SEQUENCE_NUMBER = " << header.sequence_number << "\n"
       << "CREATOR = " << header.creator << "\n"
       << "CREATOR_HARDWARE = " << header.creator_hardware << "\n"
       << "CREATION_DATE = " << header.creation_date << "\n"
       << "ARCHIVE_DATE = " << header.archive_date << "\n"
       << "FLOATING_POINT = IEEE32BIG\n";

  head << "END_HEADER\n";

  cfg_out.writeArray(head.str().c_str(), 1, head.str().size());
}


// Write a QCD archive file
// See the corresponding  qdp_*_specific.cc files

//! Writes a NERSC Gauge Connection Archive gauge configuration file
/*!
 * \ingroup io
 An architecture-specific version of this routine is called by the generic
 readArchiv functions.

 The data is written in big-endian IEEE format to the file.
 If the host nachine is little-endian, the data is byte-swapped.

  \param cfg_out    A binary writer
  \param u          The gauge configuration 
  \param mat_size   The number of floating-point numbers per link matrix to
  write. This should be 12 to write two-row format or 18 for three-row format.

  \pre The binary writer should have already opened the file.
*/
  
void writeArchiv(BinaryWriter& cfg_out, const multi1d<LatticeColorMatrix>& u,
		 int mat_size);


//-----------------------------------------------------------------------
// Write a QCD archive file
// Write a QCD (NERSC) Archive format gauge field
/*
 * \ingroup io
 *
 * \param xml        xml writer holding config info ( Modify )
 * \param u          gauge configuration ( Modify )
 * \param file       path ( Read )
 */    
void writeArchiv(ArchivGauge_t& header, const multi1d<LatticeColorMatrix>& u, const string& file)
{
  BinaryWriter cfg_out(file);

  writeArchivHeader(cfg_out, header);   // write header
  writeArchiv(cfg_out, u, header.mat_size);  // continuing writing after header

  cfg_out.close();
}


// Write a Archive configuration file
/*
 * \ingroup io
 *
 * \param xml        xml writer holding config info ( Read )
 * \param u          gauge configuration ( Read )
 * \param cfg_file   path ( Read )
 */    

void writeArchiv(XMLBufferWriter& xml, const multi1d<LatticeColorMatrix>& u, 
		 const string& cfg_file)
{
  ArchivGauge_t header;
  XMLReader  xml_in(xml);   // use the buffer writer to instantiate a reader
  read(xml_in, "/NERSC", header);

  writeArchiv(header, u, cfg_file);
}


// Write a Archive configuration file
/*
 * \ingroup io
 *
 * \param u          gauge configuration ( Read )
 * \param cfg_file   path ( Read )
 */    

void writeArchiv(const multi1d<LatticeColorMatrix>& u, 
		 const string& cfg_file)
{
  ArchivGauge_t header;
  archivGaugeInit(header);   // default header

  writeArchiv(header, u, cfg_file);
}


QDP_END_NAMESPACE();
