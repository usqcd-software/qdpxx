// $Id: qdp_iogauge.cc,v 1.10 2003-10-15 17:17:11 edwards Exp $
//
// QDP data parallel interface
/*!
 * @file
 * @brief  Various gauge readers/writers and propagator readers/writers.
 */

#include "qdp.h"
#include "qdp_iogauge.h"

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

//! Initialize header with default values
void archivGaugeInit(ArchivGauge_t& header)
{
  header.mat_size = 12;
  header.nrow = Layout::lattSize();
  header.boundary.resize(Nd);
  header.boundary = 1;   // periodic
  header.sequence_number = 0;
  header.ensemble_id = 0;
  header.ensemble_label = "NERSC archive";
  header.creator = "QDP++";
  header.creator_hardware = "QDP++";
  header.creation_date = "";
  header.archive_date  = "";

  header.w_plaq = 0;   // WARNING: bogus
  header.link = 0;     // WARNING: bogus
}


//! Source header read
void read(XMLReader& xml, const string& path, ArchivGauge_t& header)
{
  XMLReader paramtop(xml, path);

  read(paramtop, "mat_size", header.mat_size);
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
  write(xml, "nrow", header.nrow);

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
 */    

static void readArchivHeader(BinaryReader& cfg_in, ArchivGauge_t& header)
{
  if (Nd != 4)
    QDP_error_exit("Expecting Nd == 4");

  if (Nc != 3)
    QDP_error_exit("Expecting Nc == 3");

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

  /* assume matrix size is 12 (matrix is compressed) 
     and change if we find out otherwise */
  header.mat_size=12;

  /* Begin loop on lines */
  int  lat_size_cnt = 0;

  while (1)
  {
    cfg_in.read(line, max_line_length);
    QDPIO::cout << line << endl;

    // Scan for the datatype then scan for it
    char datatype[64];    /* We try to grab the datatype */
    if ( sscanf(line.c_str(), "DATATYPE = %s", datatype) == 1 ) 
    {
      /* Check if it is uncompressed */
      if (strcmp(datatype, "4D_SU3_GAUGE_3x3") == 0) 
      {
	header.mat_size=18;   /* Uncompressed matrix */
      }
    }

    // Find the lattice size of the gauge field
    int itmp, dd;
    if ( sscanf(line.c_str(), "DIMENSION_%d = %d", &dd, &itmp) == 2 ) 
    {
      /* Found a lat size */
      if (dd < 1 || dd > Nd)
	QDP_error_exit("oops, dimension number out of bounds");

      header.nrow[dd-1] = itmp;
      ++lat_size_cnt;
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
// Read a QCD archive file
//! Read a QCD (NERSC) Archive format gauge field
/*!
 * \ingroup io
 *
 * \param header     structure holding config info ( Modify )
 * \param u          gauge configuration ( Modify )
 * \param file       path ( Read )
 */    
void readArchiv(ArchivGauge_t& header, multi1d<LatticeColorMatrix>& u, const string& file)
{
  BinaryReader cfg_in(file);

  readArchivHeader(cfg_in, header);

  //
  // Read gauge field
  //
  multi1d<int> coord(Nd);
  ColorMatrix  sitefield;
  float su3[3][3][2];
  unsigned int chksum = 0;

  for(int t=0; t < Layout::lattSize()[3]; t++)  /* t */
    for(int z=0; z < Layout::lattSize()[2]; z++)  /* t */
      for(int y=0; y < Layout::lattSize()[1]; y++)  /* y */
        for(int x=0; x < Layout::lattSize()[0]; x++)  /* x */
        {
	  coord[0] = x; coord[1] = y; coord[2] = z; coord[3] = t;

          for(int dd=0; dd<Nd; dd++)        /* dir */
          {
            /* Read an fe variable and write it to the BE */
            cfg_in.readArray((char *)&(su3[0][0][0]),sizeof(float),header.mat_size);
            if (cfg_in.fail())
              QDP_error_exit("Error reading configuration");

            /* Reconstruct the third column  if necessary */
            if( header.mat_size == 12) 
            {
	      su3[2][0][0] = su3[0][1][0]*su3[1][2][0] - su3[0][1][1]*su3[1][2][1]
		- su3[0][2][0]*su3[1][1][0] + su3[0][2][1]*su3[1][1][1];
	      su3[2][0][1] = su3[0][2][0]*su3[1][1][1] + su3[0][2][1]*su3[1][1][0]
		- su3[0][1][0]*su3[1][2][1] - su3[0][1][1]*su3[1][2][0];

	      su3[2][1][0] = su3[0][2][0]*su3[1][0][0] - su3[0][2][1]*su3[1][0][1]
		- su3[0][0][0]*su3[1][2][0] + su3[0][0][1]*su3[1][2][1];
	      su3[2][1][1] = su3[0][0][0]*su3[1][2][1] + su3[0][0][1]*su3[1][2][0]
		- su3[0][2][0]*su3[1][0][1] - su3[0][2][1]*su3[1][0][0];
          
	      su3[2][2][0] = su3[0][0][0]*su3[1][1][0] - su3[0][0][1]*su3[1][1][1]
		- su3[0][1][0]*su3[1][0][0] + su3[0][1][1]*su3[1][0][1];
	      su3[2][2][1] = su3[0][1][0]*su3[1][0][1] + su3[0][1][1]*su3[1][0][0]
		- su3[0][0][0]*su3[1][1][1] - su3[0][0][1]*su3[1][1][0];
            }

            /* Copy into the big array */
            for(int kk=0; kk<Nc; kk++)      /* color */
	    {
              for(int ii=0; ii<Nc; ii++)    /* color */
	      {
		Real re = su3[ii][kk][0];
		Real im = su3[ii][kk][1];
		Complex sitecomp = cmplx(re,im);
		pokeColor(sitefield,sitecomp,ii,kk);

		if ( header.mat_size == 12 ) 
		{
		  /* If compressed ignore 3rd row for checksum */
		  if (ii < 2) 
		  {
		    chksum += *(unsigned int*)(su3+(((ii)*3+kk)*2+0));
		    chksum += *(unsigned int*)(su3+(((ii)*3+kk)*2+1));
		  }
		}
		else 
		{
		  /* If uncompressed take everything for checksum */
		  chksum += *(unsigned int*)(su3+(((ii)*3+kk)*2+0));
		  chksum += *(unsigned int*)(su3+(((ii)*3+kk)*2+1));
		}
	      }
	    }

	    pokeSite(u[dd], sitefield, coord);
          }
        }

  QDPIO::cout << "Computed (in this endian-ness, maybe not Big) checksum = " << chksum << "d\n";

  cfg_in.close();

}


//-----------------------------------------------------------------------
//! Read a Archive configuration file
/*!
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
//! Read a QCD (NERSC) Archive format gauge field
/*!
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


//-----------------------------------------------------------------------
// Write a QCD archive file
//! Write a QCD (NERSC) Archive format gauge field
/*!
 * \ingroup io
 *
 * \param xml        xml writer holding config info ( Modify )
 * \param u          gauge configuration ( Modify )
 * \param file       path ( Read )
 */    
void writeArchiv(ArchivGauge_t& header, const multi1d<LatticeColorMatrix>& u, const string& file)
{
  BinaryWriter cfg_out(file);

  writeArchivHeader(cfg_out, header);

  //
  // Write gauge field
  //
  multi1d<int> coord(Nd);
  ColorMatrix  sitefield;

  for(int t=0; t < Layout::lattSize()[3]; t++)  /* t */
    for(int z=0; z < Layout::lattSize()[2]; z++)  /* t */
      for(int y=0; y < Layout::lattSize()[1]; y++)  /* y */
        for(int x=0; x < Layout::lattSize()[0]; x++)  /* x */
        {
	  coord[0] = x; coord[1] = y; coord[2] = z; coord[3] = t;

          for(int dd=0; dd<Nd; dd++)        /* dir */
          {
	    sitefield = peekSite(u[dd], coord);

	    if ( header.mat_size == 12 ) 
	    {
	      float su3[2][3][2];

	      for(int kk=0; kk<Nc; kk++)      /* color */
		for(int ii=0; ii<2; ii++)    /* color */
		{
		  Complex sitecomp = peekColor(sitefield,ii,kk);
		  su3[ii][kk][0] = toFloat(Real(real(sitecomp)));
		  su3[ii][kk][1] = toFloat(Real(imag(sitecomp)));
		}

	      // Write a site variable
	      if (Layout::primaryNode())
		cfg_out.writeArray((char *)&(su3[0][0][0]),sizeof(float),header.mat_size);
	    }
	    else
	    {
	      float su3[3][3][2];

	      for(int kk=0; kk<Nc; kk++)      /* color */
		for(int ii=0; ii<Nc; ii++)    /* color */
		{
		  Complex sitecomp = peekColor(sitefield,ii,kk);
		  su3[ii][kk][0] = toFloat(Real(real(sitecomp)));
		  su3[ii][kk][1] = toFloat(Real(imag(sitecomp)));
		}
	      
	      // Write a site variable
	      if (Layout::primaryNode())
		cfg_out.writeArray((char *)&(su3[0][0][0]),sizeof(float),header.mat_size);
	    }
          }
        }

  if (cfg_out.fail())
    QDP_error_exit("Error writing configuration");

  cfg_out.close();
}


//! Write a Archive configuration file
/*!
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


//! Write a Archive configuration file
/*!
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
