// $Id: qdp_iogauge.cc,v 1.4 2003-06-04 18:22:57 edwards Exp $
//
// QDP data parallel interface
/*!
 * @file
 * @brief  Various gauge readers/writers and propagator readers/writers.
 */

#include "qdp.h"
#include "qdp_util.h"

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

//-----------------------------------------------------------------------
// Read a QCD archive file
//! Read a QCD (NERSC) Archive format gauge field
void readArchiv(multi1d<LatticeColorMatrix>& u, const string& file)
{
  const size_t max_line_length = 128;

  if (Nd != 4)
    QDP_error_exit("Expecting Nd == 4");

  if (Nc != 3)
    QDP_error_exit("Expecting Nc == 3");

  unsigned int chksum = 0;

  BinaryReader cfg_in(file);

  /* assume matrix size is 12 (matrix is compressed) 
     and change if we find out otherwise */
  size_t mat_size=12;

  // The expected lattice size of the gauge field
  multi1d<int> lat_size(Nd);

  /* For now, read and throw away the header */
  string line;

  cout << "Start of header" << endl;
  cfg_in.read(line, max_line_length);
  cout << line << endl;
  
  if (line != string("BEGIN_HEADER"))
    QDP_error_exit("Missing BEGIN_HEADER");

  /* Begin loop on lines */
  int  lat_size_cnt = 0;

  while (1)
  {
    cfg_in.read(line, max_line_length);
    cout << line << endl;

    // Scan for the datatype then scan for it
    char datatype[64];    /* We try to grab the datatype */
    if ( sscanf(line.c_str(), "DATATYPE = %s", datatype) == 1 ) 
    {
      /* Check if it is uncompressed */
      if (strcmp(datatype, "4D_SU3_GAUGE_3x3") == 0) 
      {
	mat_size=18;   /* Uncompressed matrix */
      }
    }

    // Find the lattice size of the gauge field
    int itmp, dd;
    if ( sscanf(line.c_str(), "DIMENSION_%d = %d", &dd, &itmp) == 2 ) 
    {
      /* Found a lat size */
      if (dd < 1 || dd > Nd)
	QDP_error_exit("oops, dimension number out of bounds");

      lat_size[dd-1] = itmp;
      ++lat_size_cnt;
      }
    
    if (line == string("END_HEADER")) break;
  }

  cout << "End of header" << endl;

  // Sanity check
  if (lat_size_cnt != Nd)
    QDP_error_exit("did not find all the lattice sizes");

  // Check lattice size agrees with the one in use
//  cout << "layout size = " << layout.LattSize() << endl;
//  cout << "gauge lat size = " << lat_size << endl;

  for(int dd=0; dd < Nd; ++dd)
    if (lat_size[dd] != Layout::lattSize()[dd])
      QDP_error_exit("readArchiv: archive lattice size does not agree with current size");

  //
  // Read gauge field
  //
  multi1d<int> coord(Nd);
  ColorMatrix  sitefield;
  float su3[3][3][2];

  for(int t=0; t < Layout::lattSize()[3]; t++)  /* t */
    for(int z=0; z < Layout::lattSize()[2]; z++)  /* t */
      for(int y=0; y < Layout::lattSize()[1]; y++)  /* y */
        for(int x=0; x < Layout::lattSize()[0]; x++)  /* x */
        {
	  coord[0] = x; coord[1] = y; coord[2] = z; coord[3] = t;

          for(int dd=0; dd<Nd; dd++)        /* dir */
          {
            /* Read an fe variable and write it to the BE */
            cfg_in.readArray((char *)&(su3[0][0][0]),sizeof(float),mat_size);
//            if (cfg_in.fail())
//              QDP_error_exit("Error reading configuration");

            /* Reconstruct the third column  if necessary */
            if( mat_size == 12) 
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

		if ( mat_size == 12 ) 
		{
		  /* If compressed ignore 3rd row for checksum */
		  if (ii < 2) 
		  {
		    chksum += (unsigned int)(su3[ii][kk][0]);
		    chksum += (unsigned int)(su3[ii][kk][1]);
		  }
		}
		else 
		{
		  /* If uncompressed take everything for checksum */
		  chksum += (unsigned int)(su3[ii][kk][0]);
		  chksum += (unsigned int)(su3[ii][kk][1]);
		}
	      }
	    }

	    pokeSite(u[dd], sitefield, coord);
          }
        }

  printf("Computed checksum = %x\n", chksum);

  cfg_in.close();
}


QDP_END_NAMESPACE();
