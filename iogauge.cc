// $Id: iogauge.cc,v 1.1 2002-10-26 01:54:30 edwards Exp $
//
// QDP data parallel interface
//

#include "qdp.h"
#include "proto.h"

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
void readArchiv(multi1d<LatticeGauge>& u, char file[])
{
#define MAX_LINE_LENGTH 1024
  char line[MAX_LINE_LENGTH];
  
  char datatype[20];    /* We try to grab the datatype */

  if (Nd != 4)
    SZ_ERROR("Expecting Nd == 4");

  if (Nc != 3)
    SZ_ERROR("Expecting Nc == 3");

  unsigned int chksum = 0;

  BinaryReader cfg_in(file);

  /* assume matrix size is 12 (matrix is compressed) 
     and change if we find out otherwise */
  size_t mat_size=12;

  // The expected lattice size of the gauge field
  multi1d<int> lat_size(Nd);

  /* For now, read and throw away the header */
  cerr << "Start of header\n";
  fgets(line,MAX_LINE_LENGTH,cfg_in.get());
  cerr << line;

  if (strcmp(line,"BEGIN_HEADER\n")!=0)
    SZ_ERROR("Missing BEGIN_HEADER");

  /* Begin loop on lines */
  int  lat_size_cnt = 0;

  while (1)
  {
    fgets(line,MAX_LINE_LENGTH,cfg_in.get());
    cerr << line;

    // Scan for the datatype then scan for it
    if ( sscanf(line, "DATATYPE = %s\n", datatype) == 1 ) 
    {
      cerr << "datatype: " << line;
      
      /* Check if it is uncompressed */
      if (strcmp(datatype, "4D_SU3_GAUGE_3x3") == 0) 
      {
	mat_size=18;   /* Uncompressed matrix */
      }
    }

    // Find the lattice size of the gauge field
    int itmp, dd;
    if ( sscanf(line, "DIMENSION_%d = %d\n", &dd, &itmp) == 2 ) 
    {
      cerr << "latsize: " << line;

      /* Found a lat size */
      if (dd < 1 || dd > Nd)
	SZ_ERROR("oops, dimension number out of bounds");

      lat_size[dd-1] = itmp;
      ++lat_size_cnt;
    }

    if (strcmp(line,"END_HEADER\n")==0) break;
  }

  cerr << "End of header\n";

  // Sanity check
  if (lat_size_cnt != Nd)
    SZ_ERROR("did not find all the lattice sizes");

  // Check lattice size agrees with the one in use
  cerr << "layout size = " << layout.LattSize() << endl;
  cerr << "gauge lat size = " << lat_size << endl;

  for(int dd=0; dd < Nd; ++dd)
    if (lat_size[dd] != layout.LattSize()[dd])
      SZ_ERROR("readArchiv: archive lattice size does not agree with current size");

  //
  // Read gauge field
  //
  multi1d<int> coord(Nd);
  Gauge  sitefield;
  float su3[3][3][2];

  for(int t=0; t < layout.LattSize()[3]; t++)  /* t */
    for(int z=0; z < layout.LattSize()[2]; z++)  /* t */
      for(int y=0; y < layout.LattSize()[1]; y++)  /* y */
        for(int x=0; x < layout.LattSize()[0]; x++)  /* x */
        {
	  coord[0] = x; coord[1] = y; coord[2] = z; coord[3] = t;

          for(int dd=0; dd<Nd; dd++)        /* dir */
          {
            /* Read an fe variable and write it to the BE */
            if (bfread((void *) &(su3[0][0][0]),sizeof(float),mat_size,cfg_in.get()) != mat_size)
            {
              SZ_ERROR("Error reading configuration");
            }

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
