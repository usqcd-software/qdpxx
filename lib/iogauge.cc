// $Id: iogauge.cc,v 1.7 2003-04-02 21:27:43 edwards Exp $
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
void readArchiv(multi1d<LatticeColorMatrix>& u, char file[])
{
#define MAX_LINE_LENGTH 1024
  char line[MAX_LINE_LENGTH];
  
  char datatype[20];    /* We try to grab the datatype */

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
  cout << "Start of header\n";
  fgets(line,MAX_LINE_LENGTH,cfg_in.get());
  cout << line;

  if (strcmp(line,"BEGIN_HEADER\n")!=0)
    QDP_error_exit("Missing BEGIN_HEADER");

  /* Begin loop on lines */
  int  lat_size_cnt = 0;

  while (1)
  {
    fgets(line,MAX_LINE_LENGTH,cfg_in.get());
    cout << line;

    // Scan for the datatype then scan for it
    if ( sscanf(line, "DATATYPE = %s\n", datatype) == 1 ) 
    {
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
      /* Found a lat size */
      if (dd < 1 || dd > Nd)
	QDP_error_exit("oops, dimension number out of bounds");

      lat_size[dd-1] = itmp;
      ++lat_size_cnt;
    }

    if (strcmp(line,"END_HEADER\n")==0) break;
  }

  cout << "End of header\n";

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
            if (bfread((void *) &(su3[0][0][0]),sizeof(float),mat_size,cfg_in.get()) != mat_size)
              QDP_error_exit("Error reading configuration");

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


//-----------------------------------------------------------------------
// Read a SZIN propagator file
//! Read a SZIN propagator file. This is a simple memory dump reader.
void readSzinQprop(LatticePropagator& q, char file[])
{
  BinaryReader cfg_in(file);

  //
  // Read propagator field
  //
  multi1d<int> lattsize_cb = Layout::lattSize();
  ColorMatrix  siteColorField;
  Propagator   siteField;
  float prop[Ns][Ns][Nc][Nc][2];
  size_t prop_size = Ns*Ns*Nc*Nc*2;
  float kappa;

  lattsize_cb[0] /= 2;  // checkerboard in the x-direction in szin

  // Read kappa
  if (bfread((void *)&kappa,sizeof(float),1,cfg_in.get()) != 1)
    QDP_error_exit("Error kappa from reading propagator");

  // Read prop
  for(int cb=0; cb < 2; ++cb)
  {
    for(int sitecb=0; sitecb < Layout::vol()/2; ++sitecb)
    {
      multi1d<int> coord = crtesn(sitecb, lattsize_cb);

      // construct the checkerboard offset
      int sum = 0;
      for(int m=1; m<Nd; m++)
	sum += coord[m];

      // The true lattice x-coord
      coord[0] = 2*coord[0] + ((sum + cb) & 1);

      /* Read an fe variable */
      if (bfread((void *) &(prop[0][0][0][0][0]),sizeof(float),prop_size,cfg_in.get()) != prop_size)
	QDP_error_exit("Error reading propagator");

      /* Copy into the big array */
      for(int s2=0; s2<Ns; s2++)    /* spin */
	for(int s1=0; s1<Ns; s1++)    /* spin */
	  for(int c2=0; c2<Nc; c2++)    /* color */
	    for(int c1=0; c1<Nc; c1++)    /* color */
	    {
	      Real re = prop[s2][s1][c2][c1][0];
	      Real im = prop[s2][s1][c2][c1][1];
	      Complex siteComp = cmplx(re,im);
	      
	      pokeColor(siteColorField,siteComp,c1,c2);  // insert complex into colormatrix
	      pokeSpin(siteField,siteColorField,s1,s2);  // insert color mat into prop
	    }

      pokeSite(q, siteField, coord);
    }
  }

  cfg_in.close();
}

QDP_END_NAMESPACE();
