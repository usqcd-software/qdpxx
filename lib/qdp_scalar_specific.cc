// $Id: qdp_scalar_specific.cc,v 1.7 2004-03-24 17:00:31 mcneile Exp $

/*! @file
 * @brief Scalar specific routines
 * 
 * Routines for scalar implementation
 */

#include "qdp.h"
#include "qdp_util.h"

QDP_BEGIN_NAMESPACE(QDP);

//-----------------------------------------------------------------------------
//! Initializer for generic map constructor
void Map::make(const MapFunc& func)
{
#if QDP_DEBUG >= 3
  QDP_info("Map::make");
#endif

  //--------------------------------------
  // Setup the communication index arrays
  goffsets.resize(Layout::vol());

  /* Get the offsets needed for neighbour comm.
     * goffsets(position)
     * the offsets contain the current site, i.e the neighbour for site i
     * is  goffsets(i,dir,mu) and NOT  i + goffset(..) 
     */
  const multi1d<int>& nrow = Layout::lattSize();

  // Loop over the sites on this node
  for(int linear=0; linear < Layout::vol(); ++linear)
  {
    // Get the true lattice coord of this linear site index
    multi1d<int> coord = Layout::siteCoords(0, linear);

    // Source neighbor for this destination site
    multi1d<int> fcoord = func(coord,+1);

    // Source linear site and node
    goffsets[linear] = Layout::linearSiteIndex(fcoord);
  }

#if 0
  for(int ipos=0; ipos < Layout::vol(); ++ipos)
    fprintf(stderr,"goffsets(%d,%d,%d) = %d\n",ipos,goffsets(ipos));
#endif
}




//-----------------------------------------------------------------------
// Read a QCD archive file
//! Read a QCD (NERSC) Archive format gauge field
/*!
 * \ingroup io
 *
 * \param cfg_in     binary writer object ( Modify )
 * \param u          gauge configuration ( Modify )
 */    

void readArchiv(BinaryReader& cfg_in, multi1d<LatticeColorMatrix>& u,
		int mat_size)
{
  ColorMatrix  sitefield;
  float su3[Nc][Nc][2];
  unsigned int chksum = 0;

  // Find the location of each site and send to primary node
  for(int site=0; site < Layout::vol(); ++site)
  {
    multi1d<int> coord = crtesn(site, Layout::lattSize());

    for(int dd=0; dd<Nd; dd++)        /* dir */
    {
      /* Read an fe variable and write it to the BE */
      cfg_in.readArray((char *)&(su3[0][0][0]),sizeof(float), mat_size);
      if (cfg_in.fail())
	QDP_error_exit("Error reading configuration");

      /* Reconstruct the third column  if necessary */
      if (mat_size == 12) 
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

	  if (mat_size == 12) 
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
}



//-----------------------------------------------------------------------
// Write a QCD archive file
//! Write a QCD (NERSC) Archive format gauge field
/*!
 * \ingroup io
 *
 * \param cfg_out    binary writer object ( Modify )
 * \param u          gauge configuration ( Read )
 */    
void writeArchiv(BinaryWriter& cfg_out, const multi1d<LatticeColorMatrix>& u,
		 int mat_size)
{
  ColorMatrix  sitefield;
  float su3[3][3][2];

  // Find the location of each site and send to primary node
  for(int site=0; site < Layout::vol(); ++site)
  {
    multi1d<int> coord = crtesn(site, Layout::lattSize());

    for(int dd=0; dd<Nd; dd++)        /* dir */
    {
      sitefield = peekSite(u[dd], coord);

      if ( mat_size == 12 ) 
      {
	for(int kk=0; kk < Nc; kk++)      /* color */
	  for(int ii=0; ii < Nc-1; ii++)    /* color */
	  {
	    Complex sitecomp = peekColor(sitefield,ii,kk);
	    su3[ii][kk][0] = toFloat(Real(real(sitecomp)));
	    su3[ii][kk][1] = toFloat(Real(imag(sitecomp)));
	  }
      }
      else
      {
	for(int kk=0; kk < Nc; kk++)      /* color */
	  for(int ii=0; ii < Nc; ii++)    /* color */
	  {
	    Complex sitecomp = peekColor(sitefield,ii,kk);
	    su3[ii][kk][0] = toFloat(Real(real(sitecomp)));
	    su3[ii][kk][1] = toFloat(Real(imag(sitecomp)));
	  }
      }

      // Write a site variable
      cfg_out.writeArray((char *)&(su3[0][0][0]),sizeof(float), mat_size);
    }
  }

  if (cfg_out.fail())
    QDP_error_exit("Error writing configuration");
}


QDP_END_NAMESPACE();
