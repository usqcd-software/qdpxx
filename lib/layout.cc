// $Id: layout.cc,v 1.4 2002-09-26 20:05:34 edwards Exp $
//
// QDP data parallel interface
//
// Layout
//
// This routine provides various layouts, including
//    lexicographic
//    2-checkerboard  (even/odd-checkerboarding of sites)
//    32-style checkerboard (even/odd-checkerboarding of hypercubes)

#include "qdp.h"
#include "proto.h"

#define  USE_LEXICO_LAYOUT
#undef   USE_CB2_LAYOUT
#undef   USE_CB32_LAYOUT

QDP_BEGIN_NAMESPACE(QDP);

/*! Main layout object */
Layout layout;

//! Calculates the lexicographic site index from the coordinate of a site
int local_site(const multi1d<int>& coord, const multi1d<int>& latt_size)
{
  int order = 0;

  for(int mmu=latt_size.size()-1; mmu >= 1; --mmu)
    order = latt_size[mmu-1]*(coord[mmu] + order);

  order += coord[0];

  return order;
}


//! The linearized site index for the corresponding lexicographic site
int Layout::LinearSiteIndex(int site)
{
  multi1d<int> coord = crtesn(site, nrow);

  return LinearSiteIndex(coord);
}


//! The lexicographic site index for the corresponding coordinate
int Layout::LexicoSiteIndex(const multi1d<int>& coord)
{
  return local_site(coord, nrow);
}


//-----------------------------------------------------------------------------
// Auxilliary operations
//! coord[mu]  <- mu  : fill with lattice coord in mu direction
LatticeInteger latticeCoordinate(int mu)
{
  LatticeInteger d;
  Subset s = global_context->Sub();

  if (mu < 0 || mu >= Nd)
    diefunc();

  if (! s.IndexRep())
  {
    const multi1d<int> &nrow = layout.LattSize();

    for(int i=s.Start(); i <= s.End(); ++i) 
    {
      int site = layout.LexicoSiteIndex(i);
      for(int k=0; k <= mu; ++k)
      {
	d.elem(i) = Integer(site % nrow[k]).elem();
	site /= nrow[k];
      } 
    }
  }
  else
    diefunc();

  return d;
}



#if defined(USE_LEXICO_LAYOUT)

#warning "Using a lexicographic layout"

//! The linearized site index for the corresponding coordinate
/*! This layout is a simple lexicographic lattice ordering */
int Layout::LinearSiteIndex(const multi1d<int>& coord)
{
  return local_site(coord, nrow);
}


//! The lexicographic site index from the corresponding linearized site
/*! This layout is a simple lexicographic lattice ordering */
int Layout::LexicoSiteIndex(int linearsite)
{
  return linearsite;
}

//! Initializer for layout
/*! This layout is a simple lexicographic lattice ordering */
void Layout::Init(const multi1d<int>& nrows)
{
  if (nrows.size() != Nd)
    SZ_ERROR("dimension of lattice size not the same as the default");

  vol=1;
  nrow = nrows;
  cb_nrow = nrow;
  for(int i=0; i < Nd; ++i) 
    vol *= nrow[i];
  
  /* Volume of checkerboard. Make sure global variable is set */
  nsubl = 1;
  vol_cb = vol / nsubl;

#if defined(DEBUG)
  fprintf(stderr,"vol=%d, nsubl=%d\n",vol,nsubl);
#endif

  InitDefaultSets();

  // Initialize RNG
  RNG::InitDefaultRNG();
}

#else
#if defined(USE_CB2_LAYOUT)

#warning "Using a 2 checkerboard (red/black) layout"

//! The linearized site index for the corresponding coordinate
/*! This layout is appropriate for a 2 checkerboard (red/black) lattice */
int Layout::LinearSiteIndex(const multi1d<int>& coord)
{
  multi1d<int> cb_coord = coord;

  cb_coord[0] >>= 1;    // Number of checkerboards
    
  int cb = 0;
  for(int m=0; m<coord.size(); ++m)
    cb += coord[m];
  cb = cb & 1;

  return local_site(cb_coord, cb_nrow) + cb*vol_cb;
}


//! The lexicographic site index from the corresponding linearized site
/*! This layout is appropriate for a 2 checkerboard (red/black) lattice */
int Layout::LexicoSiteIndex(int linearsite)
{
  int cb = linearsite / vol_cb;
  multi1d<int> coord = crtesn(linearsite % vol_cb, cb_nrow);

  int cbb = cb;
  for(int m=1; m<coord.size(); ++m)
    cbb += coord[m];
  cbb = cbb & 1;

  coord[0] = 2*coord[0] + cbb;

  return local_site(coord, nrow);
}

//! Initializer for layout
/*! This layout is appropriate for a 2 checkerboard (red/black) lattice */
void Layout::Init(const multi1d<int>& nrows)
{
  if (nrows.size() != Nd)
    SZ_ERROR("dimension of lattice size not the same as the default");

  vol=1;
  nrow = nrows;
  cb_nrow = nrow;
  for(int i=0; i < Nd; ++i) 
    vol *= nrow[i];
  
  /* Volume of checkerboard. Make sure global variable is set */
  nsubl = 2;
  vol_cb = vol / nsubl;

  // Lattice checkerboard size
  cb_nrow[0] = nrow[0] / 2;

#if defined(DEBUG)
  fprintf(stderr,"vol=%d, nsubl=%d\n",vol,nsubl);
#endif

  InitDefaultSets();

  // Initialize RNG
  RNG::InitDefaultRNG();
}

#else
#if defined(USE_CB32_LAYOUT)

#warning "Using a 32 checkerboard layout"

//! The linearized site index for the corresponding coordinate
/*! This layout is appropriate for a 32-style checkerboard lattice */
int Layout::LinearSiteIndex(const multi1d<int>& coord)
{
  int NN = coord.size();
  int subl = coord[NN-1] & 1;
  for(int m=NN-2; m >= 0; --m)
    subl = (subl << 1) + (coord[m] & 1);

  int cb = 0;
  for(int m=0; m < NN; ++m)
    cb += coord[m] >> 1;

  subl += (cb & 1) << NN;   // Final color or checkerboard

  // Construct the checkerboard lattice coord
  multi1d<int> cb_coord(NN);

  cb_coord[0] = coord[0] >> 2;
  for(int m=1; m < NN; ++m)
    cb_coord[m] = coord[m] >> 1;

  return local_site(cb_coord, cb_nrow) + subl*vol_cb;
}


//! The lexicographic site index from the corresponding linearized site
/*! This layout is appropriate for a 32-style checkerboard lattice */
int Layout::LexicoSiteIndex(int linearsite)
{
  int subl = linearsite / vol_cb;
  multi1d<int> coord = crtesn(linearsite % vol_cb, cb_nrow);

  int cb = 0;
  for(int m=1; m<coord.size(); ++m)
    cb += coord[m];
  cb &= 1;

  coord[0] <<= 2;
  for(int m=1; m<coord.size(); ++m)
    coord[m] <<= 1;

  coord[0] ^= cb << 1;
  for(int m=0; m<coord.size(); ++m)
    coord[m] ^= (subl & (1 << m)) >> m;
  coord[0] ^= (subl & (1 << Nd)) >> 1;

  return local_site(coord, nrow);
}


//! Initializer for layout
/*! This layout is appropriate for a 32-style checkerboard lattice */
void Layout::Init(const multi1d<int>& nrows)
{
  if (nrows.size() != Nd)
    SZ_ERROR("dimension of lattice size not the same as the default");

  vol=1;
  nrow = nrows;
  cb_nrow = nrow;
  for(int i=0; i < Nd; ++i) 
    vol *= nrow[i];
  
  /* Volume of checkerboard. Make sure global variable is set */
  nsubl = 1 << (Nd+1);
  vol_cb = vol / nsubl;

  // Lattice checkerboard size
  cb_nrow[0] = nrow[0] >> 2;
  for(int i=1; i < Nd; ++i) 
    cb_nrow[i] = nrow[i] >> 1;
  
#if defined(DEBUG)
  fprintf(stderr,"vol=%d, nsubl=%d\n",vol,nsubl);
#endif

  InitDefaultSets();

  // Initialize RNG
  RNG::InitDefaultRNG();
}

#else

#error "no appropriate layout defined"

#endif
#endif
#endif

QDP_END_NAMESPACE();
