// -*- C++ -*-
//
// $Id: qdp_util.h,v 1.3 2003-05-23 04:45:26 edwards Exp $
//
// QDP data parallel interface
//
// prototypes used throughout the QDP code

QDP_BEGIN_NAMESPACE(QDP);

//! Decompose a lexicographic site into coordinates
multi1d<int> crtesn(int ipos, const multi1d<int>& latt_size);

//! Calculates the lexicographic site index from the coordinate of a site
int local_site(const multi1d<int>& coord, const multi1d<int>& latt_size);

//! Unique-ify a list
multi1d<int> uniquify_list(const multi1d<int>& ll);

//! Initializer for subsets
void InitDefaultSets();

//! Initializer for maps
void InitDefaultMaps();

//! Read a QCD (NERSC) Archive format gauge field
void readArchiv(multi1d<LatticeColorMatrix>& u, const std::string& file);

QDP_END_NAMESPACE();


namespace QDPUtil
{
  //! fread on a binary file written in big-endian order
  size_t bfread(void *ptr, size_t size, size_t nmemb, FILE *stream);

  //! fwrite to a binary file in big-endian order
  size_t bfwrite(void *ptr, size_t size, size_t nmemb, FILE *stream);
}

