// -*- C++ -*-
//
// $Id: qdp_util.h,v 1.8 2003-10-15 17:14:39 edwards Exp $
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
void initDefaultSets();

//! Initializer for maps
void initDefaultMaps();

QDP_END_NAMESPACE();


namespace QDPUtil
{
  //! Is the native byte order big endian?
  bool big_endian();

  //! Byte-swap an array of data each of size nmemb
  void byte_swap(void *ptr, size_t size, size_t nmemb);

  //! fread on a binary file written in big-endian order
  size_t bfread(void *ptr, size_t size, size_t nmemb, FILE *stream);

  //! fwrite to a binary file in big-endian order
  size_t bfwrite(void *ptr, size_t size, size_t nmemb, FILE *stream);
}

