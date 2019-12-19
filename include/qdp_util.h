// -*- C++ -*-
//
//
// QDP data parallel interface
//
// prototypes used throughout the QDP code

#ifndef QDP_UTIL_INCLUDE
#define QDP_UTIL_INCLUDE

namespace QDP {

//! Decompose a lexicographic site into coordinates
multi1d<index_t> crtesn(index_t ipos, const multi1d<index_t>& latt_size);

//! Calculates the lexicographic site index from the coordinate of a site
index_t local_site(const multi1d<index_t>& coord, const multi1d<index_t>& latt_size);

//! Unique-ify a list
multi1d<index_t> uniquify_list(const multi1d<index_t>& ll);

//! Initializer for subsets
void initDefaultSets();

//! Initializer for maps
void initDefaultMaps();

} // namespace QDP

#endif
