// -*- C++ -*-
//
// $Id: qdp_util.h,v 1.10 2005-07-25 17:07:14 edwards Exp $
//
// QDP data parallel interface
//
// prototypes used throughout the QDP code

#ifndef QDP_UTIL_INCLUDE
#define QDP_UTIL_INCLUDE

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

#endif
