// -*- C++ -*-
//
// $Id: proto.h,v 1.1 2002-09-12 18:22:16 edwards Exp $
//
// QDP data parallel interface
//
// prototypes used throughout the QDP code

QDP_BEGIN_NAMESPACE(QDP);

//! Decompose a lexicographic site into coordinates
multi1d<int> crtesn(int ipos, const multi1d<int>& latt_size);

//! Calculates the lexicographic site index from the coordinate of a site
int local_site(const multi1d<int>& coord, const multi1d<int>& latt_size);

//! Initializer for subsets
void InitDefaultSets();

QDP_END_NAMESPACE();

