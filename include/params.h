// -*- C++ -*-
// $Id: params.h,v 1.2 2002-09-12 18:47:53 edwards Exp $
//
// QDP data parallel interface
//
// Basic parameters

QDP_BEGIN_NAMESPACE(QDP);

//! Number of dimensions
#define ND  2

//! Number of colors
#define NC  3

//! Number of spin components
#define NS  4

//! Lattice size
#define  LX0   2
#define  LX1   2
#define  LX2   2
#define  LX3   2


const int Nd = ND;
const int Nc = NC;
const int Ns = NS;


#if ND == 2
#define  VOLUME   LX0*LX1
#elif ND == 3
#define  VOLUME   LX0*LX1*LX2
#elif ND == 4
#define  VOLUME   LX0*LX1*LX2*LX3
#else
#error "unsupported number of dimensions"
#endif


QDP_END_NAMESPACE();
