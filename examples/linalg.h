// -*- C++ -*-
// $Id: linalg.h,v 1.1 2003-07-30 18:40:19 edwards Exp $
//
// Include file for test suite

#include "qdp.h"

using namespace QDP;

double QDP_M_eq_M_times_M(LatticeColorMatrix& dest, 
			  const LatticeColorMatrix& s1, 
			  const LatticeColorMatrix& s2,
			  int cnt);

double QDP_M_eq_Ma_times_M(LatticeColorMatrix& dest, 
			  const LatticeColorMatrix& s1, 
			  const LatticeColorMatrix& s2,
			  int cnt);

double QDP_M_peq_M_times_M(LatticeColorMatrix& dest, 
			   const LatticeColorMatrix& s1, 
			   const LatticeColorMatrix& s2,
			   int cnt);

