// -*- C++ -*-
// $Id: linalg.h,v 1.2 2003-08-04 18:57:15 edwards Exp $
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

double QDP_M_eq_M_times_Ma(LatticeColorMatrix& dest, 
			   const LatticeColorMatrix& s1, 
			   const LatticeColorMatrix& s2,
			   int cnt);

double QDP_M_eq_Ma_times_Ma(LatticeColorMatrix& dest, 
			    const LatticeColorMatrix& s1, 
			    const LatticeColorMatrix& s2,
			    int cnt);

double QDP_M_peq_M_times_M(LatticeColorMatrix& dest, 
			   const LatticeColorMatrix& s1, 
			   const LatticeColorMatrix& s2,
			   int cnt);

