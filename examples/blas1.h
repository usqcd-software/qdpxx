// -*- C++ -*-
// $Id: blas1.h,v 1.1 2004-03-20 11:53:37 bjoo Exp $
//
// Include file for test suite

#include "qdp.h"

using namespace QDP;


double QDP_SCALE(LatticeFermion& dest, 
		 const Real& a, 
		 const LatticeFermion& s,
		 int cnt);

double QDP_AXPY(LatticeFermion& dest, 
		const Real& a,
		const LatticeFermion& s1, 
		const LatticeFermion& s2,
		int cnt);

double QDP_AXMY(LatticeFermion& dest, 
		const Real& a,
		const LatticeFermion& s1, 
		const LatticeFermion& s2,
		int cnt);

double QDP_VADD(LatticeFermion& dest, 
		const LatticeFermion& s1, 
		const LatticeFermion& s2,
		int cnt);

double QDP_VSUB(LatticeFermion& dest, 
		const LatticeFermion& s1, 
		const LatticeFermion& s2,
		int cnt);

