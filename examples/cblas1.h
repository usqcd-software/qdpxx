// -*- C++ -*-
// $Id: cblas1.h,v 1.1 2004-05-09 11:54:43 bjoo Exp $
//
// Include file for test suite

#ifndef BLAS1_H
#define BLAS1_H

#include "qdp.h"

using namespace QDP;


double QDP_CSCALE(LatticeFermion& dest, 
		 const Complex& a, 
		 const LatticeFermion& s,
		 int cnt);

double QDP_CAXPY(LatticeFermion& dest, 
		const Complex& a,
		const LatticeFermion& s1, 
		const LatticeFermion& s2,
		int cnt);

double QDP_CAXMY(LatticeFermion& dest, 
		const Complex& a,
		const LatticeFermion& s1, 
		const LatticeFermion& s2,
		int cnt);

double QDP_CAXPBY(LatticeFermion& dest, 
		 const Complex& a,
		 const Complex& b,
		 const LatticeFermion& s1, 
		 const LatticeFermion& s2,
		 int cnt);

double QDP_CAXMBY(LatticeFermion& dest, 
		 const Complex& a,
		 const Complex& b,
		 const LatticeFermion& s1, 
		 const LatticeFermion& s2,
		 int cnt);

#endif
