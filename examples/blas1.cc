// $Id: blas1.cc,v 1.1 2004-03-20 11:53:37 bjoo Exp $

#include <stdlib.h>
#include <sys/time.h>

#include "qdp.h"
#include "blas1.h"


using namespace QDP;


double QDP_SCALE(LatticeFermion& dest, 
			  const Real& a, 
			  const LatticeFermion& s,
			  int cnt)
{
  StopWatch swatch;

  swatch.start();
  for (; cnt-- > 0; )
    dest =  a * s;
  swatch.stop(); 
  return swatch.getTimeInSeconds();
//  return 2;
}

double QDP_AXPY(LatticeFermion& dest, 
		          const Real& a,
			  const LatticeFermion& s1, 
			  const LatticeFermion& s2,
			  int cnt)
{
  StopWatch swatch;
 
  swatch.start();
  for (; cnt-- > 0; )
    dest = a*s1 + s2;
  swatch.stop();
  return swatch.getTimeInSeconds();
//    return 2.0;
}

double QDP_AXMY(LatticeFermion& dest, 
	                   const Real& a,
			   const LatticeFermion& s1, 
			   const LatticeFermion& s2,
			   int cnt)
{
  StopWatch swatch;
                                                                                
  swatch.start();
  for (; cnt-- > 0; )
    dest = a*s1 - s2;
  swatch.stop();

  return swatch.getTimeInSeconds();
//  return 2.0;
}

double QDP_VADD(LatticeFermion& dest, 
			    const LatticeFermion& s1, 
			    const LatticeFermion& s2,
			    int cnt)
{

  StopWatch swatch;
  swatch.start();

  for (; cnt-- > 0; )
    dest = s1 + s2; 

  swatch.stop();

  return swatch.getTimeInSeconds();
//    return 2.0;
}

double QDP_VSUB(LatticeFermion& dest, 
			  const LatticeFermion& s1, 
			  const LatticeFermion& s2,
			  int cnt)
{
  StopWatch swatch;

  swatch.start();

  for (; cnt-- > 0; )
    dest = s1 - s2;

  swatch.stop();

  return swatch.getTimeInSeconds();
//    return 2;
}

