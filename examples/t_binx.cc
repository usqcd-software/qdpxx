// $Id: t_binx.cc,v 1.6 2005-01-27 19:41:46 mcneile Exp $
//
// Write out binary with some XML markup
// in the binx format.
//

#include <iostream>
#include <cstdio>

#include "qdp.h"

using namespace QDP;


int main(int argc, char **argv)
{
  // Put the machine into a known state
  QDP_initialize(&argc, &argv);

  // Setup the layout
  const int foo[] = {2,2,2,2};
  multi1d<int> nrow(Nd);
  nrow = foo;  // Use only Nd elements
  Layout::setLattSize(nrow);
  Layout::create();

  LatticeReal a;
  double d = 17;
  int bb = 4 ;
  random(a);

  QDPIO::cout << "Test the binx IO routines\n";

  BinxWriter tobinary("t_io.bin");
  //  write(tobinary, a);
  write(tobinary, d);
  write(tobinary, bb);
  float cc = 3.2 ;
  write(tobinary, cc);

#if 0
  multi1d<int> dd(2);
  dd[0] = dd[1] = 7;
  write(tobinary,dd);
#endif

  tobinary.close();
  // Time to bolt
  QDP_finalize();

  // test the multi2d binary IO routines
  QDPIO::cout << "Test the multi2d routines\n";

  BinaryWriter dump("dump.bin");
  multi2d<Real> am(3,2); 
  for(int ii=0 ; ii < 3 ; ++ii) 
    for(int jj=0 ; jj < 2 ; ++jj) 
      am[ii][jj] = 0.5 + ii + 0.4 * jj ; 
  write(dump,am); 
  // write(dump,am,3,2); 
  dump.close(); 

  // read back i n
  multi2d<Real> am_un(3,2); 
  BinaryReader un_dump("dump.bin");
    read(un_dump,am_un); 
  //  read(un_dump,am_un,3,2); 
  un_dump.close() ; 

  for(int ii=0 ; ii < 3 ; ++ii) 
    for(int jj=0 ; jj < 2 ; ++jj) 
      {
	float xx = toFloat(am[ii][jj]) ; 
	float yy = toFloat(am_un[ii][jj]) ;

	if(  xx != yy  ) 
	  {
	    QDPIO::cout << "Error at i,j = " << ii << " " << jj  ;
	    QDPIO::cout << " xx=  " << xx << " yy= " << yy << "\n";
	  }
      }



  // test the multi3d binary IO routines
  QDPIO::cout << "Test the multi3d routines\n";


  BinaryWriter dump3("dump3.bin");
  multi3d<Real> am3(3,2,4); 
  for(int ii=0 ; ii < 3 ; ++ii) 
    for(int jj=0 ; jj < 2 ; ++jj) 
      for(int kk=0 ; kk < 4 ; ++kk) 
      am3[ii][jj][kk] = 0.5 + kk + 3.5* (ii + 0.4 * jj) ; 

  write(dump3,am3); 
  dump3.close(); 


  BinaryReader un_dump3("dump3.bin");
  multi3d<Real> un_am3(3,2,4); 
  read(un_dump3,un_am3); 
  un_dump3.close(); 


  for(int ii=0 ; ii < 3 ; ++ii) 
    for(int jj=0 ; jj < 2 ; ++jj) 
      for(int kk=0 ; kk < 4 ; ++kk) 
	{
	  float xx = toFloat(am3[ii][jj][kk]) ; 
	  float yy = toFloat(un_am3[ii][jj][kk]) ;

	if(  xx != yy  ) 
	  {
	    QDPIO::cout << "Error at i,j = " << ii << " " << jj << " " << kk  ;
	    QDPIO::cout << " xx=  " << xx << " yy= " << yy << "\n";
	  }
      }




  exit(0);
}
