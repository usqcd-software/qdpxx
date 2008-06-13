#include "qdp.h"
#include "testVaxpbyDouble.h"

#include "unittest.h"

// #include "scalarsite_sse/sse_blas_vaxpy3_double.h"

using namespace QDP;
using namespace std;
using namespace Assertions;

// Trinity of tests: Check SSE against Handrolled
//                   Check Handrolled against QDP++
//                   Check QDP++ against SSE


// Test 1. Check hand rolled against -- 'optimized'
void
testVaxpbyz4_1::run()
{
  Double a=Double(2.3);
  Double b=Double(-3.6);
  LatticeDiracFermionD3 x;
  LatticeDiracFermionD3 y;
  LatticeDiracFermionD3 z1;
  LatticeDiracFermionD3 z2;

  gaussian(x);
  gaussian(y);

  // Loop over sites
  for(int i=all.start(); i <= all.end(); i++) { 
    // Loop over spins
    for(int spin=0; spin < 4; spin++) {
      // Loop over colors (case 3 only) 
      for(int col=0; col < 3; col++) { 

	z1.elem(i).elem(spin).elem(col).real() = 
	  a.elem().elem().elem().elem()*x.elem(i).elem(spin).elem(col).real()
	  + b.elem().elem().elem().elem()*y.elem(i).elem(spin).elem(col).real();

	z1.elem(i).elem(spin).elem(col).imag() = 
	  a.elem().elem().elem().elem()*x.elem(i).elem(spin).elem(col).imag()
	  + b.elem().elem().elem().elem()*y.elem(i).elem(spin).elem(col).imag();
      }
    }
  }

  int n_4vec = (all.end() - all.start() + 1);
  REAL64* xptr = (REAL64 *)&(x.elem(all.start()).elem(0).elem(0).real());
  REAL64* yptr = &(y.elem(all.start()).elem(0).elem(0).real());
  REAL64* zptr = &(z2.elem(all.start()).elem(0).elem(0).real());

  REAL64* aptr = &(a.elem().elem().elem().elem());
  REAL64* bptr = &(b.elem().elem().elem().elem());

  vaxpbyz4(zptr, aptr,xptr, bptr, yptr, n_4vec);

  for(int i=all.start(); i <= all.end(); i++) { 
    // Loop over spins
    for(int spin=0; spin < 4; spin++) {
      // Loop over colors (case 3 only) 
      for(int col=0; col < 3; col++) { 

	double realdiff = z1.elem(i).elem(spin).elem(col).real()
	  - z2.elem(i).elem(spin).elem(col).real();

	assertion( fabs(realdiff) < 1.0e-14 );

	double imagdiff = z1.elem(i).elem(spin).elem(col).imag()
	  - z2.elem(i).elem(spin).elem(col).imag();

	assertion( fabs(imagdiff) < 1.0e-14 );

      }
    }
  }
}


// Test 2. Check hand QDP++ against -- 'optimized'
void
testVaxpbyz4_2::run()
{
  Double a=Double(2.3);
  Double b=Double(-3.6);
  LatticeDiracFermionD3 x;
  LatticeDiracFermionD3 y;
  LatticeDiracFermionD3 z1;
  LatticeDiracFermionD3 z2;

  gaussian(x);
  gaussian(y);

  z1 = a*x + b*y;

  int n_4vec = (all.end() - all.start() + 1);
  REAL64* xptr = (REAL64 *)&(x.elem(all.start()).elem(0).elem(0).real());
  REAL64* yptr = &(y.elem(all.start()).elem(0).elem(0).real());
  REAL64* zptr = &(z2.elem(all.start()).elem(0).elem(0).real());

  REAL64* aptr = &(a.elem().elem().elem().elem());
  REAL64* bptr = &(b.elem().elem().elem().elem());

  vaxpbyz4(zptr, aptr, xptr,bptr,yptr, n_4vec);

  for(int i=all.start(); i <= all.end(); i++) { 
    // Loop over spins
    for(int spin=0; spin < 4; spin++) {
      // Loop over colors (case 3 only) 
      for(int col=0; col < 3; col++) { 

	double realdiff = z1.elem(i).elem(spin).elem(col).real()
	  - z2.elem(i).elem(spin).elem(col).real();

	assertion( fabs(realdiff) < 1.0e-14 );

	double imagdiff = z1.elem(i).elem(spin).elem(col).imag()
	  - z2.elem(i).elem(spin).elem(col).imag();

	assertion( fabs(imagdiff) < 1.0e-14 );

      }
    }
  }
}



// Test 3. Check hand rolled against QDP++
void
testVaxpbyz4_3::run()
{
  Double a=Double(2.3);
  Double b=Double(-3.6);
  LatticeDiracFermionD3 x;
  LatticeDiracFermionD3 y;
  LatticeDiracFermionD3 z1;
  LatticeDiracFermionD3 z2;

  gaussian(x);
  gaussian(y);

  // Loop over sites
  for(int i=all.start(); i <= all.end(); i++) { 
    // Loop over spins
    for(int spin=0; spin < 4; spin++) {
      // Loop over colors (case 3 only) 
      for(int col=0; col < 3; col++) { 

	z1.elem(i).elem(spin).elem(col).real() = 
	  a.elem().elem().elem().elem()*x.elem(i).elem(spin).elem(col).real()
	  + b.elem().elem().elem().elem()*y.elem(i).elem(spin).elem(col).real();

	z1.elem(i).elem(spin).elem(col).imag() = 
	  a.elem().elem().elem().elem()*x.elem(i).elem(spin).elem(col).imag()
	  + b.elem().elem().elem().elem()*y.elem(i).elem(spin).elem(col).imag();
      }
    }
  }

  z2 = a*x + b*y;

  for(int i=all.start(); i <= all.end(); i++) { 
    // Loop over spins
    for(int spin=0; spin < 4; spin++) {
      // Loop over colors (case 3 only) 
      for(int col=0; col < 3; col++) { 

	double realdiff = z1.elem(i).elem(spin).elem(col).real()
	  - z2.elem(i).elem(spin).elem(col).real();

	assertion( fabs(realdiff) < 1.0e-14 );

	double imagdiff = z1.elem(i).elem(spin).elem(col).imag()
	  - z2.elem(i).elem(spin).elem(col).imag();

	assertion( fabs(imagdiff) < 1.0e-14 );

      }
    }
  }
}


/* VAXMBY Variants */

// Test 1. Check hand rolled against -- 'optimized'
void
testVaxmbyz4_1::run()
{
  Double a=Double(2.3);
  Double b=Double(-3.6);
  LatticeDiracFermionD3 x;
  LatticeDiracFermionD3 y;
  LatticeDiracFermionD3 z1;
  LatticeDiracFermionD3 z2;

  gaussian(x);
  gaussian(y);

  // Loop over sites
  for(int i=all.start(); i <= all.end(); i++) { 
    // Loop over spins
    for(int spin=0; spin < 4; spin++) {
      // Loop over colors (case 3 only) 
      for(int col=0; col < 3; col++) { 

	z1.elem(i).elem(spin).elem(col).real() = 
	  a.elem().elem().elem().elem()*x.elem(i).elem(spin).elem(col).real()
	  - b.elem().elem().elem().elem()*y.elem(i).elem(spin).elem(col).real();

	z1.elem(i).elem(spin).elem(col).imag() = 
	  a.elem().elem().elem().elem()*x.elem(i).elem(spin).elem(col).imag()
	  - b.elem().elem().elem().elem()*y.elem(i).elem(spin).elem(col).imag();
      }
    }
  }

  int n_4vec = (all.end() - all.start() + 1);
  REAL64* xptr = (REAL64 *)&(x.elem(all.start()).elem(0).elem(0).real());
  REAL64* yptr = &(y.elem(all.start()).elem(0).elem(0).real());
  REAL64* zptr = &(z2.elem(all.start()).elem(0).elem(0).real());

  REAL64* aptr = &(a.elem().elem().elem().elem());
  REAL64* bptr = &(b.elem().elem().elem().elem());

  vaxmbyz4(zptr, aptr, xptr,bptr,yptr, n_4vec);

  for(int i=all.start(); i <= all.end(); i++) { 
    // Loop over spins
    for(int spin=0; spin < 4; spin++) {
      // Loop over colors (case 3 only) 
      for(int col=0; col < 3; col++) { 

	double realdiff = z1.elem(i).elem(spin).elem(col).real()
	  - z2.elem(i).elem(spin).elem(col).real();

	assertion( fabs(realdiff) < 1.0e-14 );

	double imagdiff = z1.elem(i).elem(spin).elem(col).imag()
	  - z2.elem(i).elem(spin).elem(col).imag();

	assertion( fabs(imagdiff) < 1.0e-14 );

      }
    }
  }
}


// Test 2. Check hand QDP++ against -- 'optimized'
void
testVaxmbyz4_2::run()
{
  Double a=Double(2.3);
  Double b=Double(-3.6);
  LatticeDiracFermionD3 x;
  LatticeDiracFermionD3 y;
  LatticeDiracFermionD3 z1;
  LatticeDiracFermionD3 z2;

  gaussian(x);
  gaussian(y);

  z1 = a*x - b*y;

  int n_4vec = (all.end() - all.start() + 1);
  REAL64* xptr = (REAL64 *)&(x.elem(all.start()).elem(0).elem(0).real());
  REAL64* yptr = &(y.elem(all.start()).elem(0).elem(0).real());
  REAL64* zptr = &(z2.elem(all.start()).elem(0).elem(0).real());

  REAL64* aptr = &(a.elem().elem().elem().elem());
  REAL64* bptr = &(b.elem().elem().elem().elem());

  vaxmbyz4(zptr, aptr, xptr,bptr,yptr, n_4vec);

  for(int i=all.start(); i <= all.end(); i++) { 
    // Loop over spins
    for(int spin=0; spin < 4; spin++) {
      // Loop over colors (case 3 only) 
      for(int col=0; col < 3; col++) { 

	double realdiff = z1.elem(i).elem(spin).elem(col).real()
	  - z2.elem(i).elem(spin).elem(col).real();

	assertion( fabs(realdiff) < 1.0e-14 );

	double imagdiff = z1.elem(i).elem(spin).elem(col).imag()
	  - z2.elem(i).elem(spin).elem(col).imag();

	assertion( fabs(imagdiff) < 1.0e-14 );

      }
    }
  }
}



// Test 3. Check hand rolled against QDP++
void
testVaxmbyz4_3::run()
{
  Double a=Double(2.3);
  Double b=Double(-3.6);
  LatticeDiracFermionD3 x;
  LatticeDiracFermionD3 y;
  LatticeDiracFermionD3 z1;
  LatticeDiracFermionD3 z2;

  gaussian(x);
  gaussian(y);

  // Loop over sites
  for(int i=all.start(); i <= all.end(); i++) { 
    // Loop over spins
    for(int spin=0; spin < 4; spin++) {
      // Loop over colors (case 3 only) 
      for(int col=0; col < 3; col++) { 

	z1.elem(i).elem(spin).elem(col).real() = 
	  a.elem().elem().elem().elem()*x.elem(i).elem(spin).elem(col).real()
	  - b.elem().elem().elem().elem()*y.elem(i).elem(spin).elem(col).real();

	z1.elem(i).elem(spin).elem(col).imag() = 
	  a.elem().elem().elem().elem()*x.elem(i).elem(spin).elem(col).imag()
	  - b.elem().elem().elem().elem()*y.elem(i).elem(spin).elem(col).imag();
      }
    }
  }

  z2 = a*x - b*y;

  for(int i=all.start(); i <= all.end(); i++) { 
    // Loop over spins
    for(int spin=0; spin < 4; spin++) {
      // Loop over colors (case 3 only) 
      for(int col=0; col < 3; col++) { 

	double realdiff = z1.elem(i).elem(spin).elem(col).real()
	  - z2.elem(i).elem(spin).elem(col).real();

	assertion( fabs(realdiff) < 1.0e-14 );

	double imagdiff = z1.elem(i).elem(spin).elem(col).imag()
	  - z2.elem(i).elem(spin).elem(col).imag();

	assertion( fabs(imagdiff) < 1.0e-14 );

      }
    }
  }
}

