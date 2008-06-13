#include "unittest.h"
#include "timeVaxpyDouble.h"

using namespace std;

static double N_SECS=10;
void
time_QDP_PEQ::run(void) 
{

  LatticeFermionD3 x;
  LatticeFermionD3 y;
  Double a = Double(2.3);

  gaussian(x);
  gaussian(y);

  QDPIO::cout << "Timing QDP++ += operation" <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << endl << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    for(int i=0; i < iters; i++) { 
      y[all] += a*x;
    }
    swatch.stop();
    time=swatch.getTimeInSeconds();

    // Average time over nodes
    Internal::globalSum(time);
    time /= (double)Layout::numNodes();

    if (time < n_secs) {
      iters *=2;
      QDPIO::cout << "." << flush;
    }
  }
  while ( time < (double)n_secs );
      
  QDPIO::cout << endl;
  QDPIO::cout << "\t Timing with " << iters << " counts" << endl;

  swatch.reset();
  swatch.start();
  
  for(int i=0; i < iters; ++i) {
    y[all] += a*x;
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  Internal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)iters;

  double flops=(double)(4*Nc*Ns*Layout::vol());
  double perf=(flops/time)/(double)(1024*1024);
  QDPIO::cout << "Operator +=: " << perf << " Mflops" << endl;

}

void time_QDP_AXPYZ::run(void) 
{

  LatticeFermionD3 x;
  LatticeFermionD3 y;
  LatticeFermionD3 z;
  Double a = Double(2.3);

  gaussian(x);
  gaussian(y);

  QDPIO::cout << "Timing QDP++ AXPYZ operation  "  <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << endl << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    for(int i=0; i < iters; i++) { 
      z[all]= a*x + y;
    }
    swatch.stop();
    time=swatch.getTimeInSeconds();

    // Average time over nodes
    Internal::globalSum(time);
    time /= (double)Layout::numNodes();

    if (time < n_secs) {
      iters *=2;
      QDPIO::cout << "." << flush;
    }
  }
  while ( time < (double)n_secs );
      
  QDPIO::cout << endl;
  QDPIO::cout << "\t Timing with " << iters << " counts" << endl;

  swatch.reset();
  swatch.start();
  
  for(int i=0; i < iters; ++i) {
    z[all]=a*x+y;
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  Internal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)iters;

  double flops=(double)(4*Nc*Ns*Layout::vol());
  double perf=(flops/time)/(double)(1024*1024);
  QDPIO::cout << "Triad z=ax + y: " << perf << " Mflops" << endl;

}

void
time_VAXPYZ::run(void) 
{

  LatticeFermionD3 x;
  LatticeFermionD3 y;
  LatticeFermionD3 z;
  Double a = Double(2.3);
  REAL64* xptr = (REAL64 *)&(x.elem(all.start()).elem(0).elem(0).real());
  REAL64* yptr = (REAL64 *)&(y.elem(all.start()).elem(0).elem(0).real());
  REAL64* zptr = &(z.elem(all.start()).elem(0).elem(0).real());
  REAL64 ar = a.elem().elem().elem().elem();
  REAL64* aptr = &ar;
  int n_4vec = (all.end() - all.start() + 1);
  gaussian(x);
  gaussian(y);

  QDPIO::cout << "Timing VAXPYZ4 Kernel " <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << endl << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    
    for(int i=0; i < iters; i++) { 
       vaxpyz4(zptr, aptr, xptr, zptr, n_4vec);
    }
    swatch.stop();
    time=swatch.getTimeInSeconds();

    // Average time over nodes
    Internal::globalSum(time);
    time /= (double)Layout::numNodes();

    if (time < n_secs) {
      iters *=2;
      QDPIO::cout << "." << flush;
    }
  }
  while ( time < (double)n_secs );
      
  QDPIO::cout << endl;
  QDPIO::cout << "\t Timing with " << iters << " counts" << endl;

  swatch.reset();
  swatch.start();
  
  for(int i=0; i < iters; ++i) {
       vaxpyz4(zptr, aptr, xptr, zptr, n_4vec);
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  Internal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)iters;

  double flops=(double)(4*Nc*Ns*Layout::vol());
  double perf=(flops/time)/(double)(1024*1024);
  QDPIO::cout << "VAXPYZ4 Kernrel: " << perf << " Mflops" << endl;

}

void
time_VAXPY::run(void) 
{

  LatticeFermionD3 x;
  LatticeFermionD3 y;
  Double a = Double(2.3);
  REAL64* xptr = (REAL64 *)&(x.elem(all.start()).elem(0).elem(0).real());
  REAL64* yptr = (REAL64 *)&(y.elem(all.start()).elem(0).elem(0).real());
  REAL64 ar = a.elem().elem().elem().elem();
  REAL64* aptr = &ar;
  int n_4vec = (all.end() - all.start() + 1);
  gaussian(x);
  gaussian(y);

  QDPIO::cout << "Timing VAXPY4 Kernel " <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << endl << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    
    for(int i=0; i < iters; i++) { 
       vaxpy4(yptr, aptr, xptr, n_4vec);
    }
    swatch.stop();
    time=swatch.getTimeInSeconds();

    // Average time over nodes
    Internal::globalSum(time);
    time /= (double)Layout::numNodes();

    if (time < n_secs) {
      iters *=2;
      QDPIO::cout << "." << flush;
    }
  }
  while ( time < (double)n_secs );
      
  QDPIO::cout << endl;
  QDPIO::cout << "\t Timing with " << iters << " counts" << endl;

  swatch.reset();
  swatch.start();
  
  for(int i=0; i < iters; ++i) {
       vaxpy4(yptr, aptr, xptr, n_4vec);
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  Internal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)iters;

  double flops=(double)(4*Nc*Ns*Layout::vol());
  double perf=(flops/time)/(double)(1024*1024);
  QDPIO::cout << "VAXPY4 Kernel: " << perf << " Mflops" << endl;

}
