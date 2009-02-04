#include "unittest.h"
#include <cmath>
#include "timeVaxpyDouble.h"

using namespace std;

static double N_SECS=10;
static double T_SECS=30;

// VAXPBYZ kernel: (vaxpbyz4) z = ax + by
void
time_VAXPBYZ_double::run(void) 
{

  LatticeDiracFermionD3 x;
  LatticeDiracFermionD3 y;
  LatticeDiracFermionD3 z;

  Double a = Double(2.3);
  Double b = Double(0.5);
  gaussian(x); 
  gaussian(y); 

  QDPIO::cout << endl << "Timing VAXPBYZ4 Double Prec Kernel " <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    
    for(int i=0; i < iters; i++) { 
      z = a*x + b*y;
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
  int t_iters = (int)floor((T_SECS/N_SECS)*iters);

  QDPIO::cout << "\t Timing with " << t_iters << " counts" << endl;

  swatch.reset();
  swatch.start();
  
  for(int i=0; i < t_iters; ++i) {
    z = a*x + b*y;
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  Internal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)t_iters;

  double flops=(double)(6*Nc*Ns*Layout::vol());
  double perf=(2*flops/time)/(double)(1024*1024);
  QDPIO::cout << "VAXPBYZ4 Double Kernel: " << perf << " Mflops" << endl;


}


// Floating
// VAXPBYZ kernel: (vaxpbyz4) z = ax + by
void
time_VAXPBYZ_float::run(void) 
{

  LatticeDiracFermionF3 x;
  LatticeDiracFermionF3 y;
  LatticeDiracFermionF3 z;

  Real32 a = Real32(2.3);
  Real32 b = Real32(0.5);

  gaussian(x); 
  gaussian(y); 

  QDPIO::cout << endl << "Timing VAXPBYZ4 Single Kernel " <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    
    for(int i=0; i < iters; i++) { 
      z = a*x + b*y;
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
  int t_iters = (int)floor((T_SECS/N_SECS)*iters);

  QDPIO::cout << "\t Timing with " << t_iters << " counts" << endl;

  swatch.reset();
  swatch.start();
  
  for(int i=0; i < t_iters; ++i) {
    z = a*x + b*y;
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  Internal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)t_iters;

  double flops=(double)(6*Nc*Ns*Layout::vol());
  double perf=(2*flops/time)/(double)(1024*1024);
  QDPIO::cout << "VAXPBYZ4 Kernel (Floating): " << perf << " Mflops" << endl;


}
