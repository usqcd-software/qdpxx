#include "unittest.h"
#include "timeMatEqMatMatDouble.h"

using namespace std;

static double N_SECS=10;


// Allocate 2 vectors. First vector should be cache aligned.
// Second vector should follow the first vector immediately, unless
// this would lead to thrashing in which case it should be displaced
// by 1 line (64 bytes).
void* alloc_cache_aligned_3mat(unsigned num_sites, REAL64** x, REAL64 **y, REAL64** z)
{
  // Opteron L1 cache is 64Kb 
  unsigned long cache_alignment=64*1024;
  unsigned long bytes_per_vec=num_sites*3*3*2*sizeof(double);

  // Allocate contiguously both vectors + cache aligment
  unsigned long bytes_to_alloc=3*bytes_per_vec+cache_alignment-1;
  unsigned long pad = 0;

  // If a vector is exactly a multiple of 64Kb add in a 64 (1 line) byte pad.
  if( bytes_per_vec % cache_alignment == 0 ) {
    //    pad+=64;
  }

  REAL64 *ret_val = (REAL64*)malloc(bytes_to_alloc+pad);
  if( ret_val == 0 ) { 
    QDPIO::cout << "Failed to allocate memory" << endl;
    QDP_abort(1);
  }

  // Now align x
  *x = (REAL64 *)((((ptrdiff_t)(ret_val))+(cache_alignment-1))&(-cache_alignment));
  *y = (REAL64 *)(((ptrdiff_t)(*x))+bytes_per_vec+pad);
  *z = (REAL64 *)(((ptrdiff_t)(*y))+bytes_per_vec+pad);
  
#if 0
  QDPIO::cout << "x is at " << (unsigned long)(*x) << endl;
  QDPIO::cout << "x % cache_alignment = " << (unsigned long)(*x) % cache_alignment << endl;
  QDPIO::cout << "pad is " << pad << endl;
  QDPIO::cout << "veclen=" << bytes_per_vec << endl;
  QDPIO::cout << "y starts at " << (unsigned long)(*y) << endl;
  QDPIO::cout << "z starts at " << (unsigned long)(*z) << endl;
#endif

  return ret_val;

}

// M=M*M kernel (M \in SU3)
void
timeMeqMM_QDP::run(void) 
{

  LatticeColorMatrix x;
  LatticeColorMatrix y;
  LatticeColorMatrix z;
  gaussian(x);
  gaussian(y);

  QDPIO::cout << endl << "Timing  QDP++ MM Kernel " <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    
    for(int i=0; i < iters; i++) { 
      z = x*y;
    }
    swatch.stop();
    time=swatch.getTimeInSeconds();

    // Average time over nodes
    QDPInternal::globalSum(time);
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
    z=x*y;
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  QDPInternal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)iters;

  double flops=(double)(198*Layout::vol());
  double perf=(flops/time)/(double)(1024*1024);
  QDPIO::cout << "QDP++ MM Kernel: " << perf << " Mflops" << endl;

}

// Optimized M=MM kernel (M \in SU3)
void
timeMeqMM::run(void) 
{

  LatticeColorMatrixD3 x;
  LatticeColorMatrixD3 y;

  REAL64* xptr;
  REAL64* yptr;
  REAL64* zptr;

  REAL64* top;

  int n_mat = (all.end() - all.start() + 1);
  top = (REAL64 *)alloc_cache_aligned_3mat(n_mat, &xptr, &yptr, &zptr);

  gaussian(x);
  gaussian(y);

  /* Copy x into x_ptr, y into y_ptr */
  REAL64 *f_x = xptr;
  REAL64 *f_y = yptr;

  for(int site=all.start(); site <= all.end(); site++) { 
    for(int col1=0; col1 < 3; col1++) {
      for(int col2=0; col2 < 3; col2++) { 
	*f_x = x.elem(site).elem().elem(col1,col2).real();
	*f_y = y.elem(site).elem().elem(col1,col2).real();
	f_x++;
	f_y++;
	*f_x = x.elem(site).elem().elem(col1,col2).imag();
	*f_y = y.elem(site).elem().elem(col1,col2).imag();
	f_x++;
	f_y++;
      }
    }
  }

  QDPIO::cout << endl << "Timing SSE D  M=MM  Kernel " <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    
    for(int i=0; i < iters; i++) { 
      ssed_m_eq_mm(zptr, xptr, yptr, n_mat);
    }
    swatch.stop();
    time=swatch.getTimeInSeconds();

    // Average time over nodes
    QDPInternal::globalSum(time);
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
    ssed_m_eq_mm(zptr, xptr, yptr, n_mat);
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  QDPInternal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)iters;

  double flops=(double)(198*Layout::vol());
  double perf=(flops/time)/(double)(1024*1024);
  QDPIO::cout << "SSED MM Kernel: " << perf << " Mflops" << endl;

  free(top);
}

// M+=M*M kernel (M \in SU3)
void
timeMPeqaMM_QDP::run(void) 
{

  LatticeColorMatrix x;
  LatticeColorMatrix y;
  LatticeColorMatrix z;
  gaussian(x);
  gaussian(y);
  gaussian(z);
  Real a(-1.0);


  QDPIO::cout << endl << "Timing  QDP++ M+=MM Kernel " <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    
    for(int i=0; i < iters; i++) { 
      z += x*y;
    }
    swatch.stop();
    time=swatch.getTimeInSeconds();

    // Average time over nodes
    QDPInternal::globalSum(time);
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
    z+=x*y;
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  QDPInternal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)iters;

  double flops=(double)(234*Layout::vol());
  double perf=(flops/time)/(double)(1024*1024);
  QDPIO::cout << "QDP++ M+=MM Kernel: " << perf << " Mflops" << endl;

}

// Optimized M += aM*M kernel
void
timeMPeqaMM::run(void) 
{

  LatticeColorMatrixD3 x;
  LatticeColorMatrixD3 y;
  Double a(-1.0);

  REAL64* xptr;
  REAL64* yptr;
  REAL64* zptr;  
  REAL64* top;

  int n_mat = (all.end() - all.start() + 1);
  top = (REAL64 *)alloc_cache_aligned_3mat(n_mat, &xptr, &yptr, &zptr);

  gaussian(x);
  gaussian(y);

  /* Copy x into x_ptr, y into y_ptr */
  REAL64 *f_x = xptr;
  REAL64 *f_y = yptr;

  for(int site=all.start(); site <= all.end(); site++) { 
    for(int col1=0; col1 < 3; col1++) {
      for(int col2=0; col2 < 3; col2++) { 
	*f_x = x.elem(site).elem().elem(col1,col2).real();
	*f_y = y.elem(site).elem().elem(col1,col2).real();
	f_x++;
	f_y++;
	*f_x = x.elem(site).elem().elem(col1,col2).imag();
	*f_y = y.elem(site).elem().elem(col1,col2).imag();
	f_x++;
	f_y++;
      }
    }
  }

  REAL64* aptr = &(a.elem().elem().elem().elem());

  QDPIO::cout << endl << "Timing SSE D  M+=aMM  Kernel " <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    
    for(int i=0; i < iters; i++) { 
      ssed_m_peq_amm(zptr, aptr, xptr, yptr, n_mat);
    }
    swatch.stop();
    time=swatch.getTimeInSeconds();

    // Average time over nodes
    QDPInternal::globalSum(time);
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
    ssed_m_peq_amm(zptr, aptr, xptr, yptr, n_mat);
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  QDPInternal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)iters;

  double flops=(double)(234*Layout::vol());
  double perf=(flops/time)/(double)(1024*1024);
  QDPIO::cout << "SSED MM Kernel: " << perf << " Mflops" << endl;

  free(top);
}
