#include "unittest.h"
#include "timeVaxpyDouble.h"

using namespace std;

static double N_SECS=10;

// Allocate 2 vectors. First vector should be cache aligned.
// Second vector should follow the first vector immediately, unless
// this would lead to thrashing in which case it should be displaced
// by 1 line (64 bytes).
void* alloc_cache_aligned_2vec(unsigned num_sites, REAL64** x, REAL64 **y)
{
  // Opteron L1 cache is 64Kb 
  unsigned long cache_alignment=32*1024;
  unsigned long bytes_per_vec=num_sites*4*3*2*sizeof(double);

  // Allocate contiguously both vectors + cache aligment
  unsigned long bytes_to_alloc=2*bytes_per_vec+2*cache_alignment-1;
  unsigned long pad = 0;

  // If a vector is exactly a multiple of 64Kb add in a 64 (1 line) byte pad.
  if( bytes_per_vec % cache_alignment == 0 ) {
    pad+=128;
  }

  REAL64 *ret_val = (REAL64*)malloc(bytes_to_alloc+pad);
  if( ret_val == 0 ) { 
    QDPIO::cout << "Failed to allocate memory" << endl;
    QDP_abort(1);
  }

  // Now align x
  *x = (REAL64 *)((((ptrdiff_t)(ret_val))+(cache_alignment-1))&(-cache_alignment));
  *y = (REAL64 *)(((ptrdiff_t)(*x))+bytes_per_vec+pad);

#if 0  
  QDPIO::cout << "x is at " << (unsigned long)(*x) << endl;
  QDPIO::cout << "x % cache_alignment = " << (unsigned long)(*x) % cache_alignment << endl;
  QDPIO::cout << "pad is " << pad << endl;
  QDPIO::cout << "veclen=" << bytes_per_vec << endl;
  QDPIO::cout << "y starts at " << (unsigned long)(*y) << endl;
#endif

  return ret_val;

}

// Allocate 2 vectors. First vector should be cache aligned.
// Second vector should follow the first vector immediately, unless
// this would lead to thrashing in which case it should be displaced
// by 1 line (64 bytes).
void* alloc_cache_aligned_3vec(unsigned num_sites, REAL64** x, REAL64 **y, REAL64** z)
{
  // Opteron L1 cache is 64Kb 
  unsigned long cache_alignment=32*1024;
  unsigned long bytes_per_vec=num_sites*4*3*2*sizeof(double);

  // Allocate contiguously both vectors + cache aligment
  unsigned long bytes_to_alloc=3*bytes_per_vec+2*cache_alignment-1;
  unsigned long pad = 0;

  // If a vector is exactly a multiple of 64Kb add in a 64 (1 line) byte pad.
  if( bytes_per_vec % cache_alignment == 0 ) {
    pad+=128;
  }

  REAL64 *ret_val = (REAL64*)malloc(bytes_to_alloc+2*pad);
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
  QDPIO::cout << "z starts at " << (unsigned long)(*y) << endl;
#endif

  return ret_val;

}

// VAXPBYZ kernel: (vaxpbyz4) z = ax + by
void
time_VAXPBYZ::run(void) 
{

  LatticeDiracFermionD3 x;
  LatticeDiracFermionD3 y;
  LatticeDiracFermionD3 z;

  LatticeDiracFermionD3 x2;
  LatticeDiracFermionD3 y2;
  LatticeDiracFermionD3 z2;

  Double a = Double(2.3);
  Double b = Double(0.5);

  REAL64* top;
  REAL64* top2;

  REAL64* xptr=&(x.elem(0).elem(0).elem(0).real());
  REAL64* yptr=&(y.elem(0).elem(0).elem(0).real()); 
  REAL64* zptr=&(z.elem(0).elem(0).elem(0).real());

  REAL64* xptr2=&(x2.elem(0).elem(0).elem(0).real());
  REAL64* yptr2=&(y2.elem(0).elem(0).elem(0).real());
  REAL64* zptr2=&(z2.elem(0).elem(0).elem(0).real());

  REAL64 ar = a.elem().elem().elem().elem();
  REAL64* aptr = &ar;

  REAL64 br = b.elem().elem().elem().elem();
  REAL64* bptr = &ar;

  int n_4vec = (all.end() - all.start() + 1);
  gaussian(x); gaussian(x2);
  gaussian(y); gaussian(y2);
#if 0
  top =(REAL64*) alloc_cache_aligned_3vec(n_4vec, &xptr, &yptr, &zptr);
  top2 =(REAL64*) alloc_cache_aligned_3vec(n_4vec, &xptr2, &yptr2, &zptr2);
  gaussian(x); 
  gaussian(y);
  /* Copy x into x_ptr, y into y_ptr */
  REAL64 *f_x = xptr;
  REAL64 *f_y = yptr;
  REAL64 *g_x = xptr2;
  REAL64 *g_y = yptr2;

  for(int site=all.start(); site <= all.end(); site++) { 
    for(int spin=0; spin < 4; spin++) {
      for(int col=0; col < 3; col++) { 
	*g_x=*f_x = x.elem(site).elem(spin).elem(col).real();
	*g_y=*f_y = y.elem(site).elem(spin).elem(col).real();
	f_x++;	g_x++;
	f_y++;	g_y++;
	*g_x=*f_x = x.elem(site).elem(spin).elem(col).imag();
	*g_y=*f_y = y.elem(site).elem(spin).elem(col).imag();
	f_x++; g_x++;
	f_y++; g_y++;
      }
    }
  }
#endif

  QDPIO::cout << endl << "Timing VAXPBYZ4 Kernel " <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    
    for(int i=0; i < iters; i++) { 
      vaxpbyz4(zptr, aptr, xptr, bptr, yptr, n_4vec);
      vaxpbyz4(zptr2, aptr, xptr2, bptr, yptr2, n_4vec);
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
      vaxpbyz4(zptr, aptr, xptr, bptr, yptr, n_4vec);
      vaxpbyz4(zptr2, aptr, xptr2, bptr, yptr2, n_4vec);
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  Internal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)iters;

  double flops=(double)(6*Nc*Ns*Layout::vol());
  double perf=(2*flops/time)/(double)(1024*1024);
  QDPIO::cout << "VAXPBYZ4 Kernel: " << perf << " Mflops" << endl;

#if 0
  free(top);
  free(top2);
#endif

}

// VAXPBY kernel (z aliases y): y = ax + by 
void
time_VAXPBY::run(void) 
{

  LatticeDiracFermionD3 x;
  LatticeDiracFermionD3 y;

  Double a = Double(2.3);
  Double b = Double(0.5);

  // REAL64* xptr = (REAL64 *)&(x.elem(all.start()).elem(0).elem(0).real());
  // REAL64* yptr = (REAL64 *)&(y.elem(all.start()).elem(0).elem(0).real());
  REAL64 ar = a.elem().elem().elem().elem();

  REAL64* xptr;
  REAL64* xptr2;
  REAL64* yptr;
  REAL64* yptr2;
  REAL64* top;
  REAL64* top2;

  int n_4vec = (all.end() - all.start() + 1);
  top = (REAL64 *)alloc_cache_aligned_2vec(n_4vec, &xptr, &yptr);  

  top2 = (REAL64 *)alloc_cache_aligned_2vec(n_4vec, &xptr2, &yptr2);

  REAL64* aptr = &ar;

  REAL64 br = b.elem().elem().elem().elem();
  REAL64* bptr = &ar;


  gaussian(x);
  gaussian(y);

  /* Copy x into x_ptr, y into y_ptr */
  REAL64 *f_x = xptr;
  REAL64 *f_y = yptr;
  REAL64 *g_x = xptr2;
  REAL64 *g_y = yptr2;

  for(int site=all.start(); site <= all.end(); site++) { 
    for(int spin=0; spin < 4; spin++) {
      for(int col=0; col < 3; col++) { 
	*g_x=*f_x = x.elem(site).elem(spin).elem(col).real();
	*g_y=*f_y = y.elem(site).elem(spin).elem(col).real();
	f_x++;	g_x++;
	f_y++;	g_y++;
	*g_x=*f_x = x.elem(site).elem(spin).elem(col).imag();
	*g_y=*f_y = y.elem(site).elem(spin).elem(col).imag();
	f_x++; g_x++;
	f_y++; g_y++;
      }
    }
  }

  QDPIO::cout << endl << "Timing VAXPBY4 (aliased) Kernel " <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();

    swatch.start();    
    for(int i=0; i < iters; i++) { 
      vaxpby4(yptr, aptr, xptr, bptr, n_4vec);
      vaxpby4(yptr2, aptr, xptr2, bptr, n_4vec);
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
    vaxpby4(yptr, aptr, xptr, bptr, n_4vec);
    vaxpby4(yptr2, aptr, xptr2, bptr, n_4vec);
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  Internal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)iters;

  double flops=(double)(6*Nc*Ns*Layout::vol());
  double perf=(2*flops/time)/(double)(1024*1024);
  QDPIO::cout << "VAXPBY4 (aliased) Kernel: " << perf << " Mflops" << endl;

  free(top);
  free(top2);
}

// VAXPBYZ kernel: (vaxpbyz4) z = ax + by
void
time_VAXMBYZ::run(void) 
{

  LatticeDiracFermionD3 x;
  LatticeDiracFermionD3 y;
  LatticeDiracFermionD3 z;

  LatticeDiracFermionD3 x2;
  LatticeDiracFermionD3 y2;
  LatticeDiracFermionD3 z2;

  Double a = Double(2.3);
  Double b = Double(0.5);

  REAL64* top;
  REAL64* top2;

  REAL64* xptr=&(x.elem(0).elem(0).elem(0).real());
  REAL64* yptr=&(y.elem(0).elem(0).elem(0).real()); 
  REAL64* zptr=&(z.elem(0).elem(0).elem(0).real());

  REAL64* xptr2=&(x2.elem(0).elem(0).elem(0).real());
  REAL64* yptr2=&(y2.elem(0).elem(0).elem(0).real());
  REAL64* zptr2=&(z2.elem(0).elem(0).elem(0).real());

  REAL64 ar = a.elem().elem().elem().elem();
  REAL64* aptr = &ar;

  REAL64 br = b.elem().elem().elem().elem();
  REAL64* bptr = &ar;

  int n_4vec = (all.end() - all.start() + 1);
  gaussian(x); gaussian(x2);
  gaussian(y); gaussian(y2);

  QDPIO::cout << endl << "Timing VAXMBYZ4 Kernel " <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    
    for(int i=0; i < iters; i++) { 
      vaxmbyz4(zptr, aptr, xptr, bptr, yptr, n_4vec);
      vaxmbyz4(zptr2, aptr, xptr2, bptr, yptr2, n_4vec);
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
    vaxmbyz4(zptr, aptr, xptr, bptr, yptr, n_4vec);
    vaxmbyz4(zptr2, aptr, xptr2, bptr, yptr2, n_4vec);
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  Internal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)iters;

  double flops=(double)(6*Nc*Ns*Layout::vol());
  double perf=(2*flops/time)/(double)(1024*1024);
  QDPIO::cout << "VAXMBYZ4 Kernel: " << perf << " Mflops" << endl;
}




// VAXMBY kernel (z aliases y): y = ax + by 
void
time_VAXMBY::run(void) 
{

  LatticeDiracFermionD3 x,x2;
  LatticeDiracFermionD3 y,y2;

  Double a = Double(2.3);
  Double b = Double(0.5);

  REAL64* xptr = (REAL64 *)&(x.elem(all.start()).elem(0).elem(0).real());
  REAL64* yptr = (REAL64 *)&(y.elem(all.start()).elem(0).elem(0).real());
  REAL64 ar = a.elem().elem().elem().elem();
  REAL64* xptr2 = (REAL64 *)&(x.elem(all.start()).elem(0).elem(0).real());
  REAL64* yptr2 = (REAL64 *)&(y.elem(all.start()).elem(0).elem(0).real());

  int n_4vec = (all.end() - all.start() + 1);
  REAL64* aptr = &ar;

  REAL64 br = b.elem().elem().elem().elem();
  REAL64* bptr = &ar;


  gaussian(x);
  gaussian(y);
  gaussian(x2);
  gaussian(y2);


  QDPIO::cout << endl << "Timing VAXMBY4 (aliased) Kernel " <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    
    for(int i=0; i < iters; i++) { 
      vaxmby4(yptr, aptr, xptr, bptr, n_4vec);
      vaxmby4(yptr2, aptr, xptr2, bptr, n_4vec);
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
    vaxmby4(yptr, aptr, xptr, bptr, n_4vec);
    vaxmby4(yptr2, aptr, xptr2, bptr, n_4vec);
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  Internal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)iters;

  double flops=(double)(6*Nc*Ns*Layout::vol());
  double perf=(2*flops/time)/(double)(1024*1024);
  QDPIO::cout << "VAXMBY4 (aliased) Kernel: " << perf << " Mflops" << endl;

}

// AXPYZ kernel (vaxpyz4): z = ax + y
void
time_VAXPYZ::run(void) 
{

  LatticeDiracFermionD3 x,x2;
  LatticeDiracFermionD3 y,y2;
  LatticeDiracFermionD3 z,z2;

  Double a = Double(2.3);
  REAL64* xptr = (REAL64 *)&(x.elem(0).elem(0).elem(0).real());
  REAL64* yptr = (REAL64 *)&(y.elem(0).elem(0).elem(0).real());
  REAL64* zptr = &(z.elem(0).elem(0).elem(0).real());
  REAL64* xptr2 = (REAL64 *)&(x2.elem(0).elem(0).elem(0).real());
  REAL64* yptr2 = (REAL64 *)&(y2.elem(0).elem(0).elem(0).real());
  REAL64* zptr2 = &(z2.elem(0).elem(0).elem(0).real());

  REAL64 ar = a.elem().elem().elem().elem();
  REAL64* aptr = &ar;
  int n_4vec = (all.end() - all.start() + 1);
  gaussian(x);
  gaussian(y);
  gaussian(x2);
  gaussian(y2);

  QDPIO::cout << endl << "\t Timing VAXPYZ4 Kernel " <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    
    for(int i=0; i < iters; i++) { 
       vaxpyz4(zptr, aptr, xptr, yptr, n_4vec);
       vaxpyz4(zptr2, aptr, xptr2, yptr2, n_4vec);
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
    vaxpyz4(zptr, aptr, xptr, yptr, n_4vec);
    vaxpyz4(zptr2, aptr, xptr2, yptr2, n_4vec);
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  Internal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)iters;

  double flops=(double)(4*Nc*Ns*Layout::vol());
  double perf=(2*flops/time)/(double)(1024*1024);
  QDPIO::cout << "VAXPYZ4 Kernrel: " << perf << " Mflops" << endl;

}


// AXPY kernel (vaxpy4): y = ax + y (or y += ax)
void
time_VAXPY::run(void) 
{

  LatticeDiracFermionD3 x,x2;
  LatticeDiracFermionD3 y,y2;
  Double a = Double(2.3);
  REAL64* xptr=(REAL64 *)(&x.elem(0).elem(0).elem(0).real());
  REAL64* yptr=(REAL64 *)(&y.elem(0).elem(0).elem(0).real());
  REAL64* xptr2=(REAL64 *)(&x2.elem(0).elem(0).elem(0).real());
  REAL64* yptr2=(REAL64 *)(&y2.elem(0).elem(0).elem(0).real());
  REAL64 ar = a.elem().elem().elem().elem();
  REAL64* aptr = &ar;
  int n_4vec = (all.end() - all.start() + 1);

  gaussian(x);
  gaussian(y);
  gaussian(x2);
  gaussian(y2);

  QDPIO::cout << endl <<  "\t Timing VAXPY4 Kernel " <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    
    for(int i=0; i < iters; i++) { 
       vaxpy4(yptr, aptr, xptr, n_4vec);
       vaxpy4(yptr2, aptr, xptr2, n_4vec);
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
       vaxpy4(yptr2, aptr, xptr2, n_4vec);
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  Internal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)iters;

  double flops=(double)(4*Nc*Ns*Layout::vol());
  double perf=(2*flops/time)/(double)(1024*1024);
  QDPIO::cout << "VAXPY4 Kernel: " << perf << " Mflops" << endl;

}

// VAXMYZ kernel z aliases y (vaxmyz4): y = ax - y 
void
time_VAXMY::run(void) 
{

  LatticeDiracFermionD3 x;
  LatticeDiracFermionD3 y;
  Double a = Double(2.3);
  REAL64* xptr;
  REAL64* yptr;
  REAL64* top;

  REAL64 ar = a.elem().elem().elem().elem();
  REAL64* aptr = &ar;
  int n_4vec = (all.end() - all.start() + 1);
  top = (REAL64 *)alloc_cache_aligned_2vec(n_4vec, &xptr, &yptr);

  gaussian(x);
  gaussian(y);
  /* Copy x into x_ptr, y into y_ptr */
  REAL64 *f_x = xptr;
  REAL64 *f_y = yptr;

  for(int site=all.start(); site <= all.end(); site++) { 
    for(int spin=0; spin < 4; spin++) {
      for(int col=0; col < 3; col++) { 
	*f_x = x.elem(site).elem(spin).elem(col).real();
	*f_y = y.elem(site).elem(spin).elem(col).real();
	f_x++;
	f_y++;
	*f_x = x.elem(site).elem(spin).elem(col).imag();
	*f_y = y.elem(site).elem(spin).elem(col).imag();
	f_x++;
	f_y++;
      }
    }
  }

  QDPIO::cout << endl << "\t Timing VAXMY4 Kernel " <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    
    for(int i=0; i < iters; i++) { 
      vaxmy4(yptr, aptr, xptr, n_4vec);
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
    vaxmy4(yptr, aptr, xptr, n_4vec);
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  Internal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)iters;

  double flops=(double)(4*Nc*Ns*Layout::vol());
  double perf=(flops/time)/(double)(1024*1024);
  QDPIO::cout << "VAXMY4 Kernel (aliased): " << perf << " Mflops" << endl;
  free(top);
}

// VAYPX  (vaypx4):  y = x + ay 
void
time_VAYPX::run(void) 
{

  LatticeDiracFermionD3 x;
  LatticeDiracFermionD3 y;
  Double a = Double(2.3);
  REAL64* xptr;
  REAL64* yptr;
  REAL64* top;

  REAL64 ar = a.elem().elem().elem().elem();
  REAL64* aptr = &ar;
  int n_4vec = (all.end() - all.start() + 1);
  top = (REAL64 *)alloc_cache_aligned_2vec(n_4vec, &xptr, &yptr);

  gaussian(x);
  gaussian(y);
  /* Copy x into x_ptr, y into y_ptr */
  REAL64 *f_x = xptr;
  REAL64 *f_y = yptr;

  for(int site=all.start(); site <= all.end(); site++) { 
    for(int spin=0; spin < 4; spin++) {
      for(int col=0; col < 3; col++) { 
	*f_x = x.elem(site).elem(spin).elem(col).real();
	*f_y = y.elem(site).elem(spin).elem(col).real();
	f_x++;
	f_y++;
	*f_x = x.elem(site).elem(spin).elem(col).imag();
	*f_y = y.elem(site).elem(spin).elem(col).imag();
	f_x++;
	f_y++;
      }
    }
  }

  QDPIO::cout << endl << "\t Timing VAYPX4 Kernel " <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    
    for(int i=0; i < iters; i++) { 
      vaypx4(yptr, aptr, xptr, n_4vec);
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
    vaypx4(yptr, aptr, xptr, n_4vec);
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  Internal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)iters;

  double flops=(double)(4*Nc*Ns*Layout::vol());
  double perf=(flops/time)/(double)(1024*1024);
  QDPIO::cout << "VAYPX4 Kernel (aliased): " << perf << " Mflops" << endl;
  free(top);
}




// VAXMYZ kernel. (vaxmyz4): z=ax - y
void
time_VAXMYZ::run(void) 
{

  LatticeDiracFermionD3 x;
  LatticeDiracFermionD3 y;
  LatticeDiracFermionD3 z;
  Double a = Double(2.3);
  REAL64* xptr = (REAL64 *)&(x.elem(all.start()).elem(0).elem(0).real());
  REAL64* yptr = (REAL64 *)&(y.elem(all.start()).elem(0).elem(0).real());
  REAL64* zptr = (REAL64 *)&(z.elem(all.start()).elem(0).elem(0).real());
  REAL64 ar = a.elem().elem().elem().elem();
  REAL64* aptr = &ar;
  int n_4vec = (all.end() - all.start() + 1);
  gaussian(x);
  gaussian(y);

  QDPIO::cout << endl << "Timing VAXMYZ4 Kernel " <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    
    for(int i=0; i < iters; i++) { 
      vaxmyz4(zptr, aptr, xptr, yptr, n_4vec);
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
    vaxmyz4(zptr, aptr, xptr,yptr, n_4vec);
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  Internal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)iters;

  double flops=(double)(4*Nc*Ns*Layout::vol());
  double perf=(flops/time)/(double)(1024*1024);
  QDPIO::cout << "VAXMYZ4 Kernel: " << perf << " Mflops" << endl;

}



// VSCAL Kernel: y = a*x
void
time_VSCAL::run(void) 
{

  LatticeDiracFermionD3 x;
  LatticeDiracFermionD3 y;
  LatticeDiracFermionD3 z;
  Double a = Double(2.3);

  REAL64* xptr = (REAL64 *)&(x.elem(all.start()).elem(0).elem(0).real());
  REAL64* yptr = (REAL64 *)&(y.elem(all.start()).elem(0).elem(0).real());
  REAL64 ar = a.elem().elem().elem().elem();
  REAL64* aptr = &ar;

  int n_4vec = (all.end() - all.start() + 1);
  gaussian(x);

  QDPIO::cout << endl << "Timing VSCAL4 Kernel " <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    
    for(int i=0; i < iters; i++) { 
      vscal4(yptr, aptr, xptr, n_4vec);
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
      vscal4(yptr, aptr, xptr, n_4vec);
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  Internal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)iters;

  double flops=(double)(2*Nc*Ns*Layout::vol());
  double perf=(flops/time)/(double)(1024*1024);
  QDPIO::cout << "VSCAL4 Kernel: " << perf << " Mflops" << endl;

}


//


void
time_LOCAL_SUMSQ::run(void) 
{

  LatticeDiracFermionD3 x;
  Double lnorm;

  REAL64* xptr = (REAL64 *)&(x.elem(all.start()).elem(0).elem(0).real());
  REAL64* res = &(lnorm.elem().elem().elem().elem());
  int n_4vec = (all.end() - all.start() + 1);

  gaussian(x);

  QDPIO::cout << endl << "Timing LOCAL SUMSQ Kernel " <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    
    for(int i=0; i < iters; i++) { 
      local_sumsq4(res,xptr, n_4vec);
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
      local_sumsq4(res, xptr, n_4vec);
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  Internal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)iters;

  double flops=(double)(4*Nc*Ns*Layout::sitesOnNode());
  double perf=(flops/time)/(double)(1024*1024);
  QDPIO::cout << "Local SUMSQ Kernel: " << perf << " Mflops" << endl;

}


void
time_SUMSQ::run(void) 
{

  LatticeDiracFermionD3 x;
  Double lnorm;

  REAL64* xptr = (REAL64 *)&(x.elem(all.start()).elem(0).elem(0).real());
  REAL64* res = &(lnorm.elem().elem().elem().elem());
  int n_4vec = (all.end() - all.start() + 1);

  gaussian(x);

  QDPIO::cout << endl << "SUMSQ Kernel " <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    
    for(int i=0; i < iters; i++) { 
      local_sumsq4(res, xptr, n_4vec);
      Internal::globalSum(lnorm);
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
      local_sumsq4(res, xptr, n_4vec);
      Internal::globalSum(lnorm);
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  Internal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)iters;

  double flops=(double)(4*Nc*Ns*Layout::vol());
  double perf=(flops/time)/(double)(1024*1024);
  QDPIO::cout << "SUMSQ Kernel: " << perf << " Mflops" << endl;

}




void
time_LOCAL_VCDOT::run(void) 
{

  LatticeDiracFermionD3 x;
  LatticeDiracFermionD3 y;
  DComplex lnorm;


  REAL64* xptr = (REAL64 *)&(x.elem(all.start()).elem(0).elem(0).real());
  REAL64* yptr = (REAL64 *)&(y.elem(all.start()).elem(0).elem(0).real());
  REAL64* res = &(lnorm.elem().elem().elem().real());
  int n_4vec = (all.end() - all.start() + 1);

  gaussian(x);
  gaussian(y);

  QDPIO::cout << endl << "Timing LOCAL VCDOT Kernel " <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    
    for(int i=0; i < iters; i++) { 
      local_vcdot4(res, yptr, xptr, n_4vec);
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
    local_vcdot4(res, yptr, xptr, n_4vec);
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  Internal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)iters;

  double flops=(double)(8*Nc*Ns*Layout::sitesOnNode());
  double perf=(flops/time)/(double)(1024*1024);
  QDPIO::cout << "Local VCDOT Kernel: " << perf << " Mflops" << endl;

}


void
time_VCDOT::run(void) 
{

  LatticeDiracFermionD3 x;
  LatticeDiracFermionD3 y;
  DComplex lnorm;

  REAL64* xptr = (REAL64 *)&(x.elem(all.start()).elem(0).elem(0).real());
  REAL64* yptr = (REAL64 *)&(y.elem(all.start()).elem(0).elem(0).real());
  REAL64* res = &(lnorm.elem().elem().elem().real());
  int n_4vec = (all.end() - all.start() + 1);

  gaussian(x);
  gaussian(y);

  QDPIO::cout << endl << "VCDOT Kernel " <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    
    for(int i=0; i < iters; i++) { 
      local_vcdot4(res, yptr, xptr, n_4vec);
      Internal::globalSum(lnorm);
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
    local_vcdot4(res, yptr, xptr, n_4vec);
    Internal::globalSum(lnorm);
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  Internal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)iters;

  double flops=(double)(8*Nc*Ns*Layout::vol());
  double perf=(flops/time)/(double)(1024*1024);
  QDPIO::cout << "VCDOT Kernel: " << perf << " Mflops" << endl;
}

void
time_LOCAL_VCDOT_REAL::run(void) 
{

  LatticeDiracFermionD3 x;
  LatticeDiracFermionD3 y;
  Double lnorm;


  REAL64* xptr = (REAL64 *)&(x.elem(all.start()).elem(0).elem(0).real());
  REAL64* yptr = (REAL64 *)&(y.elem(all.start()).elem(0).elem(0).real());
  REAL64* res = &(lnorm.elem().elem().elem().elem());
  int n_4vec = (all.end() - all.start() + 1);

  gaussian(x);
  gaussian(y);

  QDPIO::cout << endl << "Timing LOCAL VCDOT REAL Kernel " <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    
    for(int i=0; i < iters; i++) { 
      local_vcdot_real4(res, yptr, xptr, n_4vec);
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
    local_vcdot_real4(res, yptr, xptr, n_4vec);
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  Internal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)iters;

  double flops=(double)(4*Nc*Ns*Layout::sitesOnNode());
  double perf=(flops/time)/(double)(1024*1024);
  QDPIO::cout << "Local VCDOT REAL Kernel: " << perf << " Mflops" << endl;

}


void
time_VCDOT_REAL::run(void) 
{

  LatticeDiracFermionD3 x;
  LatticeDiracFermionD3 y;
  Double lnorm;

  REAL64* xptr = (REAL64 *)&(x.elem(all.start()).elem(0).elem(0).real());
  REAL64* yptr = (REAL64 *)&(y.elem(all.start()).elem(0).elem(0).real());
  REAL64* res = &(lnorm.elem().elem().elem().elem());
  int n_4vec = (all.end() - all.start() + 1);

  gaussian(x);
  gaussian(y);

  QDPIO::cout << endl << "VCDOT REAL Kernel " <<endl;

  StopWatch swatch;
  double n_secs = N_SECS;
  int iters=1;
  double time=0;
  QDPIO::cout << "\t Calibrating for " << n_secs << " seconds " << endl;
  do {
    swatch.reset();
    swatch.start();
    
    for(int i=0; i < iters; i++) { 
      local_vcdot_real4(res, yptr, xptr, n_4vec);
      Internal::globalSum(lnorm);
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
    local_vcdot_real4(res, yptr, xptr, n_4vec);
    Internal::globalSum(lnorm);
  }
  swatch.stop();
  time=swatch.getTimeInSeconds();

  // Average time over nodes
  Internal::globalSum(time);
  time /= (double)Layout::numNodes();
  time /= (double)iters;

  double flops=(double)(4*Nc*Ns*Layout::vol());
  double perf=(flops/time)/(double)(1024*1024);
  QDPIO::cout << "VCDOT REAL Kernel: " << perf << " Mflops" << endl;
}



















#if 0
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

  LatticeDiracFermionD3 x;
  LatticeDiracFermionD3 y;
  LatticeDiracFermionD3 z;
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
#endif
