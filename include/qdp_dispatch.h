#ifndef QDP_DISPATCH_H
#define QDP_DISPATCH_H


#if defined(QDP_USE_OMP_THREADS)
#warning QDP using OpenMP threading
    
/* OpenMP threading version of the dispatch.*/

#include <omp.h>

template<class Arg>
void dispatch_to_threads(int numSiteTable, Arg a, void (*func)(int,int,int, Arg*)){
   
  int threads_num;
  int chucksize;
  int myId;
  int low = 0;
  int high = numSiteTable;
   
#pragma omp parallel shared(numSiteTable, threads_num, a) private(chucksize, myId, low, high) default(shared)
    {
     
      threads_num = omp_get_num_threads();
       
      chucksize = numSiteTable/threads_num;
      myId = omp_get_thread_num();
      low = chucksize * myId;
      high = chucksize * (myId+1);
      
      func(low, high, myId, &a);
    }
}

#else

#if defined(QDP_USE_QMT_THREADS)
#warning QDP using QMT threading

 /* QMT threading version of the dispatch. Call the qmt_call routine
     with userfunc, numSiteTable, and argument */

#include <qmt.h>

template<class Arg>
void dispatch_to_threads(int numSiteTable, Arg a, void (*func)(int,int,int,Arg*)){
 
   qmt_call((qmt_userfunc_t)func, numSiteTable, &a);
 
}


#else

template<class Arg>
void dispatch_to_threads(int numSiteTable, Arg a, void (*func)(int,int,int,Arg*)){
   
  int low = 0;
  int high = numSiteTable;
   
  func(low, high, 0, &a);
    
}

 
#endif

#endif

#endif
