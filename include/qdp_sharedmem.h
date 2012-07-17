#ifndef QDP_SHAREDMEM
#define QDP_SHAREDMEM


  template<class T>
  struct SharedMemory
  {
    __device__ inline operator       T*()
    {
      extern __shared__ double __smem[]; // make this double to ensure proper alignment
      return (T*)__smem;
    }
    
    __device__ inline operator const T*() const
    {
      extern __shared__ double __smem[];
      return (T*)__smem;
    }
  };
  
  

#endif

