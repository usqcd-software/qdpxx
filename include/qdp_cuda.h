#ifndef QDP_CUDA_H
#define QDP_CUDA_H


namespace QDP {

  enum QDPCudaStream { TRANSFER=0 , KERNEL=1 };

  bool CudaHostRegister(void * ptr , size_t size);
  void CudaHostUnregister(void * ptr );
  void CudaMemGetInfo(size_t *free,size_t *total);

  bool CudaHostAlloc(void **mem , const size_t size, const int flags);
  void CudaHostAllocWrite(void **mem , size_t size);
  void CudaHostFree(const void *mem);

  void CudaSyncKernelStream();
  void CudaSyncTransferStream();
  void CudaCreateStreams();
  void CudaRecordAndWaitEvent();
  void * CudaGetKernelStream();

  void CudaSetDevice(int dev);
  void CudaGetDeviceCount(int * count);

  void CudaMemcpy(const void * dest ,  const void * src , size_t size);
  void CudaMemcpyAsync(const void * dest ,  const void * src , size_t size );

  bool CudaMalloc( void **mem , const size_t size );
  //  void CudaMallocHost( void **mem , size_t size );
  void CudaFree( const void *mem );

  void CudaThreadSynchronize();
  void CudaDeviceSynchronize();
}

#endif
