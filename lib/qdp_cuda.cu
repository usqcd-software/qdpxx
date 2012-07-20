// -*- c++ -*-


#include <iostream>

#include "qdp_config_internal.h" 

#include "qdp_cuda.h"
#include "qdp_init.h"

using namespace std;



namespace QDP {


  void inline cudp_check_error(const std::string& msg,cudaError_t& ret)
  {
#ifdef GPU_DEBUG_DEEP
    QDP_info("%s\n", msg.c_str());

    if (ret != cudaSuccess) {
      string tmp(cudaGetErrorString(ret));
      QDP_error_exit("qdp_cuda.cu: %s",tmp.c_str());
    }

    //CudaThreadSynchronize();
    cudaError_t error = cudaGetLastError();
    if (ret != cudaSuccess) {
      string tmp(cudaGetErrorString(ret));
      QDP_error_exit("qdp_cuda.cu: %s",tmp.c_str());
    }

#else
    if (ret != cudaSuccess) {
      QDP_info("%s\n",msg.c_str());
      string tmp(cudaGetErrorString(ret));
      QDP_error_exit("qdp_cuda.cu: %s",tmp.c_str());
    }
#endif
  }


  cudaStream_t * QDPcudastreams;
  cudaEvent_t * QDPevCopied;

  void * CudaGetKernelStream() {
    return (void*)&QDPcudastreams[KERNEL];
  }

  void CudaCreateStreams() {
    QDPcudastreams = new cudaStream_t[2];
    for (int i=0; i<2; i++) {
      QDP_info("Creating CUDA stream %d",i);
      cudaStreamCreate(&QDPcudastreams[i]);
    }
    QDP_info("Creating CUDA event for transfers");
    QDPevCopied = new cudaEvent_t;
    cudaEventCreate(QDPevCopied);

  }

  void CudaSyncKernelStream() {
    cudaStreamSynchronize(QDPcudastreams[KERNEL]);
  }

  void CudaSyncTransferStream() {
    cudaStreamSynchronize(QDPcudastreams[TRANSFER]);
  }

  void CudaRecordAndWaitEvent() {
    cudaEventRecord( *QDPevCopied , QDPcudastreams[TRANSFER] );
    cudaStreamWaitEvent( QDPcudastreams[KERNEL] , *QDPevCopied , 0);
  }

  void CudaSetDevice(int dev)
  {
    cudaError_t ret;
    ret = cudaSetDevice(dev);
    cudp_check_error("cudaSetDevice",ret);
  }

  void CudaGetDeviceCount(int * count)
  {
    cudaError_t ret;
    ret = cudaGetDeviceCount( count );
    cudp_check_error("cudaGetDeviceCount",ret);
  }


  bool CudaHostRegister(void * ptr , size_t size)
  {
    cudaError_t ret;
    //int flags = cudaHostAllocWriteCombined | cudaHostRegisterPortable;
    int flags = 0;
    QDP_info("CUDA host register ptr=%p (%u) size=%lu (%u)",ptr,(unsigned)((size_t)ptr%4096) ,(unsigned long)size,(unsigned)((size_t)size%4096));
    ret = cudaHostRegister(ptr, size, flags);
    cudp_check_error("hostRegister",ret);
    return true;
  }

  
  void CudaHostUnregister(void * ptr )
  {
    cudaError_t ret;
    ret = cudaHostUnregister(ptr);
    cudp_check_error("hostUnregister",ret);
  }
  
  void CudaMemGetInfo(size_t *free,size_t *total)
  {
    cudaError_t ret;
    ret = cudaMemGetInfo( free , total );
    cudp_check_error("getMemInfo",ret);
  }



  bool CudaHostAlloc(void **mem , const size_t size, const int flags)
  {
    cudaError_t ret;
    ret = cudaHostAlloc(mem,size,flags);
    cudp_check_error("cudaHostAlloc",ret);
    return ret == cudaSuccess;
  }


  void CudaHostAllocWrite(void **mem , size_t size)
  {
    cudaError_t ret;
    ret = cudaHostAlloc(mem,size,cudaHostAllocWriteCombined);
    cudp_check_error("cudaHostAlloc write_combined",ret);
  }


  void CudaHostFree(const void *mem)
  {
    cudaError_t ret;
    ret = cudaFreeHost((void *)mem);
    cudp_check_error("cudaFreeHost",ret);
  }




  void CudaMemcpy( const void * dest , const void * src , size_t size)
  {
    cudaError_t ret;
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("cudaMemcpy dest=%p src=%p size=%d" ,  dest , src , size );
#endif
    //QDP_info("cudaMemcpy dest=%p src=%p size=%d" ,  dest , src , size );

    ret = cudaMemcpy(const_cast<void*>(dest),
		     const_cast<void*>(src),
		     size,cudaMemcpyDefault);

    cudp_check_error("cudaMemcpy",ret);
  }


  void CudaMemcpyAsync( const void * dest , const void * src , size_t size )
  {
    cudaError_t ret;
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("cudaMemcpy dest=%p src=%p size=%d" ,  dest , src , size );
#endif

    ret = cudaMemcpyAsync(const_cast<void*>(dest),
			  const_cast<void*>(src),
			  size,cudaMemcpyDefault,
			  QDPcudastreams[TRANSFER]);

    cudp_check_error("cudaMemcpy",ret);
  }


  bool CudaMalloc(void **mem , size_t size )
  {
    cudaError_t ret;
    ret = cudaMalloc(mem,size);
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep( "CudaMalloc %p", *mem );
#endif
    return ret == cudaSuccess;
  }

  void CudaFree(const void *mem )
  {
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep( "CudaFree %p", mem );
#endif
    cudaError_t ret;
    ret = cudaFree(const_cast<void*>(mem));
    cudp_check_error("cudaFree",ret);
  }

  void CudaThreadSynchronize()
  {
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep( "cudaThreadSynchronize" );
#endif
    cudaThreadSynchronize();
  }

  void CudaDeviceSynchronize()
  {
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep( "cudaDeviceSynchronize" );
#endif
    cudaDeviceSynchronize();
  }

}


