#ifndef REDUCTION_KERNEL_H
#define REDUCTION_KERNEL_H


namespace QDP {

template <class T1,class T2>
void reduce_convert_indirection(int size, int threads, int blocks,
				T1 *d_idata, T2 *d_odata,
				bool indirection,int * siteTable)
{
  int smemSize = threads * sizeof(T2);

  QDPJitArgs cudaArgs;

  string typeT1,typeT2;

  getTypeString( typeT1 , *d_idata , cudaArgs );
  getTypeString( typeT2 , *d_odata , cudaArgs );

  ostringstream osId;
  osId << "reduce_convert_indirection " << typeT1 << " " << typeT2;
  string strId = osId.str();

#ifdef GPU_DEBUG_DEEP
  cout << "strId = " << strId << endl;
#endif

  QDP_debug("reduce_convert_indirection dev!");


  int aSize = cudaArgs.addInt(size); // numsitetable
  int aSiteTable = cudaArgs.addIntPtr( siteTable ); // soffsetDev
  int aInd = cudaArgs.addBool( indirection ); // indir
  int aIdata = cudaArgs.addPtr( (void*)d_idata );  // misc
  int aOdata = cudaArgs.addPtr( (void*)d_odata );  // dest
      
  std::ostringstream sprg;

  sprg << "    typedef " << typeT2 << " T2;" << endl;
  sprg << "    typedef " << typeT1 << " T1;" << endl;
  sprg << "    T1* g_idata = (T1*)(" << cudaArgs.getCode(aIdata) << ");" << endl;
  sprg << "    T2* g_odata = (T2*)(" << cudaArgs.getCode(aOdata) << ");" << endl;

  sprg << "    T2 *sdata = SharedMemory<T2>();" << endl;
  sprg << "    unsigned int tid = threadIdx.x;" << endl;
  sprg << "    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;" << endl;
  sprg << "    unsigned int j;" << endl;
  sprg << "    if (" << cudaArgs.getCode(aInd) << ")" << endl;
  sprg << "      j=" << cudaArgs.getCode(aSiteTable) << "[i];" << endl;
  sprg << "    else" << endl;
  sprg << "      j=i;" << endl;
  sprg << "    " << endl;
  sprg << "    if (i < " << cudaArgs.getCode(aSize) << ")" << endl;
  sprg << "      sdata[tid] = g_idata[j];" << endl;
  sprg << "    else" << endl;
  sprg << "      zero_rep(sdata[tid]);" << endl;
  sprg << "    " << endl;
  sprg << "    __syncthreads();" << endl;
  sprg << "" << endl;
  sprg << "    int next_pow2=1;" << endl;
  sprg << "    while( next_pow2 < blockDim.x ) {" << endl;
  sprg << "      next_pow2 <<= 1;" << endl;
  sprg << "    }" << endl;
  sprg << "" << endl;
  sprg << "    for(unsigned int s=next_pow2/2; s>0; s>>=1)" << endl;
  sprg << "    {" << endl;
  sprg << "        if (tid < s) " << endl;
  sprg << "        {" << endl;
  sprg << "	  if (tid + s < blockDim.x)" << endl;
  sprg << "            sdata[tid] += sdata[tid + s];" << endl;
  sprg << "        }" << endl;
  sprg << "        __syncthreads();" << endl;
  sprg << "    }" << endl;
  sprg << "" << endl;
  sprg << "    if (tid == 0) g_odata[blockIdx.x] = sdata[0];" << endl;

  string prg = sprg.str();

#ifdef GPU_DEBUG_DEEP
  cout << "Cuda kernel code = " << endl << prg << endl << endl;
#endif

  static QDPJit::SharedLibEntry sharedLibEntry;
  if (!QDPJit::Instance().jitFixedGeom( strId , prg , cudaArgs.getDevPtr() , 
					  size , sharedLibEntry , threads, blocks , smemSize )) {
    QDP_error("reduce_convert_indirection() call to cuda jitter failed");
  }

}








template <class T>
void globalMax_kernel(int size, int threads, int blocks, 
		      T *d_idata, T *d_odata)
{
  int smemSize = threads * sizeof(T);

  QDPJitArgs cudaArgs;

  string typeT;

  getTypeString( typeT , *d_idata , cudaArgs );

  ostringstream osId;
  osId << "globalMax " << typeT;
  string strId = osId.str();

#ifdef GPU_DEBUG_DEEP
  cout << "strId = " << strId << endl;
#endif

  QDP_debug("globalMax dev!");

  int aSize = cudaArgs.addInt(size); // numsitetable
  int aIdata = cudaArgs.addPtr( (void*)d_idata );  // misc
  int aOdata = cudaArgs.addPtr( (void*)d_odata );  // dest

  std::ostringstream sprg;


  sprg << "    typedef " << typeT << " T;" << endl;
  sprg << "    T* g_idata = (T*)(" << cudaArgs.getCode(aIdata) << ");" << endl;
  sprg << "    T* g_odata = (T*)(" << cudaArgs.getCode(aOdata) << ");" << endl;

  sprg << "    T *sdata = SharedMemory<T>();" << endl;
  sprg << "    unsigned int tid = threadIdx.x;" << endl;
  sprg << "    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;" << endl;
  sprg << "    if ( i < " << cudaArgs.getCode(aSize) << " )" << endl;
  sprg << "      sdata[tid] = g_idata[i];" << endl;
  sprg << "    else" << endl;
  sprg << "      zero_rep(sdata[tid]);" << endl;
  sprg << "    " << endl;
  sprg << "    __syncthreads();" << endl;
  sprg << "" << endl;
  sprg << "    for(unsigned int s=blockDim.x/2; s>0; s>>=1) " << endl;
  sprg << "    {" << endl;
  sprg << "        if (tid < s) " << endl;
  sprg << "        {" << endl;
  sprg << " 	  if (toBool(sdata[tid + s] > sdata[tid]))" << endl;
  sprg << "            sdata[tid] = sdata[tid + s];" << endl;
  sprg << "        }" << endl;
  sprg << "        __syncthreads();" << endl;
  sprg << "    }" << endl;
  sprg << "    if (tid == 0) g_odata[blockIdx.x] = sdata[0];" << endl;


  string prg = sprg.str();

#ifdef GPU_DEBUG_DEEP
  cout << "Cuda kernel code = " << endl << prg << endl << endl;
#endif

  static QDPJit::SharedLibEntry sharedLibEntry;
  if (!QDPJit::Instance().jitFixedGeom( strId , prg , cudaArgs.getDevPtr() , 
					  size , sharedLibEntry , threads, blocks , smemSize )) {
    QDP_error("globalMax() call to cuda jitter failed");
  }

}


}


#endif
