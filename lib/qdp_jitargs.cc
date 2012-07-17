#include "qdp.h"

namespace QDP {

  QDPJitArgs::QDPJitArgs(): size(0) {
    QDP_debug_deep("cuda kernel args: allocating host memory");
    if (!CUDAHostPoolAllocator::Instance().allocate( (void**)&arrayArgs , 
						     sizeof(UnionDevPtr) * 
						     DeviceParams::Instance().getMaxKernelArg() ))
      QDP_error_exit("jit args: could not allocate host memory");
    vecType.reserve(DeviceParams::Instance().getMaxKernelArg());
  }
  QDPJitArgs::~QDPJitArgs() {
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("cuda kernel args: freeing host memory");
#endif
    CUDAHostPoolAllocator::Instance().free( (void*)arrayArgs );
    QDPCache::Instance().signoff( myId );
  }

  string QDPJitArgs::getPtrName() const  { return "args"; }

  string QDPJitArgs::getCode(int i) const {
    if (i >= size)
      QDP_error_exit("jit args: get pointer name, out of range");
    ostringstream code;
    code << getPtrName() << "[" << i << "]" << QDPuni[vecType[i]];
    return code.str();
  }

  UnionDevPtr* QDPJitArgs::getDevPtr() {
    if (size==0) {
      QDP_info("Warning: jit args return NULL pointer");
      return NULL;
    }
    myId = QDPCache::Instance().registrateOwnHostMem( sizeof(UnionDevPtr) * size , (void*)arrayArgs );
    return (UnionDevPtr*)QDPCache::Instance().getDevicePtr( myId );
  }

  int QDPJitArgs::addPtr(void * devicePtr) const {
    if (size >= DeviceParams::Instance().getMaxKernelArg())
      QDP_error_exit("jit args: not enough memory. Increase jit argument memory");
    arrayArgs[size].ptr  = devicePtr;
    vecType.push_back(0);
    return size++;
  }

  int QDPJitArgs::addInt(int i) const {
    if (size >= DeviceParams::Instance().getMaxKernelArg())
      QDP_error_exit("jit args: not enough memory. Increase jit argument memory");
    arrayArgs[size].Int  = i;
    vecType.push_back(1);
    return size++;
  }

  int QDPJitArgs::addBool(bool b) const {
    if (size >= DeviceParams::Instance().getMaxKernelArg())
      QDP_error_exit("jit args: not enough memory. Increase jit argument memory");
    arrayArgs[size].Bool = b;
    vecType.push_back(2);
    return size++;
  }

  int QDPJitArgs::addIntPtr( int * intPtr) const {
    if (size >= DeviceParams::Instance().getMaxKernelArg())
      QDP_error_exit("jit args: not enough memory. Increase jit argument memory");
    arrayArgs[size].IntPtr  = intPtr;
    vecType.push_back(3);
    return size++;
  }

  int QDPJitArgs::addSize_t(size_t i) const {
    if (size >= DeviceParams::Instance().getMaxKernelArg())
      QDP_error_exit("jit args: not enough memory. Increase jit argument memory");
    arrayArgs[size].Size_t = i;
    vecType.push_back(4);
    return size++;
  }

}
