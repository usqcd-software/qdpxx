#include "qdp.h"

namespace QDP {

  void DeviceParams::setCC(int sm) {
    switch(sm) {
    case 20:
    case 21:
      smem = 48*1024;
      smem_default = 0;
      max_gridx  = max_gridy = max_gridz = 32768; // We need a power of 2 here!
      max_blockx = max_blocky = 1024;
      max_blockz = 64;
      break;
    case 30:
      smem = 48*1024;
      smem_default = 0;
      max_gridx  = max_gridy = max_gridz = 512 * 1024; // Its 2^31-1, but this value is large enough
      max_blockx = max_blocky = 1024;
      max_blockz = 64;
      break;
    default:
      QDP_error_exit("DeviceParams::setCC compute capability %d not known!",sm);
    }
    if (Layout::primaryNode())
      QDP_info("CUDA device compute capability set to sm_%d",sm);
  }

}
