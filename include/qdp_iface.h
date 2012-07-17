#ifndef QDP_IFACE_H
#define QDP_IFACE_H

union UnionDevPtr {
  void * ptr;
  int    Int;
  bool   Bool;
  int*   IntPtr;
  size_t Size_t;
};

struct kernel_geom_t {
  int threads_per_block;
  int Nblock_x;
  int Nblock_y;
  int smemSize;
};


#endif
