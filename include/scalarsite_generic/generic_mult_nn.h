#define _inline_generic_mult_su3_nn(aa,bb,cc) \
{ \
  cc.elem(0,0) = aa.elem(0,0)*bb.elem(0,0) + aa.elem(0,1)*bb.elem(1,0) + aa.elem(0,2)*bb.elem(2,0); \
  cc.elem(1,0) = aa.elem(1,0)*bb.elem(0,0) + aa.elem(1,1)*bb.elem(1,0) + aa.elem(1,2)*bb.elem(2,0); \
  cc.elem(2,0) = aa.elem(2,0)*bb.elem(0,0) + aa.elem(2,1)*bb.elem(1,0) + aa.elem(2,2)*bb.elem(2,0); \
  \
  cc.elem(0,1) = aa.elem(0,0)*bb.elem(0,1) + aa.elem(0,1)*bb.elem(1,1) + aa.elem(0,2)*bb.elem(2,1); \
  cc.elem(1,1) = aa.elem(1,0)*bb.elem(0,1) + aa.elem(1,1)*bb.elem(1,1) + aa.elem(1,2)*bb.elem(2,1); \
  cc.elem(2,1) = aa.elem(2,0)*bb.elem(0,1) + aa.elem(2,1)*bb.elem(1,1) + aa.elem(2,2)*bb.elem(2,1); \
  \
  cc.elem(0,2) = aa.elem(0,0)*bb.elem(0,2) + aa.elem(0,1)*bb.elem(1,2) + aa.elem(0,2)*bb.elem(2,2); \
  cc.elem(1,2) = aa.elem(1,0)*bb.elem(0,2) + aa.elem(1,1)*bb.elem(1,2) + aa.elem(1,2)*bb.elem(2,2); \
  cc.elem(2,2) = aa.elem(2,0)*bb.elem(0,2) + aa.elem(2,1)*bb.elem(1,2) + aa.elem(2,2)*bb.elem(2,2); \
}
