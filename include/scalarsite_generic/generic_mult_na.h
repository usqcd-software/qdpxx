#ifdef __MIC
#warning "using _inline_generic_mult_su3_na for MIC"

#define _inline_generic_mult_su3_na(aa,bb,cc) \
{\
	mic_MatMultAdj(cc,aa,bb);	\
}

#else

#define _inline_generic_mult_su3_na(aa,bb,cc) \
{ \
  cc.elem(0,0) = multiplyAdj(aa.elem(0,0),bb.elem(0,0))  \
               + multiplyAdj(aa.elem(0,1),bb.elem(0,1))  \
               + multiplyAdj(aa.elem(0,2),bb.elem(0,2)); \
  cc.elem(1,0) = multiplyAdj(aa.elem(1,0),bb.elem(0,0))  \
               + multiplyAdj(aa.elem(1,1),bb.elem(0,1))  \
               + multiplyAdj(aa.elem(1,2),bb.elem(0,2)); \
  cc.elem(2,0) = multiplyAdj(aa.elem(2,0),bb.elem(0,0))  \
               + multiplyAdj(aa.elem(2,1),bb.elem(0,1))  \
               + multiplyAdj(aa.elem(2,2),bb.elem(0,2)); \
  \
  cc.elem(0,1) = multiplyAdj(aa.elem(0,0),bb.elem(1,0))  \
               + multiplyAdj(aa.elem(0,1),bb.elem(1,1))  \
               + multiplyAdj(aa.elem(0,2),bb.elem(1,2)); \
  cc.elem(1,1) = multiplyAdj(aa.elem(1,0),bb.elem(1,0))  \
               + multiplyAdj(aa.elem(1,1),bb.elem(1,1))  \
               + multiplyAdj(aa.elem(1,2),bb.elem(1,2)); \
  cc.elem(2,1) = multiplyAdj(aa.elem(2,0),bb.elem(1,0))  \
               + multiplyAdj(aa.elem(2,1),bb.elem(1,1))  \
               + multiplyAdj(aa.elem(2,2),bb.elem(1,2)); \
  \
  cc.elem(0,2) = multiplyAdj(aa.elem(0,0),bb.elem(2,0))  \
               + multiplyAdj(aa.elem(0,1),bb.elem(2,1))  \
               + multiplyAdj(aa.elem(0,2),bb.elem(2,2)); \
  cc.elem(1,2) = multiplyAdj(aa.elem(1,0),bb.elem(2,0))  \
               + multiplyAdj(aa.elem(1,1),bb.elem(2,1))  \
               + multiplyAdj(aa.elem(1,2),bb.elem(2,2)); \
  cc.elem(2,2) = multiplyAdj(aa.elem(2,0),bb.elem(2,0))  \
               + multiplyAdj(aa.elem(2,1),bb.elem(2,1))  \
               + multiplyAdj(aa.elem(2,2),bb.elem(2,2)); \
}

#endif
