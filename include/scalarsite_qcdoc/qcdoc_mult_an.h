#define _inline_qcdoc_mult_su3_an(aa,bb,cc) \
{ \
  register REAL B0R;  \
  register REAL B0I;  \
  register REAL B1R;  \
  register REAL B1I;  \
  register REAL B2R;  \
  register REAL B2I;  \
  register REAL A00R; \
  register REAL A00I; \
  register REAL A01R; \
  register REAL A01I; \
  register REAL A02R; \
  register REAL A02I; \
  register REAL A10R; \
  register REAL A10I; \
  register REAL A11R; \
  register REAL A11I; \
  register REAL A12R; \
  register REAL A12I; \
  register REAL A20R; \
  register REAL A20I; \
  register REAL A21R; \
  register REAL A21I; \
  register REAL A22R; \
  register REAL A22I; \
  register REAL C0R;  \
  register REAL C0I;  \
  register REAL C1R;  \
  register REAL C1I;  \
  register REAL C2R;  \
  register REAL C2I;  \
  register REAL t1;   \
  register REAL t2;   \
  \
  A00R = aa.elem(0,0).real().elem() ; \
  A00I = aa.elem(0,0).imag().elem() ; \
  A01R = aa.elem(0,1).real().elem() ; \
  A01I = aa.elem(0,1).imag().elem() ; \
  A02R = aa.elem(0,2).real().elem() ; \
  A02I = aa.elem(0,2).imag().elem() ; \
  A10R = aa.elem(1,0).real().elem() ; \
  A10I = aa.elem(1,0).imag().elem() ; \
  A11R = aa.elem(1,1).real().elem() ; \
  A11I = aa.elem(1,1).imag().elem() ; \
  A12R = aa.elem(1,2).real().elem() ; \
  A12I = aa.elem(1,2).imag().elem() ; \
  A20R = aa.elem(2,0).real().elem() ; \
  A20I = aa.elem(2,0).imag().elem() ; \
  A21R = aa.elem(2,1).real().elem() ; \
  A21I = aa.elem(2,1).imag().elem() ; \
  A22R = aa.elem(2,2).real().elem() ; \
  A22I = aa.elem(2,2).imag().elem() ; \
  \
  B0R = bb.elem(0,0).real().elem(); \
  B0I = bb.elem(0,0).imag().elem(); \
  B1R = bb.elem(0,1).real().elem(); \
  B1I = bb.elem(0,1).imag().elem(); \
  B2R = bb.elem(0,2).real().elem(); \
  B2I = bb.elem(0,2).imag().elem(); \
  \
  C0R = A00R*B0R; \
  C1R = A00R*B1R; \
  C2R = A00R*B2R; \
  C0I = -A00I*B0R; \
  B0R = bb.elem(1,0).real().elem(); \
  C1I = -A00I*B1R; \
  B1R = bb.elem(1,1).real().elem(); \
  C2I = -A00I*B2R; \
  B2R = bb.elem(1,2).real().elem(); \
  C0R = A00I*B0I + C0R; \
  C0I = A00R*B0I + C0I; \
  B0I = bb.elem(1,0).imag().elem(); \
  C1R = A00I*B1I + C1R; \
  C1I = A00R*B1I + C1I; \
  B1I = bb.elem(1,1).imag().elem(); \
  C2R = A00I*B2I + C2R; \
  C2I = A00R*B2I + C2I; \
   B2I = bb.elem(1,2).imag().elem(); \
  \
  C0R = A10R*B0R + C0R; \
  C1R = A10R*B1R + C1R; \
  C2R = A10R*B2R + C2R; \
  C0I = A10I*B0R - C0I ; \
  B0R = bb.elem(2,0).real().elem(); \
  C1I = A10I*B1R - C1I; \
  B1R = bb.elem(2,1).real().elem(); \
  C2I = A10I*B2R - C2I; \
  B2R = bb.elem(2,2).real().elem(); \
  C0R = A10I*B0I + C0R; \
  C0I = A10R*B0I + C0I; \
  B0I = bb.elem(2,0).imag().elem(); \
  C1R = A10I*B1I + C1R; \
  C1I = A10R*B1I + C1I; \
  B1I = bb.elem(2,1).imag().elem(); \
  C2R = A10I*B2I + C2R; \
  C2I = A10R*B2I + C2I; \
  B2I = bb.elem(2,2).imag().elem(); \
  \
  C0R = A20R*B0R + C0R; \
  C1R = A20R*B1R + C1R; \
  C2R = A20R*B2R + C2R; \
  C0I = A20I*B0R - C0I; \
  B0R = bb.elem(0,0).real().elem(); \
  C1I = A20I*B1R - C1I; \
  B1R = bb.elem(0,1).real().elem(); \
  C2I = A20I*B2R - C2I; \
  B2R = bb.elem(0,2).real().elem(); \
  C0R = A20I*B0I + C0R; \
  cc.elem(0,0).real().elem() = C0R; \
  C0I = A20R*B0I + C0I; \
  B0I = bb.elem(0,0).imag().elem(); \
  cc.elem(0,0).imag().elem() = C0I; \
  C1R = A20I*B1I + C1R; \
  cc.elem(0,1).real().elem() = C1R; \
  C1I = A20R*B1I + C1I; \
  B1I = bb.elem(0,1).imag().elem(); \
  cc.elem(0,1).imag().elem() = C1I; \
  C2R = A20I*B2I + C2R; \
  cc.elem(0,2).real().elem() = C2R; \
  C2I = A20R*B2I + C2I; \
  B2I = bb.elem(0,2).imag().elem(); \
  cc.elem(0,2).imag().elem() = C2I; \
  \
  C0R = A01R*B0R; \
  C1R = A01R*B1R; \
  C2R = A01R*B2R; \
  C0I = -A01I*B0R; \
  B0R = bb.elem(1,0).real().elem(); \
  C1I = -A01I*B1R; \
  B1R = bb.elem(1,1).real().elem(); \
  C2I = -A01I*B2R; \
  B2R = bb.elem(1,2).real().elem(); \
  C0R = A01I*B0I + C0R; \
  C0I = A01R*B0I + C0I; \
  B0I = bb.elem(1,0).imag().elem(); \
  C1R = A01I*B1I + C1R; \
  C1I = A01R*B1I + C1I; \
  B1I = bb.elem(1,1).imag().elem(); \
  C2R = A01I*B2I + C2R; \
  C2I = A01R*B2I + C2I; \
  B2I = bb.elem(1,2).imag().elem(); \
  \
  C0R = A11R*B0R + C0R; \
  C1R = A11R*B1R + C1R; \
  C2R = A11R*B2R + C2R; \
  C0I = A11I*B0R - C0I; \
  B0R = bb.elem(2,0).real().elem(); \
  C1I = A11I*B1R - C1I; \
  B1R = bb.elem(2,1).real().elem(); \
  C2I = A11I*B2R - C2I; \
  B2R = bb.elem(2,2).real().elem(); \
  C0R = A11I*B0I + C0R; \
  C0I = A11R*B0I + C0I; \
  B0I = bb.elem(2,0).imag().elem(); \
  C1R = A11I*B1I + C1R; \
  C1I = A11R*B1I + C1I; \
  B1I = bb.elem(2,1).imag().elem(); \
  C2R = A11I*B2I + C2R; \
  C2I = A11R*B2I + C2I; \
  B2I = bb.elem(2,2).imag().elem(); \
  C0R = A21R*B0R + C0R; \
  C1R = A21R*B1R + C1R; \
  C2R = A21R*B2R + C2R; \
  C0I = A21I*B0R - C0I; \
  B0R = bb.elem(0,0).real().elem(); \
  C1I = A21I*B1R - C1I; \
  B1R = bb.elem(0,1).real().elem(); \
  C2I = A21I*B2R - C2I; \
  B2R = bb.elem(0,2).real().elem(); \
  C0R = A21I*B0I + C0R; \
  cc.elem(1,0).real().elem() = C0R; \
  C0I = A21R*B0I + C0I; \
  B0I = bb.elem(0,0).imag().elem(); \
  cc.elem(1,0).imag().elem() = C0I; \
  C1R = A21I*B1I + C1R; \
  cc.elem(1,1).real().elem() = C1R; \
  C1I = A21R*B1I + C1I; \
  cc.elem(1,1).imag().elem() = C1I; \
  B1I = bb.elem(0,1).imag().elem(); \
  C2R = A21I*B2I + C2R; \
  cc.elem(1,2).real().elem() = C2R; \
  C2I = A21R*B2I + C2I; \
  B2I = bb.elem(0,2).imag().elem(); \
  cc.elem(1,2).imag().elem() = C2I; \
\
  C0R = A02R*B0R; \
  C1R = A02R*B1R; \
  C2R = A02R*B2R; \
  C0I = -A02I*B0R; \
  B0R = bb.elem(1,0).real().elem(); \
  C1I = -A02I*B1R; \
  B1R = bb.elem(1,1).real().elem(); \
  C2I = -A02I*B2R; \
  B2R = bb.elem(1,2).real().elem(); \
  C0R = A02I*B0I + C0R; \
  C0I = A02R*B0I + C0I; \
  B0I = bb.elem(1,0).imag().elem(); \
  C1R = A02I*B1I + C1R; \
  C1I = A02R*B1I + C1I; \
  B1I = bb.elem(1,1).imag().elem(); \
  C2R = A02I*B2I + C2R; \
  C2I = A02R*B2I + C2I; \
  B2I = bb.elem(1,2).imag().elem(); \
  \
  C0R = A12R*B0R + C0R; \
  C1R = A12R*B1R + C1R; \
  C2R = A12R*B2R + C2R; \
  C0I = A12I*B0R - C0I; \
  B0R = bb.elem(2,0).real().elem(); \
  C1I = A12I*B1R - C1I; \
  B1R = bb.elem(2,1).real().elem(); \
  C2I = A12I*B2R - C2I; \
  B2R = bb.elem(2,2).real().elem(); \
  C0R = A12I*B0I + C0R; \
  C0I = A12R*B0I + C0I; \
  B0I = bb.elem(2,0).imag().elem(); \
  C1R = A12I*B1I + C1R; \
  C1I = A12R*B1I + C1I; \
  B1I = bb.elem(2,1).imag().elem(); \
  C2R = A12I*B2I + C2R; \
  C2I = A12R*B2I + C2I; \
  B2I = bb.elem(2,2).imag().elem(); \
  \
  C0R = A22R*B0R + C0R; \
  C1R = A22R*B1R + C1R; \
  C2R = A22R*B2R + C2R; \
  C0I = A22I*B0R - C0I; \
  C1I = A22I*B1R - C1I; \
  C2I = A22I*B2R - C2I; \
  C0R = A22I*B0I + C0R; \
  cc.elem(2,0).real().elem() = C0R; \
  C0I = A22R*B0I + C0I; \
  cc.elem(2,0).imag().elem() = C0I; \
  C1R = A22I*B1I + C1R; \
  cc.elem(2,1).real().elem() = C1R; \
  C1I = A22R*B1I + C1I; \
  cc.elem(2,1).imag().elem() = C1I; \
  C2R = A22I*B2I + C2R; \
  cc.elem(2,2).real().elem() = C2R; \
  C2I = A22R*B2I + C2I; \
  cc.elem(2,2).imag().elem() = C2I; \
}
