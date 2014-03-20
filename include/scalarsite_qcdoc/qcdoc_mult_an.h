#define _inline_qcdoc_mult_su3_an(aa,bb,cc) \
{ \
   REAL B0R;  \
   REAL B0I;  \
   REAL B1R;  \
   REAL B1I;  \
   REAL B2R;  \
   REAL B2I;  \
   REAL A00R; \
   REAL A00I; \
   REAL A01R; \
   REAL A01I; \
   REAL A02R; \
   REAL A02I; \
   REAL A10R; \
   REAL A10I; \
   REAL A11R; \
   REAL A11I; \
   REAL A12R; \
   REAL A12I; \
   REAL A20R; \
   REAL A20I; \
   REAL A21R; \
   REAL A21I; \
   REAL A22R; \
   REAL A22I; \
   REAL C0R;  \
   REAL C0I;  \
   REAL C1R;  \
   REAL C1I;  \
   REAL C2R;  \
   REAL C2I;  \
   REAL t1;   \
   REAL t2;   \
  \
  A00R = aa.elem(0,0).real()  ; \
  A00I = aa.elem(0,0).imag()  ; \
  A01R = aa.elem(0,1).real()  ; \
  A01I = aa.elem(0,1).imag()  ; \
  A02R = aa.elem(0,2).real()  ; \
  A02I = aa.elem(0,2).imag()  ; \
  A10R = aa.elem(1,0).real()  ; \
  A10I = aa.elem(1,0).imag()  ; \
  A11R = aa.elem(1,1).real()  ; \
  A11I = aa.elem(1,1).imag()  ; \
  A12R = aa.elem(1,2).real()  ; \
  A12I = aa.elem(1,2).imag()  ; \
  A20R = aa.elem(2,0).real()  ; \
  A20I = aa.elem(2,0).imag()  ; \
  A21R = aa.elem(2,1).real()  ; \
  A21I = aa.elem(2,1).imag()  ; \
  A22R = aa.elem(2,2).real()  ; \
  A22I = aa.elem(2,2).imag()  ; \
  \
  B0R = bb.elem(0,0).real() ; \
  B0I = bb.elem(0,0).imag() ; \
  B1R = bb.elem(0,1).real() ; \
  B1I = bb.elem(0,1).imag() ; \
  B2R = bb.elem(0,2).real() ; \
  B2I = bb.elem(0,2).imag() ; \
  \
  C0R = A00R*B0R; \
  C1R = A00R*B1R; \
  C2R = A00R*B2R; \
  C0I = -A00I*B0R; \
  B0R = bb.elem(1,0).real() ; \
  C1I = -A00I*B1R; \
  B1R = bb.elem(1,1).real() ; \
  C2I = -A00I*B2R; \
  B2R = bb.elem(1,2).real() ; \
  C0R = A00I*B0I + C0R; \
  C0I = A00R*B0I + C0I; \
  B0I = bb.elem(1,0).imag() ; \
  C1R = A00I*B1I + C1R; \
  C1I = A00R*B1I + C1I; \
  B1I = bb.elem(1,1).imag() ; \
  C2R = A00I*B2I + C2R; \
  C2I = A00R*B2I + C2I; \
   B2I = bb.elem(1,2).imag() ; \
  \
  C0R = A10R*B0R + C0R; \
  C1R = A10R*B1R + C1R; \
  C2R = A10R*B2R + C2R; \
  C0I = A10I*B0R - C0I ; \
  B0R = bb.elem(2,0).real() ; \
  C1I = A10I*B1R - C1I; \
  B1R = bb.elem(2,1).real() ; \
  C2I = A10I*B2R - C2I; \
  B2R = bb.elem(2,2).real() ; \
  C0R = A10I*B0I + C0R; \
  C0I = A10R*B0I + C0I; \
  B0I = bb.elem(2,0).imag() ; \
  C1R = A10I*B1I + C1R; \
  C1I = A10R*B1I + C1I; \
  B1I = bb.elem(2,1).imag() ; \
  C2R = A10I*B2I + C2R; \
  C2I = A10R*B2I + C2I; \
  B2I = bb.elem(2,2).imag() ; \
  \
  C0R = A20R*B0R + C0R; \
  C1R = A20R*B1R + C1R; \
  C2R = A20R*B2R + C2R; \
  C0I = A20I*B0R - C0I; \
  B0R = bb.elem(0,0).real() ; \
  C1I = A20I*B1R - C1I; \
  B1R = bb.elem(0,1).real() ; \
  C2I = A20I*B2R - C2I; \
  B2R = bb.elem(0,2).real() ; \
  C0R = A20I*B0I + C0R; \
  cc.elem(0,0).real()  = C0R; \
  C0I = A20R*B0I + C0I; \
  B0I = bb.elem(0,0).imag() ; \
  cc.elem(0,0).imag()  = C0I; \
  C1R = A20I*B1I + C1R; \
  cc.elem(0,1).real()  = C1R; \
  C1I = A20R*B1I + C1I; \
  B1I = bb.elem(0,1).imag() ; \
  cc.elem(0,1).imag()  = C1I; \
  C2R = A20I*B2I + C2R; \
  cc.elem(0,2).real()  = C2R; \
  C2I = A20R*B2I + C2I; \
  B2I = bb.elem(0,2).imag() ; \
  cc.elem(0,2).imag()  = C2I; \
  \
  C0R = A01R*B0R; \
  C1R = A01R*B1R; \
  C2R = A01R*B2R; \
  C0I = -A01I*B0R; \
  B0R = bb.elem(1,0).real() ; \
  C1I = -A01I*B1R; \
  B1R = bb.elem(1,1).real() ; \
  C2I = -A01I*B2R; \
  B2R = bb.elem(1,2).real() ; \
  C0R = A01I*B0I + C0R; \
  C0I = A01R*B0I + C0I; \
  B0I = bb.elem(1,0).imag() ; \
  C1R = A01I*B1I + C1R; \
  C1I = A01R*B1I + C1I; \
  B1I = bb.elem(1,1).imag() ; \
  C2R = A01I*B2I + C2R; \
  C2I = A01R*B2I + C2I; \
  B2I = bb.elem(1,2).imag() ; \
  \
  C0R = A11R*B0R + C0R; \
  C1R = A11R*B1R + C1R; \
  C2R = A11R*B2R + C2R; \
  C0I = A11I*B0R - C0I; \
  B0R = bb.elem(2,0).real() ; \
  C1I = A11I*B1R - C1I; \
  B1R = bb.elem(2,1).real() ; \
  C2I = A11I*B2R - C2I; \
  B2R = bb.elem(2,2).real() ; \
  C0R = A11I*B0I + C0R; \
  C0I = A11R*B0I + C0I; \
  B0I = bb.elem(2,0).imag() ; \
  C1R = A11I*B1I + C1R; \
  C1I = A11R*B1I + C1I; \
  B1I = bb.elem(2,1).imag() ; \
  C2R = A11I*B2I + C2R; \
  C2I = A11R*B2I + C2I; \
  B2I = bb.elem(2,2).imag() ; \
  C0R = A21R*B0R + C0R; \
  C1R = A21R*B1R + C1R; \
  C2R = A21R*B2R + C2R; \
  C0I = A21I*B0R - C0I; \
  B0R = bb.elem(0,0).real() ; \
  C1I = A21I*B1R - C1I; \
  B1R = bb.elem(0,1).real() ; \
  C2I = A21I*B2R - C2I; \
  B2R = bb.elem(0,2).real() ; \
  C0R = A21I*B0I + C0R; \
  cc.elem(1,0).real()  = C0R; \
  C0I = A21R*B0I + C0I; \
  B0I = bb.elem(0,0).imag() ; \
  cc.elem(1,0).imag()  = C0I; \
  C1R = A21I*B1I + C1R; \
  cc.elem(1,1).real()  = C1R; \
  C1I = A21R*B1I + C1I; \
  cc.elem(1,1).imag()  = C1I; \
  B1I = bb.elem(0,1).imag() ; \
  C2R = A21I*B2I + C2R; \
  cc.elem(1,2).real()  = C2R; \
  C2I = A21R*B2I + C2I; \
  B2I = bb.elem(0,2).imag() ; \
  cc.elem(1,2).imag()  = C2I; \
\
  C0R = A02R*B0R; \
  C1R = A02R*B1R; \
  C2R = A02R*B2R; \
  C0I = -A02I*B0R; \
  B0R = bb.elem(1,0).real() ; \
  C1I = -A02I*B1R; \
  B1R = bb.elem(1,1).real() ; \
  C2I = -A02I*B2R; \
  B2R = bb.elem(1,2).real() ; \
  C0R = A02I*B0I + C0R; \
  C0I = A02R*B0I + C0I; \
  B0I = bb.elem(1,0).imag() ; \
  C1R = A02I*B1I + C1R; \
  C1I = A02R*B1I + C1I; \
  B1I = bb.elem(1,1).imag() ; \
  C2R = A02I*B2I + C2R; \
  C2I = A02R*B2I + C2I; \
  B2I = bb.elem(1,2).imag() ; \
  \
  C0R = A12R*B0R + C0R; \
  C1R = A12R*B1R + C1R; \
  C2R = A12R*B2R + C2R; \
  C0I = A12I*B0R - C0I; \
  B0R = bb.elem(2,0).real() ; \
  C1I = A12I*B1R - C1I; \
  B1R = bb.elem(2,1).real() ; \
  C2I = A12I*B2R - C2I; \
  B2R = bb.elem(2,2).real() ; \
  C0R = A12I*B0I + C0R; \
  C0I = A12R*B0I + C0I; \
  B0I = bb.elem(2,0).imag() ; \
  C1R = A12I*B1I + C1R; \
  C1I = A12R*B1I + C1I; \
  B1I = bb.elem(2,1).imag() ; \
  C2R = A12I*B2I + C2R; \
  C2I = A12R*B2I + C2I; \
  B2I = bb.elem(2,2).imag() ; \
  \
  C0R = A22R*B0R + C0R; \
  C1R = A22R*B1R + C1R; \
  C2R = A22R*B2R + C2R; \
  C0I = A22I*B0R - C0I; \
  C1I = A22I*B1R - C1I; \
  C2I = A22I*B2R - C2I; \
  C0R = A22I*B0I + C0R; \
  cc.elem(2,0).real()  = C0R; \
  C0I = A22R*B0I + C0I; \
  cc.elem(2,0).imag()  = C0I; \
  C1R = A22I*B1I + C1R; \
  cc.elem(2,1).real()  = C1R; \
  C1I = A22R*B1I + C1I; \
  cc.elem(2,1).imag()  = C1I; \
  C2R = A22I*B2I + C2R; \
  cc.elem(2,2).real()  = C2R; \
  C2I = A22R*B2I + C2I; \
  cc.elem(2,2).imag()  = C2I; \
}
