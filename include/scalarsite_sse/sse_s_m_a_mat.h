#define _inline_sse_scalar_mult_add_su3_matrix(aa,bb,cc,dd) \
{ \
__asm__ __volatile__ ("movss %0, %%xmm4 \n\t" \
                      "shufps $0x00, %%xmm4, %%xmm4 \n\t" \
                      "movups %1, %%xmm0 \n\t" \
                      "movups %2, %%xmm1 \n\t" \
                      "mulps %%xmm4, %%xmm1 \n\t" \
                      "addps %%xmm1, %%xmm0 \n\t" \
                      : \
                      : \
                      "m" ((cc)), \
                      "m" ((aa).elem(0,0)), \
                      "m" ((bb).elem(0,0))); \
__asm__ __volatile__ ("movups %%xmm0, %0 \n\t" \
                      : \
                      "=m" ((dd).elem(0,0))); \
__asm__ __volatile__ ("movups %0, %%xmm0 \n\t" \
                      "movups %1, %%xmm1 \n\t" \
                      "mulps %%xmm4, %%xmm1 \n\t" \
                      "addps %%xmm1, %%xmm0 \n\t" \
                      : \
                      : \
                      "m" ((aa).elem(0,2)), \
                      "m" ((bb).elem(0,2))); \
__asm__ __volatile__ ("movups %%xmm0, %0 \n\t" \
                      : \
                      "=m" ((dd).elem(0,2))); \
__asm__ __volatile__ ("movups %0, %%xmm0 \n\t" \
                      "movups %1, %%xmm1 \n\t" \
                      "mulps %%xmm4, %%xmm1 \n\t" \
                      "addps %%xmm1, %%xmm0 \n\t" \
                      : \
                      : \
                      "m" ((aa).elem(1,1)), \
                      "m" ((bb).elem(1,1))); \
__asm__ __volatile__ ("movups %%xmm0, %0 \n\t" \
                      : \
                      "=m" ((dd).elem(1,1))); \
__asm__ __volatile__ ("movups %0, %%xmm0 \n\t" \
                      "movups %1, %%xmm1 \n\t" \
                      "mulps %%xmm4, %%xmm1 \n\t" \
                      "addps %%xmm1, %%xmm0 \n\t" \
                      : \
                      : \
                      "m" ((aa).elem(2,0)), \
                      "m" ((bb).elem(2,0))); \
__asm__ __volatile__ ("movups %%xmm0, %0 \n\t" \
                      : \
                      "=m" ((dd).elem(2,0))); \
__asm__ __volatile__ ("movlps %0, %%xmm0 \n\t" \
                      "movlps %1, %%xmm1 \n\t" \
                      "mulps %%xmm4, %%xmm1 \n\t" \
                      "addps %%xmm1, %%xmm0 \n\t" \
                      : \
                      : \
                      "m" ((aa).elem(2,2)), \
                      "m" ((bb).elem(2,2))); \
__asm__ __volatile__ ("movlps %%xmm0, %0 \n\t" \
                      : \
                      "=m" ((dd).elem(2,2))); \
}
