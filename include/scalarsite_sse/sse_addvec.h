#define _inline_sse_add_su3_vector(aa,bb,cc) \
{ \
__asm__ __volatile__ ("movups %0, %%xmm0 \n\t" \
                      "movlps %1, %%xmm1 \n\t" \
                      "shufps $0x44, %%xmm1, %%xmm1 \n\t" \
                      "movups %2, %%xmm2 \n\t" \
                      "movlps %3, %%xmm3 \n\t" \
                      "shufps $0x44, %%xmm3, %%xmm3 \n\t" \
                      "addps %%xmm2, %%xmm0 \n\t" \
                      "addps %%xmm3, %%xmm1 \n\t" \
                      : \
                      : \
                      "m" ((aa).elem(0)), \
                      "m" ((aa).elem(2)), \
                      "m" ((bb).elem(0)), \
                      "m" ((bb).elem(2))); \
__asm__ __volatile__ ("movups %%xmm0, %0 \n\t" \
                      "movlps %%xmm1, %1 \n\t" \
                      : \
                      "=m" ((cc).elem(0)), \
                      "=m" ((cc).elem(2))); \
}
