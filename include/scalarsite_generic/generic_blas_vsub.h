// $Id: generic_blas_vsub.h,v 1.3 2009-09-15 20:48:42 bjoo Exp $

/*! @file
 *  @brief Generic Scalar VSUB routine
 *
 */

#ifndef QDP_GENERIC_BLAS_VSUB
#define QDP_GENERIC_BLAS_VSUB

namespace QDP {

// (Vector) Out = (Vector) In1 - (Vector) In2
inline
void vsub(REAL *Out, REAL *In1, REAL *In2, int n_3vec)
{
  register double in10r;
  register double in10i;
  register double in11r;
  register double in11i;
  register double in12r;
  register double in12i;

  register double in20r;
  register double in20i;
  register double in21r;
  register double in21i;
  register double in22r;
  register double in22i;

  register double out0r;
  register double out0i;
  register double out1r;
  register double out1i;
  register double out2r;
  register double out2i;

  register int counter =0;
  register int in1ptr =0;
  register int in2ptr =0;
  register int outptr =0;

  if( n_3vec > 0 ) {
    int len=4*n_3vec;

    in10r = (double)In1[in1ptr++];
    in20r = (double)In2[in2ptr++];
    in10i = (double)In1[in1ptr++];
    in20i = (double)In2[in2ptr++];
    for(counter = 0; counter < len-1; counter++) { 
      out0r = in10r - in20r;
      Out[outptr++] = (REAL)out0r;

      in11r = (double)In1[in1ptr++];
      in21r = (double)In2[in2ptr++];
      out0i = in10i - in20i;
      Out[outptr++] = (REAL)out0i;

      in11i = (double)In1[in1ptr++];
      in21i = (double)In2[in2ptr++];
      out1r = in11r - in21r;
      Out[outptr++] = (REAL)out1r;

      in12r = (double)In1[in1ptr++];
      in22r = (double)In2[in2ptr++];
      out1i = in11i - in21i;
      Out[outptr++] = (REAL)out1i;

      in12i = (double)In1[in1ptr++];
      in22i = (double)In2[in2ptr++];
      out2r = in12r - in22r;
      Out[outptr++] = (REAL)out2r;

      in10r = (double)In1[in1ptr++];
      in20r = (double)In2[in2ptr++];     
      out2i = in12i - in22i;
      Out[outptr++] = (REAL)out2i;

      in10i = (double)In1[in1ptr++];
      in20i = (double)In2[in2ptr++];
    }
    out0r = in10r - in20r;
    Out[outptr++] = (REAL)out0r;

    in11r = (double)In1[in1ptr++];
    in21r = (double)In2[in2ptr++];
    out0i = in10i - in20i;
    Out[outptr++] = (REAL)out0i;

    in11i = (double)In1[in1ptr++];
    in21i = (double)In2[in2ptr++];
    out1r = in11r - in21r;
    Out[outptr++] = (REAL)out1r;

    in12r = (double)In1[in1ptr++];
    in22r = (double)In2[in2ptr++];
    out1i = in11i - in21i;
    Out[outptr++] = (REAL)out1i;

    in12i = (double)In1[in1ptr++];
    in22i = (double)In2[in2ptr++];
    out2r = in12r - in22r;
    Out[outptr++] = (REAL)out2r;
    out2i = in12i - in22i;
    Out[outptr++] = (REAL)out2i;
  }
}


} // namespace QDP;

#endif // guard
