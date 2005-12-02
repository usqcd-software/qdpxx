#include "qdp.h"

namespace QDP {

  // Fully specialised strings -- can live in the .cc file - defined only once
  
  // This is for the QIO type strings
  template<>
  char* QIOStringTraits< multi1d<LatticeColorMatrixF3> >::tname = "QDP_F3_ColorMatrix";

  template<>
  char* QIOStringTraits< multi1d<LatticeColorMatrixD3> >::tname = "QDP_D3_ColorMatrix";

  // This is for the QIO precision strings
  template<>
  char* QIOStringTraits<float>::tprec = "F";
  
  template<>
  char* QIOStringTraits<double>::tprec = "D";
  
  template<>
  char* QIOStringTraits<int>::tprec = "I";

}
