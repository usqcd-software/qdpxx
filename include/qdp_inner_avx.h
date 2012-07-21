// -*- C++ -*-

/*! \file
 * \brief Inner grid for AVX architecure
 */

#pragma once

#include <immintrin.h>

namespace QDP {

#if 1
  template<>
  inline void assign_ilattice(float* l, const ILattice<float, 8>& r)
  {
    // copy to YMM register
    float* rf = (float *)&(r.elem(0));

    // copy and store
    _mm256_store_ps(l, _mm256_load_ps(rf));
  }

  template<>
  inline void add_ilattice(ILattice<float, 8>& dest, 
			   const ILattice<float, 8>& rhs)
  {
#warning "Using AVX for float + float "


    // copy to YMM register
    float* rf = (float *)&(rhs.elem(0));
    float* df = (float *)&(dest.elem(0));

    _mm256_store_ps(df,_mm256_add_ps(_mm256_load_ps(df), _mm256_load_ps(rf)));
  }

  // ILattice * ILattice
  template<>
  inline typename BinaryReturn<ILattice<float,8>, ILattice<float,8>, OpMultiply>::Type_t
  operator*(const ILattice<float,8>& l, const ILattice<float,8>& r)
  {
    typename BinaryReturn<ILattice<float, 8>, ILattice<float,8>, OpMultiply>::Type_t  d;
    // copy to YMM register
    float* lf = (float *)&(l.elem(0));
    float* rf = (float *)&(r.elem(0));
    float* df = (float *)&(d.elem(0));

    if ((unsigned long)lf & 31 != 0 || (unsigned long)rf & 31 != 0 || (unsigned long)df & 31 != 0)
      fprintf (stderr, "memory address lf = %p rf = %p df = %p\n",
	       lf, rf, df);

    _mm256_store_ps(df,_mm256_mul_ps(_mm256_load_ps(lf), _mm256_load_ps(rf)));

    return d;
  }


// ILattice + ILattice
  template<>
  inline typename BinaryReturn<ILattice<float,8>, ILattice<float,8>, OpAdd>::Type_t
  operator+(const ILattice<float,8>& l, const ILattice<float,8>& r)
  {
    typename BinaryReturn<ILattice<float,8>, ILattice<float,8>, OpAdd>::Type_t  d;
    // copy to YMM register
    float* lf = (float *)&(l.elem(0));
    float* rf = (float *)&(r.elem(0));
    float* df = (float *)&(d.elem(0));

    if ((unsigned long)lf & 31 != 0 || (unsigned long)rf & 31 != 0 || (unsigned long)df & 31 != 0)
      fprintf (stderr, "memory address lf = %p rf = %p df = %p\n",
	       lf, rf, df);

    _mm256_store_ps(df,_mm256_add_ps(_mm256_load_ps(lf), _mm256_load_ps(rf)));

    return d;
  }


  // ILattice - ILattice
  template<>
  inline typename BinaryReturn<ILattice<float,8>, ILattice<float,8>, OpSubtract>::Type_t
  operator-(const ILattice<float,8>& l, const ILattice<float,8>& r)
  {
    typename BinaryReturn<ILattice<float,8>, ILattice<float,8>, OpSubtract>::Type_t  d;

    // copy to YMM register
    float* lf = (float *)&(l.elem(0));
    float* rf = (float *)&(r.elem(0));
    float* df = (float *)&(d.elem(0));

    if ((unsigned long)lf & 31 != 0 || (unsigned long)rf & 31 != 0 || (unsigned long)df & 31 != 0)
      fprintf (stderr, "memory address lf = %p rf = %p df = %p\n",
	       lf, rf, df);

    _mm256_store_ps(df,_mm256_sub_ps(_mm256_load_ps(lf), _mm256_load_ps(rf)));
    return d;
  }
#endif
}
