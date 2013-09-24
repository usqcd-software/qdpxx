// -*- C++ -*-

/*! \file
 * \brief Inner grid for AVX architecure
 */

#ifndef QDP_INNER_AVX_H
#define QDP_INNER_AVX_H

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

  // IScalar * ILattice
  template<>
  inline typename BinaryReturn<IScalar<float>, ILattice<float,8>, OpMultiply>::Type_t
  operator*(const IScalar<float>& l, const ILattice<float,8>& r)
  {
#warning " Using AVX for scalar float * float"
    typename BinaryReturn<IScalar<float>, ILattice<float,8>, OpMultiply>::Type_t  d;

    // Load the float and broadcast it.
    __m256 lvec = _mm256_broadcast_ss((const float *)&(l.elem()));

    float* rf = (float *)&(r.elem(0));
    float* df = (float *)&(d.elem(0));

    _mm256_store_ps(df,_mm256_mul_ps(lvec, _mm256_load_ps(rf)));

    return d;
  }

  //  ILattice * IScalar
  template<>
  inline typename BinaryReturn<ILattice<float,8>, IScalar<float>, OpMultiply>::Type_t
  operator*(const ILattice<float,8>& l, const IScalar<float>& r)
  {
#warning " Using AVX for scalar float * float"
    typename BinaryReturn<ILattice<float,8>, IScalar<float>, OpMultiply>::Type_t  d;

    // Load the float and broadcast it.
    __m256 rvec = _mm256_broadcast_ss((const float *)&(r.elem()));

    float* lf = (float *)&(l.elem(0));
    float* df = (float *)&(d.elem(0));

    _mm256_store_ps(df,_mm256_mul_ps(_mm256_load_ps(lf),rvec));

    return d;
  }


  // ILattice * ILattice
  template<>
  inline typename BinaryReturn<ILattice<float,8>, ILattice<float,8>, OpMultiply>::Type_t
  operator*(const ILattice<float,8>& l, const ILattice<float,8>& r)
  {
#warning " Using AVX for float * float"
    typename BinaryReturn<ILattice<float, 8>, ILattice<float,8>, OpMultiply>::Type_t  d;
    // copy to YMM register
    float* lf = (float *)&(l.elem(0));
    float* rf = (float *)&(r.elem(0));
    float* df = (float *)&(d.elem(0));


    _mm256_store_ps(df,_mm256_mul_ps(_mm256_load_ps(lf), _mm256_load_ps(rf)));

    return d;
  }


// ILattice + ILattice
  template<>
  inline typename BinaryReturn<ILattice<float,8>, ILattice<float,8>, OpAdd>::Type_t
  operator+(const ILattice<float,8>& l, const ILattice<float,8>& r)
  {
#warning " Using AVX for float + float"
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
#warning " Using AVX for float - float"
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

  /* Double precision */
#if 1
  template<>
  inline void assign_ilattice(double* l, const ILattice<double, 4>& r)
  {
    // copy to YMM register
    double* rf = (double *)&(r.elem(0));

    // copy and store
    _mm256_store_pd(l, _mm256_load_pd(rf));
  }

  template<>
  inline void add_ilattice(ILattice<double, 4>& dest, 
			   const ILattice<double, 4>& rhs)
  {
#warning "Using AVX for double += double "


    // copy to YMM register
    double* rf = (double *)&(rhs.elem(0));
    double* df = (double *)&(dest.elem(0));

    _mm256_store_pd(df,_mm256_add_pd(_mm256_load_pd(df), _mm256_load_pd(rf)));
  }

  // IScalar * ILattice
  template<>
  inline typename BinaryReturn<IScalar<double>, ILattice<double,4>, OpMultiply>::Type_t
  operator*(const IScalar<double>& l, const ILattice<double,4>& r)
  {
#warning "Using AVX for scalar double * double "
    typename BinaryReturn<IScalar<double>, ILattice<double,4>, OpMultiply>::Type_t  d;

    // Load the float and broadcast it.
    __m256d lvec = _mm256_broadcast_sd((const double *)&(l.elem()));

    double* rf = (double *)&(r.elem(0));
    double* df = (double *)&(d.elem(0));

    _mm256_store_pd(df,_mm256_mul_pd(lvec, _mm256_load_pd(rf)));

    return d;
  }

  // ILattice*IScalar
  template<>
  inline typename BinaryReturn<ILattice<double,4>, IScalar<double>, OpMultiply>::Type_t
  operator*(const ILattice<double,4>& l, const IScalar<double>& r)
  {
#warning "Using AVX for double * scalar double "
    typename BinaryReturn<ILattice<double,4>, IScalar<double>, OpMultiply>::Type_t  d;

    // Load the float and broadcast it.
    __m256d rvec = _mm256_broadcast_sd((const double *)&(r.elem()));

    double* lf = (double *)&(l.elem(0));
    double* df = (double *)&(d.elem(0));

    _mm256_store_pd(df,_mm256_mul_pd(_mm256_load_pd(lf), rvec));

    return d;
  }

  // ILattice * ILattice
  template<>
  inline typename BinaryReturn<ILattice<double,4>, ILattice<double,4>, OpMultiply>::Type_t
  operator*(const ILattice<double,4>& l, const ILattice<double,4>& r)
  {
#warning "Using AVX for double * double "
    typename BinaryReturn<ILattice<double, 4>, ILattice<double,4>, OpMultiply>::Type_t  d;
    // copy to YMM register
    double* lf = (double *)&(l.elem(0));
    double* rf = (double *)&(r.elem(0));
    double* df = (double *)&(d.elem(0));

    if ((unsigned long)lf & 31 != 0 || (unsigned long)rf & 31 != 0 || (unsigned long)df & 31 != 0)
      fprintf (stderr, "memory address lf = %p rf = %p df = %p\n",
	       lf, rf, df);

    _mm256_store_pd(df,_mm256_mul_pd(_mm256_load_pd(lf), _mm256_load_pd(rf)));

    return d;
  }


// ILattice + ILattice
  template<>
  inline typename BinaryReturn<ILattice<double,4>, ILattice<double,4>, OpAdd>::Type_t
  operator+(const ILattice<double,4>& l, const ILattice<double,4>& r)
  {
#warning "Using AVX for double + double "
    typename BinaryReturn<ILattice<double,4>, ILattice<double,4>, OpAdd>::Type_t  d;
    // copy to YMM register
    double* lf = (double *)&(l.elem(0));
    double* rf = (double *)&(r.elem(0));
    double* df = (double *)&(d.elem(0));

    if ((unsigned long)lf & 31 != 0 || (unsigned long)rf & 31 != 0 || (unsigned long)df & 31 != 0)
      fprintf (stderr, "memory address lf = %p rf = %p df = %p\n",
	       lf, rf, df);

    _mm256_store_pd(df,_mm256_add_pd(_mm256_load_pd(lf), _mm256_load_pd(rf)));

    return d;
  }


  // ILattice - ILattice
  template<>
  inline typename BinaryReturn<ILattice<double,4>, ILattice<double,4>, OpSubtract>::Type_t
  operator-(const ILattice<double,4>& l, const ILattice<double,4>& r)
  {
#warning "Using AVX for double - double "
    typename BinaryReturn<ILattice<double,4>, ILattice<double,4>, OpSubtract>::Type_t  d;

    // copy to YMM register
    double* lf = (double *)&(l.elem(0));
    double* rf = (double *)&(r.elem(0));
    double* df = (double *)&(d.elem(0));

    if ((unsigned long)lf & 63 != 0 || (unsigned long)rf & 63 != 0 || (unsigned long)df & 63 != 0)
      fprintf (stderr, "memory address lf = %p rf = %p df = %p\n",
	       lf, rf, df);

    _mm256_store_pd(df,_mm256_sub_pd(_mm256_load_pd(lf), _mm256_load_pd(rf)));
    return d;
  }
#endif

  // Masked copies
  template<>
  inline void
  copy_inner_mask(ILattice<float,8>& dest, const ILattice<bool,8>& mask, const ILattice<float,8>& src)
  {
    __m256i avxmask=_mm256_set_epi32( ( mask.elem(7) ? (int) -1 : 0 ), 
				      ( mask.elem(6) ? (int) -1 : 0 ), 
				       ( mask.elem(5) ? (int) -1 : 0 ), 
				       ( mask.elem(4) ? (int) -1 : 0 ), 
				       ( mask.elem(3) ? (int) -1 : 0 ), 
				       ( mask.elem(2) ? (int) -1 : 0 ), 
				       ( mask.elem(1) ? (int) -1 : 0 ), 
				       ( mask.elem(0) ? (int) -1 : 0 ));
    
    float *dptr = (float *)&(dest.elem(0));
    const float *srcptr = (const float *)&(src.elem(0));

    _mm256_maskstore_ps(dptr, avxmask, _mm256_maskload_ps(srcptr, avxmask));
  }
  

  // Masked copies
  template<>
  inline void
  copy_inner_mask(ILattice<double,4>& dest, const ILattice<bool,4>& mask, const ILattice<double,4>& src)
  {
    __m256i avxmask=_mm256_set_epi64x( ( mask.elem(3) ? (long long) -1 : 0 ), 
					( mask.elem(2) ? (long long) -1 : 0 ), 
					( mask.elem(1) ? (long long) -1 : 0 ), 
					( mask.elem(0) ? (long long) -1 : 0 ));
    
    double *dptr = (double *)&(dest.elem(0));
    const double *srcptr = (const double *)&(src.elem(0));

    _mm256_maskstore_pd(dptr, avxmask, _mm256_maskload_pd(srcptr, avxmask));
}
};

#endif
