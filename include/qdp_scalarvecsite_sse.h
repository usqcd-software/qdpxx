// -*- C++ -*-
// $Id: qdp_scalarvecsite_sse.h,v 1.1 2003-08-21 03:29:46 edwards Exp $

/*! @file
 * @brief Intel SSE optimizations
 *
 * SSE optimizations of basic operations
 */

#ifndef QDP_SCALARVECSITE_SSE_H
#define QDP_SCALARVECSITE_SSE_H

// These SSE asm instructions are only supported under GCC/G++
#if defined(__GNUC__)

QDP_BEGIN_NAMESPACE(QDP);

/*! @defgroup optimizations  Optimizations
 *
 * Optimizations for basic QDP operations
 *
 * @{
 */

// Use this def just to safe some typing later on in the file
#define RComplexFloat  RComplex<ILattice<float,4> >


//! Specialized Inner lattice class
/*! Uses sse  */
template<> class ILattice<float, 4>
{
public:
  typedef float  T;
  static const int N = 4;

  ILattice() {}
  ~ILattice() {}

  //---------------------------------------------------------
  //! construct dest = const
  ILattice(const WordType<float>::Type_t& rhs)
    {
      for(int i=0; i < N; ++i)
	elem(i) = rhs;
    }

  //! construct dest = rhs
  template<class T1>
  ILattice(const ILattice<T1,N>& rhs)
    {
      for(int i=0; i < N; ++i)
	elem(i) = rhs.elem(i);
    }

  //! construct dest = rhs
  template<class T1>
  ILattice(const T1& rhs)
    {
      for(int i=0; i < N; ++i)
	elem(i) = rhs;
    }

  //---------------------------------------------------------
#if 0
  //! dest = const
  /*! Fill with an integer constant. Will be promoted to underlying word type */
  inline
  ILattice& operator=(const typename WordType<T>::Type_t& rhs)
    {
      for(int i=0; i < N; ++i)
	elem(i) = rhs;

      return *this;
    }
#endif

  //---------------------------------------------------------
  //! ILattice = IScalar
  /*! Set equal to an IScalar */
  template<class T1>
  inline
  ILattice& operator=(const IScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) = rhs.elem();

      return *this;
    }

  //! ILattice += IScalar
  template<class T1>
  inline
  ILattice& operator+=(const IScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) += rhs.elem();

      return *this;
    }

  //! ILattice -= IScalar
  template<class T1>
  inline
  ILattice& operator-=(const IScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) -= rhs.elem();

      return *this;
    }

  //! ILattice *= IScalar
  template<class T1>
  inline
  ILattice& operator*=(const IScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) *= rhs.elem();

      return *this;
    }

  //! ILattice /= IScalar
  template<class T1>
  inline
  ILattice& operator/=(const IScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) /= rhs.elem();

      return *this;
    }

  //! ILattice %= IScalar
  template<class T1>
  inline
  ILattice& operator%=(const IScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) %= rhs.elem();

      return *this;
    }

  //! ILattice |= IScalar
  template<class T1>
  inline
  ILattice& operator|=(const IScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) |= rhs.elem();

      return *this;
    }

  //! ILattice &= IScalar
  template<class T1>
  inline
  ILattice& operator&=(const IScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) &= rhs.elem();

      return *this;
    }

  //! ILattice ^= IScalar
  template<class T1>
  inline
  ILattice& operator^=(const IScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) ^= rhs.elem();

      return *this;
    }

  //! ILattice <<= IScalar
  template<class T1>
  inline
  ILattice& operator<<=(const IScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) <<= rhs.elem();

      return *this;
    }

  //! ILattice >>= IScalar
  template<class T1>
  inline
  ILattice& operator>>=(const IScalar<T1>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) >>= rhs.elem();

      return *this;
    }

  //---------------------------------------------------------
  //! ILattice = ILattice
  /*! Set equal to another ILattice */
  inline
  ILattice& operator=(const ILattice& rhs) 
    {
      F = rhs.F;

      return *this;
    }

  //! ILattice += ILattice
  inline
  ILattice& operator+=(const ILattice& rhs) 
    {
      F += rhs.F;

      return *this;
    }

  //! ILattice -= ILattice
  template<class T1>
  inline
  ILattice& operator-=(const ILattice& rhs) 
    {
      F -= rhs.F;

      return *this;
    }

  //! ILattice *= ILattice
  template<class T1>
  inline
  ILattice& operator*=(const ILattice<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) *= rhs.elem(i);

      return *this;
    }

  //! ILattice /= ILattice
  template<class T1>
  inline
  ILattice& operator/=(const ILattice<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) /= rhs.elem(i);

      return *this;
    }

  //! ILattice %= ILattice
  template<class T1>
  inline
  ILattice& operator%=(const ILattice<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) %= rhs.elem(i);

      return *this;
    }

  //! ILattice |= ILattice
  template<class T1>
  inline
  ILattice& operator|=(const ILattice<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) |= rhs.elem(i);

      return *this;
    }

  //! ILattice &= ILattice
  template<class T1>
  inline
  ILattice& operator&=(const ILattice<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) &= rhs.elem(i);

      return *this;
    }

  //! ILattice ^= ILattice
  template<class T1>
  inline
  ILattice& operator^=(const ILattice<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) ^= rhs.elem(i);

      return *this;
    }

  //! ILattice <<= ILattice
  template<class T1>
  inline
  ILattice& operator<<=(const ILattice<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) <<= rhs.elem(i);

      return *this;
    }

  //! ILattice >>= ILattice
  template<class T1>
  inline
  ILattice& operator>>=(const ILattice<T1,N>& rhs) 
    {
      for(int i=0; i < N; ++i)
	elem(i) >>= rhs.elem(i);

      return *this;
    }

#if 0
  // NOTE: intentially avoid defining a copy constructor - let the compiler
  // generate one via the bit copy mechanism. This effectively achieves
  // the first form of the if below (QDP_USE_ARRAY_INITIALIZER) without having
  // to use that syntax which is not strictly legal in C++.
#endif

  //! Deep copy constructor
#if defined(QDP_USE_ARRAY_INITIALIZER)
  ILattice(const ILattice& a) : F(a.F) {}
#else
  /*! This is a copy form - legal but not necessarily efficient */
  ILattice(const ILattice& a)
    {
      // fprintf(stderr,"copy ILattice\n");
      F = a.F;
    }
#endif

public:
  //! The backdoor
  /*! 
   * Used by optimization routines (e.g., SSE) that need the memory address of data.
   * BTW: to make this a friend would be a real pain since functions are templatized.
   */
//  inline T* data() const {return &F;}


public:
  T& elem(int i) {return *(&F + i);}
  const T& elem(int i) const {return *(&F + i);}

private:
  // SSE attributes
  typedef float v4sf __attribute__ ((mode(V4SF)));
  v4sf F;

} QDP_ALIGN16;   // possibly force alignment


/*! @} */   // end of group optimizations

QDP_END_NAMESPACE();

#endif  // defined(__GNUC__)

#endif
