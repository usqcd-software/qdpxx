// -*- C++ -*-
// $Id: qdp_inner.h,v 1.1 2003-05-22 20:06:27 edwards Exp $
//
// QDP data parallel interface
//

QDP_BEGIN_NAMESPACE(QDP);

//-------------------------------------------------------------------------------------
//! Scalar inner lattice
/*! All inner lattices are of IScalar or ILattice type */
template<class T> class IScalar: public ScalarDom<T>
{
public:
  IScalar() {}
  ~IScalar() {}

//private:
  /*! Hide copy constructor to prevent copying */
  IScalar(const IScalar&) {}
};


//-------------------------------------------------------------------------------------
//! Inner lattice class
/*!
   * Lattice class for vector like architectures. 
   * Also mixed mode super-scalar architectures with vector extensions
   */
template<class T> class ILattice
{
public:
  /*! Constructor should check on its real size */
  ILattice() {F = new T[geom.Vol()];fprintf(stderr,"create ILattice[%d]=0x%x\n",geom.Vol(),F);}
  ~ILattice() {fprintf(stderr,"destroy ILattice=0x%x\n",F);delete[] F;}

public:
  T& elem(int i) {return F[i];}
  const T& elem(int i) const {return F[i];}

private:
  /*! Representation */
  T* F;
};

QDP_END_NAMESPACE();
