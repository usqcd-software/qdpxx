// -*- C++ -*-
// $Id: outersubtype.h,v 1.1 2002-10-02 20:31:39 edwards Exp $
//
// QDP data parallel interface
//

QDP_BEGIN_NAMESPACE(QDP);

//! OScalar class narrowed to a subset
/*! 
 * Only used for lvalues
 */
template<class T>
class OSubScalar: public QDPSubType<T, OScalar<T> >
{
public:
  OSubScalar(const OScalar<T>& a, const Subset& ss): QDPSubType<T,OScalar<T> >(a,ss) {}
  OSubScalar(const OSubScalar& a): QDPSubType<T,OScalar<T> >(a) {}
  ~OSubScalar() {}

  //---------------------------------------------------------
  // Operators
  // NOTE: all assignment-like operators except operator= are
  // inherited from QDPSubType

  inline
  void operator=(const typename WordType<T>::Type_t& rhs)
    {
      assign(rhs);
    }

  template<class T1,class C1>
  inline
  void operator=(const QDPType<T1,C1>& rhs)
    {
      assign(rhs);
    }

  template<class T1,class C1>
  inline
  void operator=(const QDPExpr<T1,C1>& rhs)
    {
      assign(rhs);
    }


private:
  // Hide default constructor
  OSubScalar() {}
};



//-------------------------------------------------------------------------------------
//! OLattice class narrowed to a subset
/*! 
 * Only used for lvalues
 */
template<class T> 
class OSubLattice: public QDPSubType<T, OLattice<T> >
{
public:
  OSubLattice(const OLattice<T>& a, const Subset& ss): QDPSubType<T,OLattice<T> >(a,ss) {}
  OSubLattice(const OSubLattice& a): QDPSubType<T,OLattice<T> >(a) {}
  ~OSubLattice() {}

  //---------------------------------------------------------
  // Operators
  // NOTE: all assignment-like operators except operator= are
  // inherited from QDPType

  inline
  void operator=(const typename WordType<T>::Type_t& rhs)
    {
      assign(rhs);
    }

  template<class T1,class C1>
  inline
  void operator=(const QDPType<T1,C1>& rhs)
    {
      assign(rhs);
    }

  template<class T1,class C1>
  inline
  void operator=(const QDPExpr<T1,C1>& rhs)
    {
      assign(rhs);
    }


private:
  // Hide default constructor
  OSubLattice() {}
};



// OSubLattice Op Scalar(Expression(source))
/* Implementation in relevant specific files */
template<class T, class T1, class Op, class RHS>
void evaluate(OSubLattice<T>& dest, const Op& op, const QDPExpr<RHS,OSubScalar<T1> >& rhs);

// OSubLattice Op OSubLattice(Expression(source))
/* Implementation in relevant specific files */
template<class T, class T1, class Op, class RHS>
void evaluate(OSubLattice<T>& dest, const Op& op, const QDPExpr<RHS,OSubLattice<T1> >& rhs);



//-----------------------------------------------------------------------------
// Traits classes to support operations of simple scalars (floating constants, 
// etc.) on QDPTypes
//-----------------------------------------------------------------------------

template<class T>
struct WordType<OSubScalar<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};

template<class T>
struct WordType<OSubLattice<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};


//-----------------------------------------------------------------------------
// Scalar Operations
//-----------------------------------------------------------------------------

//! dest = 0
template<class T> 
void zero(OSubScalar<T> dest) 
{
  zero(dest.field().elem());
}

//! dest = (mask) ? s1 : dest
template<class T1, class T2> 
void copymask(OSubScalar<T2> dest, const OSubScalar<T1>& mask, const OSubScalar<T2>& s1) 
{
  copymask(dest.field().elem(), mask.elem(), s1.elem());
}


//-----------------------------------------------------------------------------
// Random numbers
//! dest  = random  
/*! Implementation is in the specific files */
template<class T>
void random(OSubScalar<T> d);


//! dest  = gaussian
template<class T>
void gaussian(OSubScalar<T> dd)
{
  OLattice<T>& d = dd.field();
  const Subset& s = dd.subset();

  OScalar<T>  r1, r2;

  random(r1(s));
  random(r2(s));

  fill_gaussian(d.elem(), r1.elem(), r2.elem());
}


QDP_END_NAMESPACE();
