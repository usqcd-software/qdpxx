// -*- C++ -*-
// $Id: qdptype.h,v 1.3 2002-10-12 00:58:32 edwards Exp $
//
// QDP data parallel interface
//

QDP_BEGIN_NAMESPACE(QDP);


/*! @addtogroup group1 */
/*! @{ */


//! QDPType - major type class/container for all QDP objects
/*! 
 * This is the top level class all users should access for functional
 * and infix operations
 */
template<class T, class C> 
class QDPType
{
public:
  //! Main constructor 
  QDPType() {}
  //! Copy constructor
  QDPType(const QDPType&) {}
  //! Destructor
  ~QDPType() {}


  //---------------------------------------------------------
  // Operators

  inline
  C& assign(const typename WordType<C>::Type_t& rhs)
    {
      C* me = static_cast<C*>(this);
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(*me,OpAssign(),PETE_identity(Scalar_t(rhs)));
      return *me;
    }

  inline
  C& assign(const Zero&)
    {
      C* me = static_cast<C*>(this);
      zero_rep(*me);
      return *me;
    }

  template<class T1,class C1>
  inline
  C& assign(const QDPType<T1,C1>& rhs)
    {
      C* me = static_cast<C*>(this);
      evaluate(*me,OpAssign(),PETE_identity(rhs));
      return *me;
    }

  template<class T1,class C1>
  inline
  C& assign(const QDPExpr<T1,C1>& rhs)
    {
      C* me = static_cast<C*>(this);
      evaluate(*me,OpAssign(),rhs);
      return *me;
    }


  inline
  C& operator+=(const typename WordType<C>::Type_t& rhs)
    {
      C* me = static_cast<C*>(this);
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(*me,OpAddAssign(),PETE_identity(Scalar_t(rhs)));
      return *me;
    }

  template<class T1,class C1>
  inline
  C& operator+=(const QDPType<T1,C1>& rhs)
    {
      C* me = static_cast<C*>(this);
      evaluate(*me,OpAddAssign(),PETE_identity(rhs));
      return *me;
    }

  template<class T1,class C1>
  inline
  C& operator+=(const QDPExpr<T1,C1>& rhs)
    {
      C* me = static_cast<C*>(this);
      evaluate(*me,OpAddAssign(),rhs);
      return *me;
    }


  inline
  C& operator-=(const typename WordType<C>::Type_t& rhs)
    {
      C* me = static_cast<C*>(this);
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(*me,OpSubtractAssign(),PETE_identity(Scalar_t(rhs)));
      return *me;
    }

  template<class T1,class C1>
  inline
  C& operator-=(const QDPType<T1,C1>& rhs)
    {
      C* me = static_cast<C*>(this);
      evaluate(*me,OpSubtractAssign(),PETE_identity(rhs));
      return *me;
    }

  template<class T1,class C1>
  inline
  C& operator-=(const QDPExpr<T1,C1>& rhs)
    {
      C* me = static_cast<C*>(this);
      evaluate(*me,OpSubtractAssign(),rhs);
      return *me;
    }


  inline
  C& operator*=(const typename WordType<C>::Type_t& rhs)
    {
      C* me = static_cast<C*>(this);
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(*me,OpMultiplyAssign(),PETE_identity(Scalar_t(rhs)));
      return *me;
    }

  template<class T1,class C1>
  inline
  C& operator*=(const QDPType<T1,C1>& rhs)
    {
      C* me = static_cast<C*>(this);
      evaluate(*me,OpMultiplyAssign(),PETE_identity(rhs));
      return *me;
    }

  template<class T1,class C1>
  inline
  C& operator*=(const QDPExpr<T1,C1>& rhs)
    {
      C* me = static_cast<C*>(this);
      evaluate(*me,OpMultiplyAssign(),rhs);
      return *me;
    }


  inline
  C& operator/=(const typename WordType<C>::Type_t& rhs)
    {
      C* me = static_cast<C*>(this);
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(*me,OpDivideAssign(),PETE_identity(Scalar_t(rhs)));
      return *me;
    }

  template<class T1,class C1>
  inline
  C& operator/=(const QDPType<T1,C1>& rhs)
    {
      C* me = static_cast<C*>(this);
      evaluate(*me,OpDivideAssign(),PETE_identity(rhs));
      return *me;
    }

  template<class T1,class C1>
  inline
  C& operator/=(const QDPExpr<T1,C1>& rhs)
    {
      C* me = static_cast<C*>(this);
      evaluate(*me,OpDivideAssign(),rhs);
      return *me;
    }


  inline
  C& operator%=(const typename WordType<C>::Type_t& rhs)
    {
      C* me = static_cast<C*>(this);
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(*me,OpModAssign(),PETE_identity(Scalar_t(rhs)));
      return *me;
    }

  template<class T1,class C1>
  inline
  C& operator%=(const QDPType<T1,C1>& rhs)
    {
      C* me = static_cast<C*>(this);
      evaluate(*me,OpModAssign(),PETE_identity(rhs));
      return *me;
    }

  template<class T1,class C1>
  inline
  C& operator%=(const QDPExpr<T1,C1>& rhs)
    {
      C* me = static_cast<C*>(this);
      evaluate(*me,OpModAssign(),rhs);
      return *me;
    }


  inline
  C& operator|=(const typename WordType<C>::Type_t& rhs)
    {
      C* me = static_cast<C*>(this);
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(*me,OpBitwiseOrAssign(),PETE_identity(Scalar_t(rhs)));
      return *me;
    }

  template<class T1,class C1>
  inline
  C& operator|=(const QDPType<T1,C1>& rhs)
    {
      C* me = static_cast<C*>(this);
      evaluate(*me,OpBitwiseOrAssign(),PETE_identity(rhs));
      return *me;
    }

  template<class T1,class C1>
  inline
  C& operator|=(const QDPExpr<T1,C1>& rhs)
    {
      C* me = static_cast<C*>(this);
      evaluate(*me,OpBitwiseOrAssign(),PETE_identity(rhs));
      return *me;
    }


  inline
  C& operator&=(const typename WordType<C>::Type_t& rhs)
    {
      C* me = static_cast<C*>(this);
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(*me,OpBitwiseAndAssign(),PETE_identity(Scalar_t(rhs)));
      return *me;
    }

  template<class T1,class C1>
  inline
  C& operator&=(const QDPType<T1,C1>& rhs)
    {
      C* me = static_cast<C*>(this);
      evaluate(*me,OpBitwiseAndAssign(),PETE_identity(rhs));
      return *me;
    }

  template<class T1,class C1>
  inline
  C& operator&=(const QDPExpr<T1,C1>& rhs)
    {
      C* me = static_cast<C*>(this);
      evaluate(*me,OpBitwiseAndAssign(),rhs);
      return *me;
    }


  inline
  C& operator^=(const typename WordType<C>::Type_t& rhs)
    {
      C* me = static_cast<C*>(this);
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(*me,OpBitwiseXorAssign(),PETE_identity(Scalar_t(rhs)));
      return *me;
    }

  template<class T1,class C1>
  inline
  C& operator^=(const QDPType<T1,C1>& rhs)
    {
      C* me = static_cast<C*>(this);
      evaluate(*me,OpBitwiseXorAssign(),PETE_identity(rhs));
      return *me;
    }

  template<class T1,class C1>
  inline
  C& operator^=(const QDPExpr<T1,C1>& rhs)
    {
      C* me = static_cast<C*>(this);
      evaluate(*me,OpBitwiseXorAssign(),rhs);
      return *me;
    }


  inline
  C& operator<<=(const typename WordType<C>::Type_t& rhs)
    {
      C* me = static_cast<C*>(this);
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(*me,OpLeftShiftAssign(),PETE_identity(Scalar_t(rhs)));
      return *me;
    }

  template<class T1,class C1>
  inline
  C& operator<<=(const QDPType<T1,C1>& rhs)
    {
      C* me = static_cast<C*>(this);
      evaluate(*me,OpLeftShiftAssign(),PETE_identity(rhs));
      return *me;
    }

  template<class T1,class C1>
  inline
  C& operator<<=(const QDPExpr<T1,C1>& rhs)
    {
      C* me = static_cast<C*>(this);
      evaluate(*me,OpLeftShiftAssign(),rhs);
      return *me;
    }


  inline
  C& operator>>=(const typename WordType<C>::Type_t& rhs)
    {
      C* me = static_cast<C*>(this);
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(*me,OpRightShiftAssign(),PETE_identity(Scalar_t(rhs)));
      return *me;
    }

  template<class T1,class C1>
  inline
  C& operator>>=(const QDPType<T1,C1>& rhs)
    {
      C* me = static_cast<C*>(this);
      evaluate(*me,OpRightShiftAssign(),PETE_identity(rhs));
      return *me;
    }

  template<class T1,class C1>
  inline
  C& operator>>=(const QDPExpr<T1,C1>& rhs)
    {
      C* me = static_cast<C*>(this);
      evaluate(*me,OpRightShiftAssign(),rhs);
      return *me;
    }


public:
  T& elem(int i) {return static_cast<const C*>(this)->elem(i);}
  const T& elem(int i) const {return static_cast<const C*>(this)->elem(i);}

  T& elem() {return static_cast<const C*>(this)->elem();}
  const T& elem() const {return static_cast<const C*>(this)->elem();}
};

/*! @} */ // end of group1

//-----------------------------------------------------------------------------
// We need to specialize CreateLeaf<T> for our class, so that operators
// know what to stick in the leaves of the expression tree.
//-----------------------------------------------------------------------------

template<class T, class C>
struct CreateLeaf<QDPType<T,C> >
{
  typedef QDPType<T,C> Inp_t;
  typedef Reference<Inp_t> Leaf_t;
//  typedef Inp_t Leaf_t;
  inline static
  Leaf_t make(const Inp_t &a) { return Leaf_t(a); }
};


//-----------------------------------------------------------------------------
// Specialization of LeafFunctor class for applying the EvalLeaf1
// tag to a QDPType. The apply method simply returns the array
// evaluated at the point.
//-----------------------------------------------------------------------------

template<class T, class C>
struct LeafFunctor<QDPType<T,C>, ElemLeaf>
{
  typedef Reference<T> Type_t;
//  typedef T Type_t;
  inline static Type_t apply(const QDPType<T,C> &a, const ElemLeaf &f)
    { 
      return Type_t(a.elem());
    }
};

template<class T, class C>
struct LeafFunctor<QDPType<T,C>, EvalLeaf1>
{
  typedef Reference<T> Type_t;
//  typedef T Type_t;
  inline static Type_t apply(const QDPType<T,C> &a, const EvalLeaf1 &f)
    { 
      return Type_t(a.elem(f.val1()));
    }
};


QDP_END_NAMESPACE();

