// -*- C++ -*-
// $Id: qdpsubtype.h,v 1.3 2002-10-12 04:10:15 edwards Exp $

/*! @file
 * @brief QDPType after a subset
 *
 * Subclass of QDPType used for subset operations
 */

QDP_BEGIN_NAMESPACE(QDP);


//! QDPSubType - type representing a field living on a subset
/*! 
 * This class is meant to be an auxilliary class used only for
 * things like lvalues - left hand side of expressions, arguments
 * to calls that modify the source (like RNG), etc.
 */
template<class T, class C> 
class QDPSubType
{
public:
  //! Can only construct from a QDPType and a Subset
  QDPSubType(const QDPType<T,C>& a, const Subset& ss): F(a), s(ss) {}

  //! Copy constructor
  QDPSubType(const QDPSubType& a): F(a.F), s(a.s) {}

  //! Destructor
  ~QDPSubType() {}


  //---------------------------------------------------------
  // Operators

  inline
  void assign(const typename WordType<C>::Type_t& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(me,OpAssign(),PETE_identity(Scalar_t(rhs)),s);
    }

  inline
  void assign(const Zero&)
    {
      zero_rep(static_cast<C&>(const_cast<QDPType<T,C>&>(F)),s);
    }

  template<class T1,class C1>
  inline
  void assign(const QDPType<T1,C1>& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      evaluate(me,OpAssign(),PETE_identity(rhs),s);
    }

  template<class T1,class C1>
  inline
  void assign(const QDPExpr<T1,C1>& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      evaluate(me,OpAssign(),rhs,s);
    }


  inline
  void operator+=(const typename WordType<C>::Type_t& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(me,OpAddAssign(),PETE_identity(Scalar_t(rhs)),s);
    }

  template<class T1,class C1>
  inline
  void operator+=(const QDPType<T1,C1>& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      evaluate(me,OpAddAssign(),PETE_identity(rhs),s);
    }

  template<class T1,class C1>
  inline
  void operator+=(const QDPExpr<T1,C1>& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      evaluate(me,OpAddAssign(),rhs,s);
    }


  inline
  void operator-=(const typename WordType<C>::Type_t& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(me,OpSubtractAssign(),PETE_identity(Scalar_t(rhs)),s);
    }

  template<class T1,class C1>
  inline
  void operator-=(const QDPType<T1,C1>& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      evaluate(me,OpSubtractAssign(),PETE_identity(rhs),s);
    }

  template<class T1,class C1>
  inline
  void operator-=(const QDPExpr<T1,C1>& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      evaluate(me,OpSubtractAssign(),rhs,s);
    }


  inline
  void operator*=(const typename WordType<C>::Type_t& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(me,OpMultiplyAssign(),PETE_identity(Scalar_t(rhs)),s);
    }

  template<class T1,class C1>
  inline
  void operator*=(const QDPType<T1,C1>& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      evaluate(me,OpMultiplyAssign(),PETE_identity(rhs),s);
    }

  template<class T1,class C1>
  inline
  void operator*=(const QDPExpr<T1,C1>& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      evaluate(me,OpMultiplyAssign(),rhs,s);
    }


  inline
  void operator/=(const typename WordType<C>::Type_t& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(me,OpDivideAssign(),PETE_identity(Scalar_t(rhs)),s);
    }

  template<class T1,class C1>
  inline
  void operator/=(const QDPType<T1,C1>& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      evaluate(me,OpDivideAssign(),PETE_identity(rhs),s);
    }

  template<class T1,class C1>
  inline
  void operator/=(const QDPExpr<T1,C1>& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      evaluate(me,OpDivideAssign(),rhs,s);
    }


  inline
  void operator%=(const typename WordType<C>::Type_t& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(me,OpModAssign(),PETE_identity(Scalar_t(rhs)),s);
    }

  template<class T1,class C1>
  inline
  void operator%=(const QDPType<T1,C1>& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      evaluate(me,OpModAssign(),PETE_identity(rhs),s);
    }

  template<class T1,class C1>
  inline
  void operator%=(const QDPExpr<T1,C1>& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      evaluate(me,OpModAssign(),rhs,s);
    }


  inline
  void operator|=(const typename WordType<C>::Type_t& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(me,OpBitwiseOrAssign(),PETE_identity(Scalar_t(rhs)),s);
    }

  template<class T1,class C1>
  inline
  void operator|=(const QDPType<T1,C1>& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      evaluate(me,OpBitwiseOrAssign(),PETE_identity(rhs),s);
    }

  template<class T1,class C1>
  inline
  void operator|=(const QDPExpr<T1,C1>& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      evaluate(me,OpBitwiseOrAssign(),PETE_identity(rhs),s);
    }


  inline
  void operator&=(const typename WordType<C>::Type_t& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(me,OpBitwiseAndAssign(),PETE_identity(Scalar_t(rhs)),s);
    }

  template<class T1,class C1>
  inline
  void operator&=(const QDPType<T1,C1>& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      evaluate(me,OpBitwiseAndAssign(),PETE_identity(rhs),s);
    }

  template<class T1,class C1>
  inline
  void operator&=(const QDPExpr<T1,C1>& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      evaluate(me,OpBitwiseAndAssign(),rhs,s);
    }


  inline
  void operator^=(const typename WordType<C>::Type_t& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(me,OpBitwiseXorAssign(),PETE_identity(Scalar_t(rhs)));
    }

  template<class T1,class C1>
  inline
  void operator^=(const QDPType<T1,C1>& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      evaluate(me,OpBitwiseXorAssign(),PETE_identity(rhs),s);
    }

  template<class T1,class C1>
  inline
  void operator^=(const QDPExpr<T1,C1>& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      evaluate(me,OpBitwiseXorAssign(),rhs,s);
    }


  inline
  void operator<<=(const typename WordType<C>::Type_t& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(me,OpLeftShiftAssign(),PETE_identity(Scalar_t(rhs)),s);
    }

  template<class T1,class C1>
  inline
  void operator<<=(const QDPType<T1,C1>& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      evaluate(me,OpLeftShiftAssign(),PETE_identity(rhs),s);
    }

  template<class T1,class C1>
  inline
  void operator<<=(const QDPExpr<T1,C1>& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      evaluate(me,OpLeftShiftAssign(),rhs,s);
    }


  inline
  void operator>>=(const typename WordType<C>::Type_t& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(me,OpRightShiftAssign(),PETE_identity(Scalar_t(rhs)),s);
    }

  template<class T1,class C1>
  inline
  void operator>>=(const QDPType<T1,C1>& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      evaluate(me,OpRightShiftAssign(),PETE_identity(rhs),s);
    }

  template<class T1,class C1>
  inline
  void operator>>=(const QDPExpr<T1,C1>& rhs)
    {
      C& me = static_cast<C&>(const_cast<QDPType<T,C>&>(F));
      evaluate(me,OpRightShiftAssign(),rhs,s);
    }

private:
  //! Hide default constructor 
  QDPSubType() {}


public:
  C& field() {return static_cast<C&>(const_cast<QDPType<T,C>&>(F));}
  const Subset& subset() const {return s;}


private:
  const QDPType<T,C>&  F;
  const Subset& s;
};



QDP_END_NAMESPACE();

