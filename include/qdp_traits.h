// -*- C++ -*-
// $Id: qdp_traits.h,v 1.6 2004-07-02 21:54:04 edwards Exp $

/*! @file
 * @brief Traits classes
 * 
 * Traits classes needed internally
 */

QDP_BEGIN_NAMESPACE(QDP);


//-----------------------------------------------------------------------------
// Traits class for returning the QDP container class
//-----------------------------------------------------------------------------

template<class T>
struct QDPContainer
{
  typedef T Type_t;
};

template<class T,class C>
struct QDPContainer<QDPType<T,C> >
{
  typedef C Type_t;
};

template<class T,class C>
struct QDPContainer<QDPExpr<T,C> >
{
  typedef C Type_t;
};

template<class T>
struct QDPContainer<Reference<T> >
{
  typedef typename QDPContainer<T>::Type_t Type_t;
};


template<class T1,class Op>
struct QDPContainer<UnaryNode<Op,T1> >
{
  typedef typename UnaryReturn<T1,Op>::Type_t     Return_t;
  typedef typename QDPContainer<Return_t>::Type_t Type_t;
};

template<class T1,class T2,class Op>
struct QDPContainer<BinaryNode<Op,T1,T2> >
{
  typedef typename BinaryReturn<T1,T2,Op>::Type_t Return_t;
  typedef typename QDPContainer<Return_t>::Type_t Type_t;
};

template<class T1,class T2,class T3,class Op>
struct QDPContainer<TrinaryNode<Op,T1,T2,T3> >
{
  typedef typename TrinaryReturn<T1,T2,T3,Op>::Type_t Return_t;
  typedef typename QDPContainer<Return_t>::Type_t Type_t;
};


//-----------------------------------------------------------------------------
// Traits class for returning the subset-ted class name of a outer grid class
//-----------------------------------------------------------------------------

template<class T, class S>
struct QDPSubTypeTrait {};

//-----------------------------------------------------------------------------
// Traits classes to support operations of simple scalars (floating constants, 
// etc.) on QDPTypes
//-----------------------------------------------------------------------------

//! Find the underlying word type of a field
template<class T>
struct WordType
{
  typedef T  Type_t;
};


//-----------------------------------------------------------------------------
// Constructors for simple word types
//-----------------------------------------------------------------------------

//! Construct simple word type. Default behavior is empty
template<class T>
struct SimpleScalar {};


//! Construct simple word type used at some level within primitives
template<class T>
struct InternalScalar {};


//! Makes a primitive scalar leaving grid alone
template<class T>
struct PrimitiveScalar {};


//! Makes a lattice scalar leaving primitive indices alone
template<class T>
struct LatticeScalar {};


//! Construct simple word type used at some level within primitives
template<class T>
struct RealScalar {};


//! Construct primitive type of input but always RScalar complex type
template<class T>
struct NoComplex {};


//! Simple zero tag
struct Zero {};

//! Put zero in some unnamed space
namespace {
Zero zero;
}

QDP_END_NAMESPACE();

