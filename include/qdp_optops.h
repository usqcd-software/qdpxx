// -*- C++ -*-
// $Id: qdp_optops.h,v 1.4 2004-07-02 19:25:16 edwards Exp $

/*! @file
 * @brief PETE optimized operations on QDPTypes
 *
 * These are optimizations that collapse several QDP operations into
 * one mega operation. Assembly/specialization opts do NOT go here.
 */

QDP_BEGIN_NAMESPACE(QDP);

//-----------------------------------------------------------------------------
// Optimization hooks
//-----------------------------------------------------------------------------

struct OpAdjMultiply
{
  PETE_EMPTY_CONSTRUCTORS(OpAdjMultiply)
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, OpAdjMultiply >::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
//  cerr << "adjMultiply" << endl;
//  return (adj(a)*b);
    return adjMultiply(a,b);
  }
};

struct OpMultiplyAdj
{
  PETE_EMPTY_CONSTRUCTORS(OpMultiplyAdj)
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, OpMultiplyAdj >::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
//  cerr << "multiplyAdj" << endl;
//  return (a*adj(b));
    return multiplyAdj(a,b);
  }
};

struct OpAdjMultiplyAdj
{
  PETE_EMPTY_CONSTRUCTORS(OpAdjMultiplyAdj)
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, OpAdjMultiplyAdj >::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
//  cerr << "adjMultiplyAdj" << endl;
//  return (adj(a)*adj(b));
    return adjMultiplyAdj(a,b);
  }
};

// adjMultiply(l,r)  <-  adj(l)*r
template<class T1,class C1,class T2,class C2>
inline typename MakeReturn<BinaryNode<OpAdjMultiply,
  typename CreateLeaf<QDPExpr<UnaryNode<OpIdentity,T1>,C1> >::Leaf_t,
  typename CreateLeaf<QDPType<T2,C2> >::Leaf_t>,
  typename BinaryReturn<C1,C2,OpAdjMultiply>::Type_t >::Expression_t
operator*(const QDPExpr<UnaryNode<FnAdjoint,T1>,C1> & l,
	  const QDPType<T2,C2> & r)
{
  typedef UnaryNode<OpIdentity,T1> NewExpr1_t; // The adj does not change container type

  typedef BinaryNode<OpAdjMultiply,
    typename CreateLeaf<QDPExpr<UnaryNode<OpIdentity,T1>,C1> >::Leaf_t,
    typename CreateLeaf<QDPType<T2,C2> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<C1,C2,OpAdjMultiply>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<QDPExpr<NewExpr1_t,C1> >::make(NewExpr1_t(l.expression().child())),
    CreateLeaf<QDPType<T2,C2> >::make(r)));
}

// adjMultiply<l,Expr>  <-  adj(l)*Expr
template<class T1,class C1,class T2,class C2>
inline typename MakeReturn<BinaryNode<OpAdjMultiply,
  typename CreateLeaf<QDPExpr<UnaryNode<OpIdentity,T1>,C1> >::Leaf_t,
  typename CreateLeaf<QDPExpr<T2,C2> >::Leaf_t>,
  typename BinaryReturn<C1,C2,OpAdjMultiply>::Type_t >::Expression_t
operator*(const QDPExpr<UnaryNode<FnAdjoint,T1>,C1> & l,
	  const QDPExpr<T2,C2> & r)
{
  typedef UnaryNode<OpIdentity,T1> NewExpr1_t; // The adj does not change container type

  typedef BinaryNode<OpAdjMultiply,
    typename CreateLeaf<QDPExpr<UnaryNode<OpIdentity,T1>,C1> >::Leaf_t,
    typename CreateLeaf<QDPExpr<T2,C2> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<C1,C2,OpAdjMultiply>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<QDPExpr<NewExpr1_t,C1> >::make(NewExpr1_t(l.expression().child())),
    CreateLeaf<QDPExpr<T2,C2> >::make(r)));
}

// multplyAdj(l,r)  <-  l*adj(r)
template<class T1,class C1,class T2,class C2>
inline typename MakeReturn<BinaryNode<OpMultiplyAdj,
  typename CreateLeaf<QDPType<T1,C1> >::Leaf_t,
  typename CreateLeaf<QDPExpr<UnaryNode<OpIdentity,T2>,C2> >::Leaf_t>,
  typename BinaryReturn<C1,C2,OpMultiplyAdj>::Type_t >::Expression_t
operator*(const QDPType<T1,C1> & l,
	  const QDPExpr<UnaryNode<FnAdjoint,T2>,C2> & r)
{
  typedef UnaryNode<OpIdentity,T2> NewExpr2_t; // The adj does not change container type

  typedef BinaryNode<OpMultiplyAdj,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t,
    typename CreateLeaf<QDPExpr<UnaryNode<OpIdentity,T2>,C2> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<C1,C2,OpMultiplyAdj>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<QDPType<T1,C1> >::make(l),
    CreateLeaf<QDPExpr<NewExpr2_t,C2> >::make(NewExpr2_t(r.expression().child()))));
}

// multiplyAdj(Expr,r)  <-  Expr*adj(r)
template<class T1,class C1,class T2,class C2>
inline typename MakeReturn<BinaryNode<OpMultiplyAdj,
  typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t,
  typename CreateLeaf<QDPExpr<UnaryNode<OpIdentity,T2>,C2> >::Leaf_t>,
  typename BinaryReturn<C1,C2,OpMultiplyAdj>::Type_t >::Expression_t
operator*(const QDPExpr<T1,C1> & l,
	  const QDPExpr<UnaryNode<FnAdjoint,T2>,C2> & r)
{
  typedef UnaryNode<OpIdentity,T2> NewExpr2_t; // The adj does not change container type

  typedef BinaryNode<OpMultiplyAdj,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t,
    typename CreateLeaf<QDPExpr<UnaryNode<OpIdentity,T2>,C2> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<C1,C2,OpMultiplyAdj>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<QDPExpr<T1,C1> >::make(l),
    CreateLeaf<QDPExpr<NewExpr2_t,C2> >::make(NewExpr2_t(r.expression().child()))));
}

// adjMultiplyAdj(l,r)  <-  adj(l)*adj(r)
template<class T1,class C1,class T2,class C2>
inline typename MakeReturn<BinaryNode<OpAdjMultiplyAdj,
  typename CreateLeaf<QDPExpr<UnaryNode<OpIdentity,T1>,C1> >::Leaf_t,
  typename CreateLeaf<QDPExpr<UnaryNode<OpIdentity,T2>,C2> >::Leaf_t>,
  typename BinaryReturn<C1,C2,OpAdjMultiplyAdj>::Type_t >::Expression_t
operator*(const QDPExpr<UnaryNode<FnAdjoint,T1>,C1> & l,
	  const QDPExpr<UnaryNode<FnAdjoint,T2>,C2> & r)
{
  typedef UnaryNode<OpIdentity,T1> NewExpr1_t; // The adj does not change container type
  typedef UnaryNode<OpIdentity,T2> NewExpr2_t; // The adj does not change container type

  typedef BinaryNode<OpAdjMultiplyAdj,
    typename CreateLeaf<QDPExpr<UnaryNode<OpIdentity,T1>,C1> >::Leaf_t,
    typename CreateLeaf<QDPExpr<UnaryNode<OpIdentity,T2>,C2> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<C1,C2,OpAdjMultiplyAdj>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<QDPExpr<NewExpr1_t,C1> >::make(NewExpr1_t(l.expression().child())),
    CreateLeaf<QDPExpr<NewExpr2_t,C2> >::make(NewExpr2_t(r.expression().child()))));
}


#if 0

struct FnTraceMultiply
{
  PETE_EMPTY_CONSTRUCTORS(FnTraceMultiply)
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, FnTraceMultiply >::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
    cerr << "FnTraceMultiply()" << endl;
    return trace(a*b);
//    return traceMultiply(a,b);
  }
};


// traceMultiply(l,r)  <-  trace(l*r)
template<class T1,class T2,class CC>
inline typename MakeReturn<BinaryNode<FnTraceMultiply,T1,T2>,
  typename BinaryReturn<
    typename QDPContainer<T1>::Type_t,
    typename QDPContainer<T2>::Type_t,
    FnTraceMultiply>::Type_t >::Expression_t
trace(const QDPExpr<BinaryNode<OpMultiply,T1,T2>,CC> & ll)
{
  cerr << "trace(ll) -> traceMultiply(l,r)" << endl;

  typedef typename QDPContainer<T1>::Type_t  C1;
  typedef typename QDPContainer<T2>::Type_t  C2;
  typedef BinaryNode<FnTraceMultiply,T1,T2> Tree_t;
  typedef typename BinaryReturn<C1,C2,FnTraceMultiply>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    ll.expression().left(), 
    ll.expression().right()));
}

#endif


QDP_END_NAMESPACE();

