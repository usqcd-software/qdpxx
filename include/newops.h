// -*- C++ -*-
// $Id: newops.h,v 1.3 2002-10-12 00:58:32 edwards Exp $
//
// QDP data parallel interface
//

QDP_BEGIN_NAMESPACE(QDP);


//-----------------------------------------------------------------------------
// Operator tags that are only used for type resolution
//-----------------------------------------------------------------------------

struct FnSpinProject
{
  PETE_EMPTY_CONSTRUCTORS(FnSpinProject)
};

struct FnSpinReconstruct
{
  PETE_EMPTY_CONSTRUCTORS(FnSpinReconstruct)
};

struct FnQuarkContractXX
{
  PETE_EMPTY_CONSTRUCTORS(FnQuarkContractXX)
};

struct FnSum
{
  PETE_EMPTY_CONSTRUCTORS(FnSum)
};

struct FnNorm2
{
  PETE_EMPTY_CONSTRUCTORS(FnNorm2)
};

struct FnInnerproduct
{
  PETE_EMPTY_CONSTRUCTORS(FnInnerproduct)
};

struct FnInnerproductReal
{
  PETE_EMPTY_CONSTRUCTORS(FnInnerproductReal)
};

struct FnSliceSum
{
  PETE_EMPTY_CONSTRUCTORS(FnSliceSum)
};

struct FnSumMulti
{
  PETE_EMPTY_CONSTRUCTORS(FnSumMulti)
};

struct FnNorm2Multi
{
  PETE_EMPTY_CONSTRUCTORS(FnNorm2Multi)
};

struct FnInnerproductMulti
{
  PETE_EMPTY_CONSTRUCTORS(FnInnerproduct)
};

struct FnInnerproductRealMulti
{
  PETE_EMPTY_CONSTRUCTORS(FnInnerproductReal)
};


//-----------------------------------------------------------------------------
// Additional operator tags 
//-----------------------------------------------------------------------------

struct OpGammaConstMultiply
{
  PETE_EMPTY_CONSTRUCTORS(OpGammaConstMultiply)
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, OpGammaConstMultiply >::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
    return (a * b);
  }
};


struct OpMultiplyGammaConst
{
  PETE_EMPTY_CONSTRUCTORS(OpMultiplyGammaConst)
  template<class T1, class T2>
  inline typename BinaryReturn<T1, T2, OpMultiplyGammaConst >::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
    return (a * b);
  }
};


// Member function definition in primgamma.h
struct OpGammaTypeMultiply
{
  PETE_EMPTY_CONSTRUCTORS(OpGammaTypeMultiply)
  template<int N, class T>
  inline T
  operator()(const GammaType<N>& a, const T &b) const;
};


// Member function definition in primgamma.h
struct OpMultiplyGammaType
{
  PETE_EMPTY_CONSTRUCTORS(OpMultiplyGammaType)
  template<class T, int N>
  inline T
  operator()(const T &a, const GammaType<N>& b) const;
};


//-----------------------------------------------------------------------------
// Leaf stuff
//-----------------------------------------------------------------------------
#if 1
template<int N>
struct CreateLeaf<GammaType<N> >
{
  typedef GammaType<N> Input_t;
  typedef Input_t Leaf_t;
//  typedef Reference<Input_t> Leaf_t;

  inline static
  Leaf_t make(const Input_t& a) { return Leaf_t(a); }
};

template<int N>
struct LeafFunctor<GammaType<N>, ElemLeaf>
{
  typedef GammaType<N> Type_t;
  inline static Type_t apply(const GammaType<N> &a, const ElemLeaf &f)
    {return a;}
};

template<int N>
struct LeafFunctor<GammaType<N>, EvalLeaf1>
{
  typedef GammaType<N> Type_t;
  inline static Type_t apply(const GammaType<N> &a, const EvalLeaf1 &f)
    {return a;}
};

#endif


//-----------------------------------------------------------------------------
// Additional operators
//-----------------------------------------------------------------------------

/*! @addtogroup group1 */
/*! @{ */
 
// GammaConst * QDPType
template<int N,int m,class T2,class C2>
inline typename MakeReturn<BinaryNode<OpGammaConstMultiply,
  typename CreateLeaf<GammaConst<N,m> >::Leaf_t,
  typename CreateLeaf<QDPType<T2,C2> >::Leaf_t>,
  typename BinaryReturn<GammaConst<N,m>,C2,OpGammaConstMultiply>::Type_t >::Expression_t
operator*(const GammaConst<N,m> & l,const QDPType<T2,C2> & r)
{
  typedef BinaryNode<OpGammaConstMultiply,
    typename CreateLeaf<GammaConst<N,m> >::Leaf_t,
    typename CreateLeaf<QDPType<T2,C2> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<GammaConst<N,m>,C2,OpGammaConstMultiply>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<GammaConst<N,m> >::make(l),
    CreateLeaf<QDPType<T2,C2> >::make(r)));
}

// GammaConst * QDPType
template<int N,int m,class T2,class C2>
inline typename MakeReturn<BinaryNode<OpGammaConstMultiply,
  typename CreateLeaf<GammaConst<N,m> >::Leaf_t,
  typename CreateLeaf<QDPExpr<T2,C2> >::Leaf_t>,
  typename BinaryReturn<GammaConst<N,m>,C2,OpGammaConstMultiply>::Type_t >::Expression_t
operator*(const GammaConst<N,m> & l,const QDPExpr<T2,C2> & r)
{
  typedef BinaryNode<OpGammaConstMultiply,
    typename CreateLeaf<GammaConst<N,m> >::Leaf_t,
    typename CreateLeaf<QDPExpr<T2,C2> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<GammaConst<N,m>,C2,OpGammaConstMultiply>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<GammaConst<N,m> >::make(l),
    CreateLeaf<QDPExpr<T2,C2> >::make(r)));
}

// QDPType * GammaConst
template<class T1,class C1,int N,int m>
inline typename MakeReturn<BinaryNode<OpMultiplyGammaConst,
  typename CreateLeaf<QDPType<T1,C1> >::Leaf_t,
  typename CreateLeaf<GammaConst<N,m> >::Leaf_t>,
  typename BinaryReturn<C1,GammaConst<N,m>,OpMultiplyGammaConst>::Type_t >::Expression_t
operator*(const QDPType<T1,C1> & l,const GammaConst<N,m> & r)
{
  typedef BinaryNode<OpMultiplyGammaConst,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t,
    typename CreateLeaf<GammaConst<N,m> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<C1,GammaConst<N,m>,OpMultiplyGammaConst>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<QDPType<T1,C1> >::make(l),
    CreateLeaf<GammaConst<N,m> >::make(r)));
}

// QDPType * GammaConst
template<class T1,class C1,int N,int m>
inline typename MakeReturn<BinaryNode<OpMultiplyGammaConst,
  typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t,
  typename CreateLeaf<GammaConst<N,m> >::Leaf_t>,
  typename BinaryReturn<C1,GammaConst<N,m>,OpMultiplyGammaConst>::Type_t >::Expression_t
operator*(const QDPExpr<T1,C1> & l,const GammaConst<N,m> & r)
{
  typedef BinaryNode<OpMultiplyGammaConst,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t,
    typename CreateLeaf<GammaConst<N,m> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<C1,GammaConst<N,m>,OpMultiplyGammaConst>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<QDPExpr<T1,C1> >::make(l),
    CreateLeaf<GammaConst<N,m> >::make(r)));
}


// GammaType * QDPType
template<int N,class T2,class C2>
inline typename MakeReturn<BinaryNode<OpGammaTypeMultiply,
  typename CreateLeaf<GammaType<N> >::Leaf_t,
  typename CreateLeaf<QDPType<T2,C2> >::Leaf_t>,
  typename BinaryReturn<GammaType<N>,C2,OpGammaTypeMultiply>::Type_t >::Expression_t
operator*(const GammaType<N> & l,const QDPType<T2,C2> & r)
{
  typedef BinaryNode<OpGammaTypeMultiply,
    typename CreateLeaf<GammaType<N> >::Leaf_t,
    typename CreateLeaf<QDPType<T2,C2> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<GammaType<N>,C2,OpGammaTypeMultiply>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<GammaType<N> >::make(l),
    CreateLeaf<QDPType<T2,C2> >::make(r)));
}

// GammaType * QDPType
template<int N,class T2,class C2>
inline typename MakeReturn<BinaryNode<OpGammaTypeMultiply,
  typename CreateLeaf<GammaType<N> >::Leaf_t,
  typename CreateLeaf<QDPExpr<T2,C2> >::Leaf_t>,
  typename BinaryReturn<GammaType<N>,C2,OpGammaTypeMultiply>::Type_t >::Expression_t
operator*(const GammaType<N> & l,const QDPExpr<T2,C2> & r)
{
  typedef BinaryNode<OpGammaTypeMultiply,
    typename CreateLeaf<GammaType<N> >::Leaf_t,
    typename CreateLeaf<QDPExpr<T2,C2> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<GammaType<N>,C2,OpGammaTypeMultiply>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<GammaType<N> >::make(l),
    CreateLeaf<QDPExpr<T2,C2> >::make(r)));
}

// QDPType * GammaType
template<class T1,class C1,int N>
inline typename MakeReturn<BinaryNode<OpMultiplyGammaType,
  typename CreateLeaf<QDPType<T1,C1> >::Leaf_t,
  typename CreateLeaf<GammaType<N> >::Leaf_t>,
  typename BinaryReturn<C1,GammaType<N>,OpMultiplyGammaType>::Type_t >::Expression_t
operator*(const QDPType<T1,C1> & l,const GammaType<N> & r)
{
  typedef BinaryNode<OpMultiplyGammaType,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t,
    typename CreateLeaf<GammaType<N> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<C1,GammaType<N>,OpMultiplyGammaType>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<QDPType<T1,C1> >::make(l),
    CreateLeaf<GammaType<N> >::make(r)));
}

// QDPType * GammaType
template<class T1,class C1,int N>
inline typename MakeReturn<BinaryNode<OpMultiplyGammaType,
  typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t,
  typename CreateLeaf<GammaType<N> >::Leaf_t>,
  typename BinaryReturn<C1,GammaType<N>,OpMultiplyGammaType>::Type_t >::Expression_t
operator*(const QDPExpr<T1,C1> & l,const GammaType<N> & r)
{
  typedef BinaryNode<OpMultiplyGammaType,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t,
    typename CreateLeaf<GammaType<N> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<C1,GammaType<N>,OpMultiplyGammaType>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<QDPExpr<T1,C1> >::make(l),
    CreateLeaf<GammaType<N> >::make(r)));
}
/*! @} */ // end of group1


// Explicit casts
template<class T1,class T2,class C2>
inline typename MakeReturn<UnaryNode<OpCast<T1>,
  typename CreateLeaf<QDPType<T2,C2> >::Leaf_t>,
  typename UnaryReturn<C2,OpCast<T1> >::Type_t>::Expression_t
peteCast(const T1&, const QDPType<T2,C2>& l)
{
  typedef UnaryNode<OpCast<T1>,
    typename CreateLeaf<QDPType<T2,C2> >::Leaf_t> Tree_t;
  typedef typename UnaryReturn<C2,OpCast<T1> >::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<QDPType<T2,C2> >::make(l)));
}


QDP_END_NAMESPACE();

