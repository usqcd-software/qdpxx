#ifndef QDP_CACHELOCK
#define QDP_CACHELOCK

#include "qdp.h"  

namespace QDP 
{

#include <PETE/ForEachTypeParser.h>

  struct LockTag
  {
    LockTag( QDPJitArgs& jitArgs):  jitArgs(jitArgs) {}

    const QDPJitArgs& getJitArgs() const {
      return jitArgs; 
    }

    template<class T>
    bool insertObject( const OScalar<T> & key ) const 
    {
      jitArgs.addPtr( key.getFdev() );
      return true;
    }
    template<class T>
    bool insertObject( const OLattice<T> & key ) const 
    {
      jitArgs.addPtr( key.getFdev() );
      return true;
    }

    bool insertObject(  const FnPeekColorMatrix & op ) const 
    {
      jitArgs.addInt( op.getRow() );
      jitArgs.addInt( op.getCol() );
      return true;
    }

    bool insertObject(  const FnPeekColorVector & op ) const 
    {
#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep("ParseTag::insertObject FnPeekColorVector");
#endif
      jitArgs.addInt( op.getRow() );
      return true;
    }
    bool insertObject(  const FnPeekSpinMatrix & op ) const 
    {
#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep("ParseTag::insertObject FnPeekSpinMatrix ");
#endif
      jitArgs.addInt( op.getRow() );
      jitArgs.addInt( op.getCol() );
      return true;
    }
    bool insertObject(  const FnPeekSpinVector & op ) const 
    {
#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep("ParseTag::insertObject FnPeekSpinVector");
#endif
      jitArgs.addInt( op.getRow() );
      return true;
    }
    bool insertObject(  const FnPokeColorMatrix & op ) const 
    {
#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep("ParseTag::insertObject FnPokeColorMatrix");
#endif
      jitArgs.addInt( op.getRow() );
      jitArgs.addInt( op.getCol() );
      return true;
    }
    bool insertObject(  const FnPokeColorVector & op ) const 
    {
#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep("ParseTag::insertObject FnPokeColorVector");
#endif
      jitArgs.addInt( op.getRow() );
      return true;
    }
    bool insertObject(  const FnPokeSpinMatrix & op ) const 
    {
#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep("ParseTag::insertObject FnPokeSpinMatrix");
#endif
      jitArgs.addInt( op.getRow() );
      jitArgs.addInt( op.getCol() );
      return true;
    }
    bool insertObject(  const FnPokeSpinVector & op ) const 
    {
#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep("ParseTag::insertObject FnPokeSpinVector");
#endif
      jitArgs.addInt( op.getRow() );
      return true;
    }


    template<int N>
    bool insertObject(  const GammaType<N> &s ) const 
    {
#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep("ParseTag::insertObject GammaType<N>");
#endif
      jitArgs.addInt( s.elem() );
      return true;
    }




  private:    
    const QDPJitArgs& jitArgs;
  };


 
  template<class RHS, class C1>
  bool cacheLock( 
		      const QDPExpr<RHS,C1>& rhs , 
		  
		      QDPJitArgs& cudaArgs )
  {
    typedef QDPExpr<RHS,C1>  Expr; 
    typedef typename CreateLeaf<Expr>::Leaf_t Expr_t; 
    const Expr_t &e = CreateLeaf<Expr>::make(rhs);
    typedef ForEachTypeParser<RHS, LockTag, LockTag, AndCombine > Print_t;

    LockTag pt(cudaArgs);
    bool ret = Print_t::apply(e, pt , pt , AndCombine());
    return ret;
  }



  template<class T, class C>
  bool cacheLock( 
		      const QDPType<T,C>& dest , 
		  
		      QDPJitArgs& cudaArgs )
  {
    LockTag pt(cudaArgs);
    bool ret = LeafFunctor<C,LockTag>::apply(static_cast<const C&>(dest), pt );
    return ret;
  }



  template<class T>
  bool cacheLockOp(
			const T& op , 
		
			QDPJitArgs& cudaArgs ) 
  {
    typedef OpVisitor<T, LockTag>          Visitor_t;
    LockTag pt(cudaArgs);
    bool ret = Visitor_t::visit(op,pt);
    return ret;
  }


  bool cacheLock( 
		      const Subset& s , 
		  
		      QDPJitArgs& cudaArgs );





  template<class T, class C>
  struct LeafFunctor<QDPType<T,C>, LockTag>
  {
    typedef bool Type_t;
    static bool apply(const QDPType<T,C> &s, const LockTag &f)
    { 
      return LeafFunctor<C,LockTag>::apply(static_cast<const C&>(s),f);
    }
  };


  template<class T>
  struct LeafFunctor<QDPType<T,OLattice<T> >, LockTag>
  {
    typedef bool Type_t;
    static bool apply(const QDPType<T,OLattice<T> > &s, const LockTag &f)
    { 
      return LeafFunctor<OLattice<T>,LockTag>::apply(static_cast < const OLattice<T> & > (s),f);
    }
  };


  template<class T>
  struct LeafFunctor<QDPType<T,OScalar<T> >, LockTag>
  {
    typedef bool Type_t;
    static bool apply(const QDPType<T,OScalar<T> > &s, const LockTag &f)
    { 
      return LeafFunctor<OScalar<T>,LockTag>::apply(static_cast<const OScalar<T>&>(s),f);
    }
  };


  template<class T, class CTag>
  struct ForEachTypeParser<Reference<T>,LockTag,LockTag,CTag>
  {
    typedef LeafFunctor<T,LockTag> Tag_t;
    typedef typename Tag_t::Type_t Type_t;
    
    static Type_t apply(const Reference<T> &expr, const LockTag &f,
			const LockTag &v, const CTag &c)
    {
      return Tag_t::apply(expr.reference(),f);
    }
  };




  template<class Op, class A, class CTag>
  struct ForEachTypeParser<UnaryNode<Op, A>, LockTag,LockTag, CTag>
  {
    typedef ForEachTypeParser<A, LockTag,LockTag, CTag> ForEachA_t;
    typedef OpVisitor<Op, LockTag>          Visitor_t;
    typedef typename ForEachA_t::Type_t   TypeA_t;
    typedef Combine1<TypeA_t, Op, CTag>   Combiner_t;
    typedef typename Combiner_t::Type_t   Type_t;
    static Type_t apply(const UnaryNode<Op, A> &expr, const LockTag &f, 
			const LockTag &v, const CTag &c)
    {
      if (!Visitor_t::visit(expr.operation(),f))
	return false;

      TypeA_t A_val  = ForEachA_t::apply(expr.child(), f, f, c);

      Type_t val = Combiner_t::combine(A_val, expr.operation(), c);

      return val;
    }
  };


  template<class A, class CTag>
  struct ForEachTypeParser<UnaryNode<OpIdentity, A>, LockTag,LockTag, CTag>
  {
    typedef ForEachTypeParser<A, LockTag,LockTag, CTag> ForEachA_t;
    typedef OpVisitor<OpIdentity, LockTag>          Visitor_t;
    typedef typename ForEachA_t::Type_t   TypeA_t;
    typedef Combine1<TypeA_t, OpIdentity, CTag>   Combiner_t;
    typedef typename Combiner_t::Type_t   Type_t;
    static Type_t apply(const UnaryNode<OpIdentity, A> &expr, const LockTag &f, 
			const LockTag &v, const CTag &c)
    {
      TypeA_t A_val  = ForEachA_t::apply(expr.child(), f, f, c);
      Type_t val = Combiner_t::combine(A_val, expr.operation(), c);

      return val;
    }
  };


  template<class A, class CTag>
  struct ForEachTypeParser<UnaryNode<FnMap, A>, LockTag,LockTag, CTag>
  {

    typedef ForEachTypeParser<A, LockTag,LockTag, CTag> ForEachA_t;
    typedef OpVisitor<FnMap, LockTag>          Visitor_t;
    typedef typename ForEachA_t::Type_t   TypeA_t;
    typedef Combine1<TypeA_t, FnMap, CTag>   Combiner_t;
    typedef typename Combiner_t::Type_t   Type_t;

    typedef typename ForEach<A, EvalLeaf1, OpCombine>::Type_t AInnerTypeA_t;
    typedef typename Combine1<AInnerTypeA_t, FnMap, OpCombine>::Type_t InnerTypeBB_t;

    static Type_t apply(const UnaryNode<FnMap, A> &expr, const LockTag &f, 
			const LockTag &v, const CTag &c)
    {
      const Map& map = expr.operation().map;
      FnMap& fnmap = const_cast<FnMap&>(expr.operation());

      // get goffsets[] on device
      //
      int goffsetsId = expr.operation().map.getGoffsetsId();
      void * goffsetsDev = QDPCache::Instance().getDevicePtr( goffsetsId );
      int posGoff = f.getJitArgs().addPtr( goffsetsDev );

      // get receive buffer on device and NULL otherwise
      //
      int posRcvBuf;
      if (map.hasOffnode()) {
	const FnMapRsrc& rRSrc = fnmap.getCached();
	int rcvId = rRSrc.getRcvId();
	void * rcvBufDev = QDPCache::Instance().getDevicePtr( rcvId );
	posRcvBuf = f.getJitArgs().addPtr( rcvBufDev );
      } else {
	posRcvBuf = f.getJitArgs().addPtr( NULL );
      }

      TypeA_t A_val  = ForEachA_t::apply(expr.child(), f, f, c);
      Type_t val = Combiner_t::combine(A_val, expr.operation(), c);

      //LockTag ff( f.getJitArgs() , newIdx.str() );
      //TypeA_t A_val  = ForEachA_t::apply(expr.child(), ff, ff, c);
      //Type_t val = Combiner_t::combine(A_val, expr.operation(), c);
      //f.ossCode << ff.ossCode.str();

      return val;
    }
  };


  template<class Op, class A, class B, class CTag>
  struct ForEachTypeParser<BinaryNode<Op, A, B>, LockTag, LockTag, CTag>
  {
    typedef ForEachTypeParser<A, LockTag, LockTag, CTag> ForEachA_t;
    typedef ForEachTypeParser<B, LockTag, LockTag, CTag> ForEachB_t;
    typedef OpVisitor<Op, LockTag>                Visitor_t;

    typedef typename ForEachA_t::Type_t  TypeA_t;
    typedef typename ForEachB_t::Type_t  TypeB_t;

    typedef Combine2<TypeA_t, TypeB_t, Op, CTag>  Combiner_t;

    typedef typename Combiner_t::Type_t Type_t;

    static Type_t apply(const BinaryNode<Op, A, B> &expr, const LockTag &f, 
			const LockTag &v, const CTag &c) 
    {
      if (!Visitor_t::visit(expr.operation(),f))
	return false;


      TypeA_t left_val  = ForEachA_t::apply(expr.left(), f, f, c);
      TypeB_t right_val = ForEachB_t::apply(expr.right(), f, f, c);

      Type_t val = Combiner_t::combine(left_val, right_val, expr.operation(), c);

      return val;
    }
  };


  template<class Op, class A, class B, class C, class CTag>
  struct ForEachTypeParser<TrinaryNode<Op, A, B, C>, LockTag, LockTag, CTag>
  {
    typedef ForEachTypeParser<A, LockTag, LockTag, CTag> ForEachA_t;
    typedef ForEachTypeParser<B, LockTag, LockTag, CTag> ForEachB_t;
    typedef ForEachTypeParser<C, LockTag, LockTag, CTag> ForEachC_t;
    typedef OpVisitor<Op, LockTag>                Visitor_t;

    typedef typename ForEachA_t::Type_t  TypeA_t;
    typedef typename ForEachB_t::Type_t  TypeB_t;
    typedef typename ForEachC_t::Type_t  TypeC_t;

    typedef Combine3<TypeA_t, TypeB_t, TypeC_t, Op, CTag> Combiner_t;

    typedef typename Combiner_t::Type_t Type_t;

    static Type_t apply(const TrinaryNode<Op, A, B, C> &expr, const LockTag &f, 
			const LockTag &v, const CTag &c) 
    {
      if (!Visitor_t::visit(expr.operation(),f))
	return false;

      TypeA_t left_val  = ForEachA_t::apply(expr.left(), f, f, c);
      TypeB_t middle_val = ForEachB_t::apply(expr.middle(), f, f, c);
      TypeB_t right_val = ForEachC_t::apply(expr.right(), f, f, c);

      Type_t val = Combiner_t::combine(left_val, middle_val, right_val, expr.operation(), c);

      return val;
    }
  };





  template<>
  struct LeafFunctor<float, LockTag>
  {
    typedef bool Type_t;
    static bool apply(const float &s, const LockTag &f)
    { 
      return true;
    }
  };

  template<>
  struct LeafFunctor<double, LockTag>
  {
    typedef bool Type_t;
    static bool apply(const double &s, const LockTag &f)
    { 
      return true;
    }
  };

  template<>
  struct LeafFunctor<int, LockTag>
  {
    typedef bool Type_t;
    static bool apply(const int &s, const LockTag &f)
    { 
      return true;
    }
  };

  template<>
  struct LeafFunctor<char, LockTag>
  {
    typedef bool Type_t;
    static bool apply(const char &s, const LockTag &f)
    { 
      return true;
    }
  };

  template<>
  struct LeafFunctor<bool, LockTag>
  {
    typedef bool Type_t;
    static bool apply(const bool &s, const LockTag &f)
    { 
      return true;
    }
  };

  template<class T>
  struct LeafFunctor<IScalar<T>, LockTag>
  {
    typedef bool Type_t;
    static bool apply(const IScalar<T> &s, const LockTag &f)
    { 
      LeafFunctor<T,LockTag>::apply(s.elem(),f);
      return true;
    }
  };

  template<class T, int N>
  struct LeafFunctor<ILattice<T,N>, LockTag>
  {
    typedef bool Type_t;
    static bool apply(const ILattice<T,N> &s, const LockTag &f)
    { 
      LeafFunctor<T,LockTag>::apply(s.elem(0),f);
      return true;
    }
  };

  template<class T>
  struct LeafFunctor<RScalar<T>, LockTag>
  {
    typedef bool Type_t;
    static bool apply(const RScalar<T> &s, const LockTag &f)
    { 
      LeafFunctor<T,LockTag>::apply(s.elem(),f);
      return true;
    }
  };

  template<class T>
  struct LeafFunctor<RComplex<T>, LockTag>
  {
    typedef bool Type_t;
    static bool apply(const RComplex<T> &s, const LockTag &f)
    { 
      LeafFunctor<T,LockTag>::apply(s.real(),f);
      return true;
    }
  };

  template<class T>
  struct LeafFunctor<PScalar<T>, LockTag>
  {
    typedef bool Type_t;
    static bool apply(const PScalar<T> &s, const LockTag &f)
    { 
      LeafFunctor<T,LockTag>::apply(s.elem(),f);
      return true;
    }
  };

  template<class T>
  struct LeafFunctor<PSeed<T>, LockTag>
  {
    typedef bool Type_t;
    static bool apply(const PSeed<T> &s, const LockTag &f)
    { 
      LeafFunctor<T,LockTag>::apply(s.elem(0),f);
      return true;
    }
  };

  template <class T, int N>
  struct LeafFunctor<PColorMatrix<T,N>, LockTag>
  {
    typedef bool Type_t;
    static bool apply(const PColorMatrix<T,N> &s, const LockTag &f)
    { 
      LeafFunctor<T,LockTag>::apply(s.elem(0,0),f);
      return true;
    }
  };

  template <class T, int N>
  struct LeafFunctor<PSpinMatrix<T,N>, LockTag>
  {
    typedef bool Type_t;
    static bool apply(const PSpinMatrix<T,N> &s, const LockTag &f)
    { 
      LeafFunctor<T,LockTag>::apply(s.elem(0,0),f);
      return true;
    }
  };

  template <class T, int N>
  struct LeafFunctor<PColorVector<T,N>, LockTag>
  {
    typedef bool Type_t;
    static bool apply(const PColorVector<T,N> &s, const LockTag &f)
    { 
      LeafFunctor<T,LockTag>::apply(s.elem(0),f);
      return true;
    }
  };

  template <class T, int N>
  struct LeafFunctor<PSpinVector<T,N>, LockTag>
  {
    typedef bool Type_t;
    static bool apply(const PSpinVector<T,N> &s, const LockTag &f)
    { 
      LeafFunctor<T,LockTag>::apply(s.elem(0),f);
      return true;
    }
  };

  template <class T, int N, template<class,int> class C>
  struct LeafFunctor<PMatrix<T,N,C>, LockTag>
  {
    typedef bool Type_t;
    static bool apply(const PMatrix<T,N,C> &s, const LockTag &f)
    { 
      return LeafFunctor<C<T,N>,LockTag>::apply(static_cast<const C<T,N>&>(s),f);
    }
  };

  template <class T, int N, template<class,int> class C>
  struct LeafFunctor<PVector<T,N,C>, LockTag>
  {
    typedef bool Type_t;
    static bool apply(const PVector<T,N,C> &s, const LockTag &f)
    { 
      return LeafFunctor<C<T,N>,LockTag>::apply(static_cast<const C<T,N>&>(s),f);
    }
  };

  template<class T>
  struct LeafFunctor<OScalar<T>, LockTag>
  {
    typedef bool Type_t;
    static bool apply(const OScalar<T> &s, const LockTag &f)
    { 
      string strIdentifier;

      if (f.insertObject( s )) {
	return true;
      } else {
	return false;
      }
    }
  };

  template<class T>
  struct LeafFunctor<OLattice<T>, LockTag>
  {
    typedef bool Type_t;
    static bool apply(const OLattice<T> &s, const LockTag &f)
    { 
      string strIdentifier;

      if (f.insertObject(  s )) {
	return true;
      } else {
	return false;
      }
    }
  };

  template<int N>
  struct LeafFunctor<GammaType<N>, LockTag>
  {
    typedef bool Type_t;
    static bool apply(const GammaType<N> &s, const LockTag &f)
    { 
      string strIdentifier;
      
      ostringstream ossTmp;
      ossTmp << "GammaType<" << N << ">";
      string strType = ossTmp.str();
      
      if (f.insertObject(  s )) {
	return true;
      } else {
	return false;
      }

    }
  };

  template<int N, int m>
  struct LeafFunctor<GammaConst<N,m>, LockTag>
  {
    typedef bool Type_t;
    static bool apply(const GammaConst<N,m> &s, const LockTag &f)
    { 
      return true;
    }
  };



#include"qdp_cachelock_peteops.h"
#include"qdp_cachelock_qdpops.h"



  //-----------------------------------------------------------------------------
  // Additional operator tags 
  //-----------------------------------------------------------------------------



  template <>
  struct OpVisitor<FnPeekColorMatrix, LockTag> 
  { 
    static bool visit(const FnPeekColorMatrix & op, const LockTag & f) 
    { 
      string strIdentifier;
      if (f.insertObject( op )) {
	return true;
      } else {
	return false;
      }
    }
  };

  template <>
  struct OpVisitor<FnPeekColorVector, LockTag> 
  { 
    static bool visit(const FnPeekColorVector & op, const LockTag & f) 
    { 
      string strIdentifier;
      if (f.insertObject( op )) {
	return true;
      } else {
	return false;
      }
    }
  };

  template <>
  struct OpVisitor<FnPeekSpinMatrix, LockTag> 
  { 
    static bool visit(const FnPeekSpinMatrix & op, const LockTag & f) 
    { 
      string strIdentifier;
      if (f.insertObject( op )) {
	return true;
      } else {
	return false;
      }
    }
  };

  template <>
  struct OpVisitor<FnPeekSpinVector, LockTag> 
  { 
    static bool visit(const FnPeekSpinVector & op, const LockTag & f) 
    { 
      string strIdentifier;
      if (f.insertObject( op )) {
	return true;
      } else {
	return false;
      }
    }
  };


  template <>
  struct OpVisitor<FnPokeColorMatrix, LockTag> 
  { 
    static bool visit(const FnPokeColorMatrix & op, const LockTag & f) 
    { 
      string strIdentifier;
      if (f.insertObject( op )) {
	return true;
      } else {
	return false;
      }
    }
  };

  template <>
  struct OpVisitor<FnPokeColorVector, LockTag> 
  { 
    static bool visit(const FnPokeColorVector & op, const LockTag & f) 
    { 
      string strIdentifier;
      if (f.insertObject( op )) {
	return true;
      } else {
	return false;
      }
    }
  };

  template <>
  struct OpVisitor<FnPokeSpinMatrix, LockTag> 
  { 
    static bool visit(const FnPokeSpinMatrix & op, const LockTag & f) 
    { 
      string strIdentifier;
      if (f.insertObject( op )) {
	return true;
      } else {
	return false;
      }
    }
  };

  template <>
  struct OpVisitor<FnPokeSpinVector, LockTag> 
  { 
    static bool visit(const FnPokeSpinVector & op, const LockTag & f) 
    { 
      string strIdentifier;
      if (f.insertObject( op )) {
	return true;
      } else {
	return false;
      }
    }
  };


  template <>
  struct OpVisitor<FnSum, LockTag> : public BracketPrinter<FnSum>
  { 
    static bool visit(FnSum op, const LockTag & t) 
    { return true; }
  };

  template <>
  struct OpVisitor<FnNorm2, LockTag> : public BracketPrinter<FnNorm2>
  { 
    static bool visit(FnNorm2 op, const LockTag & t) 
    { return true; }
  };

  template <>
  struct OpVisitor<OpGammaConstMultiply, LockTag> : public BracketPrinter<OpGammaConstMultiply>
  { 
    static bool visit(OpGammaConstMultiply op, const LockTag & t) 
    { return true; }
  };

  template <>
  struct OpVisitor<OpMultiplyGammaConst, LockTag> : public BracketPrinter<OpMultiplyGammaConst>
  { 
    static bool visit(OpMultiplyGammaConst op, const LockTag & t) 
    { return true; }
  };

  template <>
  struct OpVisitor<OpGammaTypeMultiply, LockTag> : public BracketPrinter<OpGammaTypeMultiply>
  { 
    static bool visit(OpGammaTypeMultiply op, const LockTag & t) 
    { return true; }
  };

  template <>
  struct OpVisitor<OpMultiplyGammaType, LockTag> : public BracketPrinter<OpMultiplyGammaType>
  { 
    static bool visit(OpMultiplyGammaType op, const LockTag & t) 
    { return true; }
  };


  //------------------------------------------------------------------------
  // Special optimizations
  //------------------------------------------------------------------------

  template <>
  struct OpVisitor<OpAdjMultiply, LockTag> : public BracketPrinter<OpAdjMultiply>
  { 
    static bool visit(OpAdjMultiply op, const LockTag & t) 
    { return true; }
  };

  template <>
  struct OpVisitor<OpMultiplyAdj, LockTag> : public BracketPrinter<OpMultiplyAdj>
  { 
    static bool visit(OpMultiplyAdj op, const LockTag & t) 
    { return true; }
  };

  template <>
  struct OpVisitor<OpAdjMultiplyAdj, LockTag> : public BracketPrinter<OpAdjMultiplyAdj>
  { 
    static bool visit(OpAdjMultiplyAdj op, const LockTag & t) 
    { return true; }
  };

  template <>
  struct OpVisitor<FnTraceMultiply, LockTag> : public BracketPrinter<FnTraceMultiply>
  { 
    static bool visit(FnTraceMultiply op, const LockTag & t) 
    { return true; }
  };

  template <>
  struct OpVisitor<FnTraceColorMultiply, LockTag> : public BracketPrinter<FnTraceColorMultiply>
  { 
    static bool visit(FnTraceColorMultiply op, const LockTag & t) 
    { return true; }
  };

  template <>
  struct OpVisitor<FnTraceSpinMultiply, LockTag> : public BracketPrinter<FnTraceSpinMultiply>
  { 
    static bool visit(FnTraceSpinMultiply op, const LockTag & t) 
    { return true; }
  };



  template <>
  struct OpVisitor<FnTraceSpinOuterProduct, LockTag> : public BracketPrinter<FnTraceSpinOuterProduct>
  { 
    static bool visit(FnTraceSpinOuterProduct op, const LockTag & t) 
    { return true; }
  };


  template <>
  struct OpVisitor<FnTraceSpinQuarkContract13, LockTag> : public BracketPrinter<FnTraceSpinQuarkContract13>
  { 
    static bool visit(FnTraceSpinQuarkContract13 op, const LockTag & t) 
    { return true; }
  };


} 

#endif  
