#ifndef QDP_TYPEPARSER
#define QDP_TYPEPARSER

#include "qdp.h"  
#include <sstream>  

namespace QDP 
{

#include <PETE/ForEachTypeParser.h>


  struct ParseTag
  {
    ParseTag( const QDPJitArgs& jitArgs , const string& idxName);
    const QDPJitArgs& getJitArgs() const;

    template<class T>
    bool insertObject( string& strIdentifier , const string& strType, const OScalar<T> & key ) const 
    {
#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep("ParseTag::insertObject OScalar<T>");
#endif
      int pos = jitArgs.addPtr( key.getFdev() );
      ostringstream code;
      code << " *(("<< strType <<"*)("<< jitArgs.getPtrName() << "[ " << pos  << " ].ptr " << ")) ";
      strIdentifier = code.str();
      return true;
    }

    template<class T>
    bool insertObject( string& strIdentifier , const string& strType, const OLattice<T> & key ) const 
    {
#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep("ParseTag::insertObject OLattice<T>");
#endif
      int pos = jitArgs.addPtr( key.getFdev() );
      ostringstream code;
      code << " (("<< strType <<"*)("<< jitArgs.getPtrName() << "[ " << pos  << " ].ptr " << "))[" << stringIdx << "] ";
      strIdentifier = code.str();
      return true;
    }


    bool insertObject( string& strIdentifier , const string& strType, const FnPeekColorMatrix & op ) const;
    bool insertObject( string& strIdentifier , const string& strType, const FnPeekColorVector & op ) const;
    bool insertObject( string& strIdentifier , const string& strType, const FnPeekSpinMatrix & op ) const;
    bool insertObject( string& strIdentifier , const string& strType, const FnPeekSpinVector & op ) const;
    bool insertObject( string& strIdentifier , const string& strType, const FnPokeColorMatrix & op ) const;
    bool insertObject( string& strIdentifier , const string& strType, const FnPokeColorVector & op ) const;
    bool insertObject( string& strIdentifier , const string& strType, const FnPokeSpinMatrix & op ) const;
    bool insertObject( string& strIdentifier , const string& strType, const FnPokeSpinVector & op ) const;


    template<int N>
    bool insertObject( string& strIdentifier , const string& strType, const GammaType<N> &s ) const 
    {
#ifdef GPU_DEBUG_DEEP
      QDP_debug_deep("ParseTag::insertObject GammaType<N>");
#endif
      int pos_m = jitArgs.addInt( s.elem() );
      ostringstream code;
      code << strType << "(" 
	   << jitArgs.getPtrName() << "[ " << pos_m  << " ].Int " 
	   << ")";
      strIdentifier = code.str();
      return true;
    }

    string getIndex() const;

    mutable ostringstream ossCode;

    // const QDPJitArgs& getJitArgs() const { return jitArgs; }
    // const string& getStringIdx() const { return stringIdx; }

  private:    
    const QDPJitArgs& jitArgs;
    string stringIdx;

  };


 
  template<class RHS, class C1>
  bool getCodeString( string& codeString,
		      const QDPExpr<RHS,C1>& rhs , 
		      const string& idx , 
		      QDPJitArgs& cudaArgs )
  {
    typedef QDPExpr<RHS,C1>  Expr; 
    typedef typename CreateLeaf<Expr>::Leaf_t Expr_t; 
    const Expr_t &e = CreateLeaf<Expr>::make(rhs);
    typedef ForEachTypeParser<RHS, ParseTag, ParseTag, AndCombine > Print_t;

    ParseTag pt(cudaArgs,idx);
    bool ret = Print_t::apply(e, pt , pt , AndCombine());
    codeString = pt.ossCode.str();
    return ret;
  }



  template<class T, class C>
  bool getCodeString( string& codeString , 
		      const QDPType<T,C>& dest , 
		      const string& idx , 
		      QDPJitArgs& cudaArgs )
  {
    ParseTag pt(cudaArgs,idx);
    bool ret = LeafFunctor<C,ParseTag>::apply(static_cast<const C&>(dest), pt );
    codeString = pt.ossCode.str();
    return ret;
  }



  template<class T>
  bool getCodeStringOp( string& codeString , 
			const T& op , 
			const string& idx , 
			QDPJitArgs& cudaArgs ) 
  {
    typedef OpVisitor<T, ParseTag>          Visitor_t;
    ParseTag pt(cudaArgs,idx);
    bool ret = Visitor_t::visit(op,pt);
    codeString = pt.ossCode.str();
    return ret;
  }


  void getTypeString( string& typeString , const float& l );
  void getTypeString( string& typeString , const double& l );



  template<class T>
  void getTypeString( string& typeString , const OLattice<T>& l , QDPJitArgs& cudaArgs )
  {
    ParseTag pt(cudaArgs,"nothing");
    LeafFunctor<T,ParseTag>::apply(T(),pt);
    typeString = pt.ossCode.str();
  }

  template<class T>
  void getTypeString( string& typeString , const OScalar<T>& l , QDPJitArgs& cudaArgs )
  {
    ParseTag pt(cudaArgs,"nothing");
    LeafFunctor<T,ParseTag>::apply(T(),pt);
    typeString = pt.ossCode.str();
  }


  template<typename T>
  void getTypeStringT(string& s,const QDPJitArgs& cudaArgs) {
    T l;
    ParseTag pt( cudaArgs , "nothing");
    LeafFunctor<T,ParseTag>::apply( l ,pt);
    s = pt.ossCode.str();
  }

  template<class T>
  void getTypeString( string& typeString , const PScalar<T>& l , QDPJitArgs& cudaArgs )
  {
    ParseTag pt(cudaArgs,"nothing");
    LeafFunctor<PScalar<T>,ParseTag>::apply( l ,pt);
    typeString = pt.ossCode.str();
  }

  template<class T,int N>
    void getTypeString( string& typeString , const PSpinVector<T,N>& l , QDPJitArgs& cudaArgs )
  {
    ParseTag pt(cudaArgs,"nothing");
    LeafFunctor<PSpinVector<T,N>,ParseTag>::apply( l ,pt);
    typeString = pt.ossCode.str();
  }

  template<class T,int N>
    void getTypeString( string& typeString , const PSpinMatrix<T,N>& l , QDPJitArgs& cudaArgs )
  {
    ParseTag pt(cudaArgs,"nothing");
    LeafFunctor<PSpinMatrix<T,N>,ParseTag>::apply( l ,pt);
    typeString = pt.ossCode.str();
  }



  template<class T, class C>
  struct LeafFunctor<QDPType<T,C>, ParseTag>
  {
    typedef bool Type_t;
    static bool apply(const QDPType<T,C> &s, const ParseTag &f)
    { 
      return LeafFunctor<C,ParseTag>::apply(static_cast<const C&>(s),f);
    }
  };


  template<class T>
  struct LeafFunctor<QDPType<T,OLattice<T> >, ParseTag>
  {
    typedef bool Type_t;
    static bool apply(const QDPType<T,OLattice<T> > &s, const ParseTag &f)
    { 
      return LeafFunctor<OLattice<T>,ParseTag>::apply(static_cast < const OLattice<T> & > (s),f);
    }
  };

  template<class T>
  struct LeafFunctor<QDPType<T,OScalar<T> >, ParseTag>
  {
    typedef bool Type_t;
    static bool apply(const QDPType<T,OScalar<T> > &s, const ParseTag &f)
    { 
      return LeafFunctor<OScalar<T>,ParseTag>::apply(static_cast<const OScalar<T>&>(s),f);
    }
  };


  template<class T, class CTag>
  struct ForEachTypeParser<Reference<T>,ParseTag,ParseTag,CTag>
  {
    typedef LeafFunctor<T,ParseTag> Tag_t;
    typedef typename Tag_t::Type_t Type_t;
    
    static Type_t apply(const Reference<T> &expr, const ParseTag &f,
			const ParseTag &v, const CTag &c)
    {
      return Tag_t::apply(expr.reference(),f);
      //return 0;
    }
  };




  template<class Op, class A, class CTag>
  struct ForEachTypeParser<UnaryNode<Op, A>, ParseTag,ParseTag, CTag>
  {
    typedef ForEachTypeParser<A, ParseTag,ParseTag, CTag> ForEachA_t;
    typedef OpVisitor<Op, ParseTag>          Visitor_t;
    typedef typename ForEachA_t::Type_t   TypeA_t;
    typedef Combine1<TypeA_t, Op, CTag>   Combiner_t;
    typedef typename Combiner_t::Type_t   Type_t;
    static Type_t apply(const UnaryNode<Op, A> &expr, const ParseTag &f, 
			const ParseTag &v, const CTag &c)
    {
      if (!Visitor_t::visit(expr.operation(),f))
	return false;

      f.ossCode << "(";
      TypeA_t A_val  = ForEachA_t::apply(expr.child(), f, f, c);

      f.ossCode << ")";

      Type_t val = Combiner_t::combine(A_val, expr.operation(), c);

      return val;
    }
  };


  template<class A, class CTag>
  struct ForEachTypeParser<UnaryNode<OpIdentity, A>, ParseTag,ParseTag, CTag>
  {
    typedef ForEachTypeParser<A, ParseTag,ParseTag, CTag> ForEachA_t;
    typedef OpVisitor<OpIdentity, ParseTag>          Visitor_t;
    typedef typename ForEachA_t::Type_t   TypeA_t;
    typedef Combine1<TypeA_t, OpIdentity, CTag>   Combiner_t;
    typedef typename Combiner_t::Type_t   Type_t;
    static Type_t apply(const UnaryNode<OpIdentity, A> &expr, const ParseTag &f, 
			const ParseTag &v, const CTag &c)
    {
      TypeA_t A_val  = ForEachA_t::apply(expr.child(), f, f, c);
      Type_t val = Combiner_t::combine(A_val, expr.operation(), c);

      return val;
    }
  };


  template<class A, class CTag>
  struct ForEachTypeParser<UnaryNode<FnMap, A>, ParseTag,ParseTag, CTag>
  {

    typedef ForEachTypeParser<A, ParseTag,ParseTag, CTag> ForEachA_t;
    typedef OpVisitor<FnMap, ParseTag>          Visitor_t;
    typedef typename ForEachA_t::Type_t   TypeA_t;
    typedef Combine1<TypeA_t, FnMap, CTag>   Combiner_t;
    typedef typename Combiner_t::Type_t   Type_t;

    typedef typename ForEach<A, EvalLeaf1, OpCombine>::Type_t AInnerTypeA_t;
    typedef typename Combine1<AInnerTypeA_t, FnMap, OpCombine>::Type_t InnerTypeBB_t;

    static Type_t apply(const UnaryNode<FnMap, A> &expr, const ParseTag &f, 
			const ParseTag &v, const CTag &c)
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
	QDP_info_primary("offnode: yes");
	const FnMapRsrc& rRSrc = fnmap.getCached();
	int rcvId = rRSrc.getRcvId();
	void * rcvBufDev = QDPCache::Instance().getDevicePtr( rcvId );
	posRcvBuf = f.getJitArgs().addPtr( rcvBufDev );
      } else {
	QDP_info_primary("offnode: no");
	posRcvBuf = f.getJitArgs().addPtr( NULL );
      }

      string codeTypeA;
      typedef InnerTypeBB_t TTT;
      TTT ttt;
      getTypeStringT<TTT>( codeTypeA , f.getJitArgs() );

      ostringstream newIdx;
      newIdx << "((int*)(" << f.getJitArgs().getPtrName() << "[ " << posGoff  << " ].ptr))" << "[" << f.getIndex() << "]";

      ParseTag ff( f.getJitArgs() , newIdx.str() );
      TypeA_t A_val  = ForEachA_t::apply(expr.child(), ff, ff, c);
      Type_t val = Combiner_t::combine(A_val, expr.operation(), c);

      ostringstream code;
      code << newIdx.str() << " < 0 ? " <<
	"(" <<
	"((" << codeTypeA << "*)(" << f.getJitArgs().getPtrName() << 
	"[ " << posRcvBuf  << " ].ptr))" << "[-" << newIdx.str() << "-1]" << 
	"):(" <<
	ff.ossCode.str() <<
	")";

      f.ossCode << code.str();

      //ParseTag ff( f.getJitArgs() , newIdx.str() );
      //TypeA_t A_val  = ForEachA_t::apply(expr.child(), ff, ff, c);
      //Type_t val = Combiner_t::combine(A_val, expr.operation(), c);
      //f.ossCode << ff.ossCode.str();

      return val;
    }
  };




  template<class Op, class A, class B, class CTag>
  struct ForEachTypeParser<BinaryNode<Op, A, B>, ParseTag, ParseTag, CTag>
  {
    typedef ForEachTypeParser<A, ParseTag, ParseTag, CTag> ForEachA_t;
    typedef ForEachTypeParser<B, ParseTag, ParseTag, CTag> ForEachB_t;
    typedef OpVisitor<Op, ParseTag>                Visitor_t;

    typedef typename ForEachA_t::Type_t  TypeA_t;
    typedef typename ForEachB_t::Type_t  TypeB_t;

    typedef Combine2<TypeA_t, TypeB_t, Op, CTag>  Combiner_t;

    typedef typename Combiner_t::Type_t Type_t;

    static Type_t apply(const BinaryNode<Op, A, B> &expr, const ParseTag &f, 
			const ParseTag &v, const CTag &c) 
    {
      if (!Visitor_t::visit(expr.operation(),f))
	return false;

      f.ossCode << "(";
      TypeA_t left_val  = ForEachA_t::apply(expr.left(), f, f, c);

      f.ossCode << ",";
      TypeB_t right_val = ForEachB_t::apply(expr.right(), f, f, c);

      f.ossCode << ")";

      Type_t val = Combiner_t::combine(left_val, right_val, expr.operation(), c);

      return val;
    }
  };


  template<class Op, class A, class B, class C, class CTag>
  struct ForEachTypeParser<TrinaryNode<Op, A, B, C>, ParseTag, ParseTag, CTag>
  {
    typedef ForEachTypeParser<A, ParseTag, ParseTag, CTag> ForEachA_t;
    typedef ForEachTypeParser<B, ParseTag, ParseTag, CTag> ForEachB_t;
    typedef ForEachTypeParser<C, ParseTag, ParseTag, CTag> ForEachC_t;
    typedef OpVisitor<Op, ParseTag>                Visitor_t;

    typedef typename ForEachA_t::Type_t  TypeA_t;
    typedef typename ForEachB_t::Type_t  TypeB_t;
    typedef typename ForEachC_t::Type_t  TypeC_t;

    typedef Combine3<TypeA_t, TypeB_t, TypeC_t, Op, CTag> Combiner_t;

    typedef typename Combiner_t::Type_t Type_t;

    static Type_t apply(const TrinaryNode<Op, A, B, C> &expr, const ParseTag &f, 
			const ParseTag &v, const CTag &c) 
    {
      if (!Visitor_t::visit(expr.operation(),f))
	return false;

      f.ossCode << "(";

      TypeA_t left_val  = ForEachA_t::apply(expr.left(), f, f, c);
      f.ossCode << ",";

      TypeB_t middle_val = ForEachB_t::apply(expr.middle(), f, f, c);
      f.ossCode << ",";

      TypeB_t right_val = ForEachC_t::apply(expr.right(), f, f, c);
      f.ossCode << ")";

      Type_t val = Combiner_t::combine(left_val, middle_val, right_val, expr.operation(), c);

      return val;
    }
  };





  template<>
  struct LeafFunctor<float, ParseTag>
  {
    typedef bool Type_t;
    static bool apply(const float &s, const ParseTag &f)
    { 
      f.ossCode << "float"; 
      return true;
    }
  };

  template<>
  struct LeafFunctor<double, ParseTag>
  {
    typedef bool Type_t;
    static bool apply(const double &s, const ParseTag &f)
    { 
      f.ossCode << "double"; 
      return true;
    }
  };

  template<>
  struct LeafFunctor<int, ParseTag>
  {
    typedef bool Type_t;
    static bool apply(const int &s, const ParseTag &f)
    { 
      f.ossCode << "int"; 
      return true;
    }
  };

  template<>
  struct LeafFunctor<char, ParseTag>
  {
    typedef bool Type_t;
    static bool apply(const char &s, const ParseTag &f)
    { 
      f.ossCode << "char"; 
      return true;
    }
  };

  template<>
  struct LeafFunctor<bool, ParseTag>
  {
    typedef bool Type_t;
    static bool apply(const bool &s, const ParseTag &f)
    { 
      f.ossCode << "bool"; 
      return true;
    }
  };

  template<class T>
  struct LeafFunctor<IScalar<T>, ParseTag>
  {
    typedef bool Type_t;
    static bool apply(const IScalar<T> &s, const ParseTag &f)
    { 
      f.ossCode << "IScalar<";
      LeafFunctor<T,ParseTag>::apply(s.elem(),f);
      f.ossCode << "> "; 
      return true;
    }
  };

  template<class T, int N>
  struct LeafFunctor<ILattice<T,N>, ParseTag>
  {
    typedef bool Type_t;
    static bool apply(const ILattice<T,N> &s, const ParseTag &f)
    { 
      f.ossCode << "ILattice<";
      LeafFunctor<T,ParseTag>::apply(s.elem(0),f);
      f.ossCode << "," << N << "> "; 
      return true;
    }
  };

  template<class T>
  struct LeafFunctor<RScalar<T>, ParseTag>
  {
    typedef bool Type_t;
    static bool apply(const RScalar<T> &s, const ParseTag &f)
    { 
      f.ossCode << "RScalar<"; 
      LeafFunctor<T,ParseTag>::apply(s.elem(),f);
      f.ossCode << "> "; 
      return true;
    }
  };

  template<class T>
  struct LeafFunctor<RComplex<T>, ParseTag>
  {
    typedef bool Type_t;
    static bool apply(const RComplex<T> &s, const ParseTag &f)
    { 
      f.ossCode << "RComplex<"; 
      LeafFunctor<T,ParseTag>::apply(s.real(),f);
      f.ossCode << "> "; 
      return true;
    }
  };

  template<class T>
  struct LeafFunctor<PScalar<T>, ParseTag>
  {
    typedef bool Type_t;
    static bool apply(const PScalar<T> &s, const ParseTag &f)
    { 
      f.ossCode << "PScalar<"; 
      LeafFunctor<T,ParseTag>::apply(s.elem(),f);
      f.ossCode << "> "; 
      return true;
    }
  };

  template<class T>
  struct LeafFunctor<PSeed<T>, ParseTag>
  {
    typedef bool Type_t;
    static bool apply(const PSeed<T> &s, const ParseTag &f)
    { 
      f.ossCode << "PSeed<"; 
      LeafFunctor<T,ParseTag>::apply(s.elem(0),f);
      f.ossCode << "> "; 
      return true;
    }
  };

  template <class T, int N>
  struct LeafFunctor<PColorMatrix<T,N>, ParseTag>
  {
    typedef bool Type_t;
    static bool apply(const PColorMatrix<T,N> &s, const ParseTag &f)
    { 
      f.ossCode << "PColorMatrix<"; 
      LeafFunctor<T,ParseTag>::apply(s.elem(0,0),f);
      f.ossCode << "," << N << "> "; 
      return true;
    }
  };

  template <class T, int N>
  struct LeafFunctor<PSpinMatrix<T,N>, ParseTag>
  {
    typedef bool Type_t;
    static bool apply(const PSpinMatrix<T,N> &s, const ParseTag &f)
    { 
      f.ossCode << "PSpinMatrix<"; 
      LeafFunctor<T,ParseTag>::apply(s.elem(0,0),f);
      f.ossCode << "," << N << "> "; 
      return true;
    }
  };

  template <class T, int N>
  struct LeafFunctor<PColorVector<T,N>, ParseTag>
  {
    typedef bool Type_t;
    static bool apply(const PColorVector<T,N> &s, const ParseTag &f)
    { 
      f.ossCode << "PColorVector<"; 
      LeafFunctor<T,ParseTag>::apply(s.elem(0),f);
      f.ossCode << "," << N << "> "; 
      return true;
    }
  };

  template <class T, int N>
  struct LeafFunctor<PSpinVector<T,N>, ParseTag>
  {
    typedef bool Type_t;
    static bool apply(const PSpinVector<T,N> &s, const ParseTag &f)
    { 
      f.ossCode << "PSpinVector<"; 
      LeafFunctor<T,ParseTag>::apply(s.elem(0),f);
      f.ossCode << "," << N << "> "; 
      return true;
    }
  };

  template <class T, int N, template<class,int> class C>
  struct LeafFunctor<PMatrix<T,N,C>, ParseTag>
  {
    typedef bool Type_t;
    static bool apply(const PMatrix<T,N,C> &s, const ParseTag &f)
    { 
      return LeafFunctor<C<T,N>,ParseTag>::apply(static_cast<const C<T,N>&>(s),f);
    }
  };

  template <class T, int N, template<class,int> class C>
  struct LeafFunctor<PVector<T,N,C>, ParseTag>
  {
    typedef bool Type_t;
    static bool apply(const PVector<T,N,C> &s, const ParseTag &f)
    { 
      return LeafFunctor<C<T,N>,ParseTag>::apply(static_cast<const C<T,N>&>(s),f);
    }
  };

  template<class T>
  struct LeafFunctor<OScalar<T>, ParseTag>
  {
    typedef bool Type_t;
    static bool apply(const OScalar<T> &s, const ParseTag &f)
    { 
      string strIdentifier;
      
      ParseTag ptSubType(f.getJitArgs(),"null from OScalar");
      LeafFunctor<T,ParseTag>::apply(T(),ptSubType);

      if (f.insertObject( strIdentifier , ptSubType.ossCode.str() , s )) {

	f.ossCode << strIdentifier;
	
	return true;
      } else {
	return false;
      }
    }
  };

  template<class T>
  struct LeafFunctor<OLattice<T>, ParseTag>
  {
    typedef bool Type_t;
    static bool apply(const OLattice<T> &s, const ParseTag &f)
    { 
      string strIdentifier;
      
      ParseTag ptSubType(f.getJitArgs(),"null from OLattice");
      LeafFunctor<T,ParseTag>::apply(T(),ptSubType);

      if (f.insertObject( strIdentifier , ptSubType.ossCode.str() , s )) {

	f.ossCode << strIdentifier;
	
	return true;
      } else {
	return false;
      }
    }
  };

  template<int N>
  struct LeafFunctor<GammaType<N>, ParseTag>
  {
    typedef bool Type_t;
    static bool apply(const GammaType<N> &s, const ParseTag &f)
    { 
      string strIdentifier;
      
      ostringstream ossTmp;
      ossTmp << "GammaType<" << N << ">";
      string strType = ossTmp.str();
      
      if (f.insertObject( strIdentifier , strType , s )) {

	f.ossCode << strIdentifier;
	
	return true;
      } else {
	return false;
      }

    }
  };

  template<int N, int m>
  struct LeafFunctor<GammaConst<N,m>, ParseTag>
  {
    typedef bool Type_t;
    static bool apply(const GammaConst<N,m> &s, const ParseTag &f)
    { 
      f.ossCode << "GammaConst<" << N << "," << m << ">() ";
      //f.ossCode << "GammaConst<" << N << "," << m << ">()";
      return true;
    }
  };



  template<class Op>
  struct BracketPrinter
  {
    static void start(Op,const ParseTag &p)
    { p.ossCode << "("; }

    static void center(Op,const ParseTag &p)
    { p.ossCode << ","; }

    static void finish(Op,const ParseTag &p)
    { p.ossCode << ") "; }
  };



#include"qdp_parsetree_peteops.h"
#include"qdp_parsetree_qdpops.h"



  //-----------------------------------------------------------------------------
  // Additional operator tags 
  //-----------------------------------------------------------------------------



  template <>
  struct OpVisitor<FnPeekColorMatrix, ParseTag> 
  { 
    static bool visit(const FnPeekColorMatrix & op, const ParseTag & f) 
    { 
      string strIdentifier;
      if (f.insertObject( strIdentifier , "FnPeekColorMatrix" , op )) {
	f.ossCode << strIdentifier;
	return true;
      } else {
	return false;
      }
    }
  };

  template <>
  struct OpVisitor<FnPeekColorVector, ParseTag> 
  { 
    static bool visit(const FnPeekColorVector & op, const ParseTag & f) 
    { 
      string strIdentifier;
      if (f.insertObject( strIdentifier , "FnPeekColorVector" , op )) {
	f.ossCode << strIdentifier;
	return true;
      } else {
	return false;
      }
    }
  };

  template <>
  struct OpVisitor<FnPeekSpinMatrix, ParseTag> 
  { 
    static bool visit(const FnPeekSpinMatrix & op, const ParseTag & f) 
    { 
      string strIdentifier;
      if (f.insertObject( strIdentifier , "FnPeekSpinMatrix" , op )) {
	f.ossCode << strIdentifier;
	return true;
      } else {
	return false;
      }
    }
  };

  template <>
  struct OpVisitor<FnPeekSpinVector, ParseTag> 
  { 
    static bool visit(const FnPeekSpinVector & op, const ParseTag & f) 
    { 
      string strIdentifier;
      if (f.insertObject( strIdentifier , "FnPeekSpinVector" , op )) {
	f.ossCode << strIdentifier;
	return true;
      } else {
	return false;
      }
    }
  };


  template <>
  struct OpVisitor<FnPokeColorMatrix, ParseTag> 
  { 
    static bool visit(const FnPokeColorMatrix & op, const ParseTag & f) 
    { 
      string strIdentifier;
      if (f.insertObject( strIdentifier , "FnPokeColorMatrix" , op )) {
	f.ossCode << strIdentifier;
	return true;
      } else {
	return false;
      }
    }
  };

  template <>
  struct OpVisitor<FnPokeColorVector, ParseTag> 
  { 
    static bool visit(const FnPokeColorVector & op, const ParseTag & f) 
    { 
      string strIdentifier;
      if (f.insertObject( strIdentifier , "FnPokeColorVector" , op )) {
	f.ossCode << strIdentifier;
	return true;
      } else {
	return false;
      }
    }
  };

  template <>
  struct OpVisitor<FnPokeSpinMatrix, ParseTag> 
  { 
    static bool visit(const FnPokeSpinMatrix & op, const ParseTag & f) 
    { 
      string strIdentifier;
      if (f.insertObject( strIdentifier , "FnPokeSpinMatrix" , op )) {
	f.ossCode << strIdentifier;
	return true;
      } else {
	return false;
      }
    }
  };

  template <>
  struct OpVisitor<FnPokeSpinVector, ParseTag> 
  { 
    static bool visit(const FnPokeSpinVector & op, const ParseTag & f) 
    { 
      string strIdentifier;
      if (f.insertObject( strIdentifier , "FnPokeSpinVector" , op )) {
	f.ossCode << strIdentifier;
	return true;
      } else {
	return false;
      }
    }
  };



  template <>
  struct OpVisitor<FnSum, ParseTag> : public BracketPrinter<FnSum>
  { 
    static bool visit(FnSum op, const ParseTag & t) 
    { t.ossCode << "FnSum()";  return true; }
  };

  template <>
  struct OpVisitor<FnNorm2, ParseTag> : public BracketPrinter<FnNorm2>
  { 
    static bool visit(FnNorm2 op, const ParseTag & t) 
    { t.ossCode << "FnNorm2()"; return true; }
  };

  template <>
  struct OpVisitor<OpGammaConstMultiply, ParseTag> : public BracketPrinter<OpGammaConstMultiply>
  { 
    static bool visit(OpGammaConstMultiply op, const ParseTag & t) 
    { t.ossCode << "OpGammaConstMultiply()"; return true; }
  };

  template <>
  struct OpVisitor<OpMultiplyGammaConst, ParseTag> : public BracketPrinter<OpMultiplyGammaConst>
  { 
    static bool visit(OpMultiplyGammaConst op, const ParseTag & t) 
    { t.ossCode << "OpMultiplyGammaConst()"; return true; }
  };

  template <>
  struct OpVisitor<OpGammaTypeMultiply, ParseTag> : public BracketPrinter<OpGammaTypeMultiply>
  { 
    static bool visit(OpGammaTypeMultiply op, const ParseTag & t) 
    { t.ossCode << "OpGammaTypeMultiply()"; return true; }
  };

  template <>
  struct OpVisitor<OpMultiplyGammaType, ParseTag> : public BracketPrinter<OpMultiplyGammaType>
  { 
    static bool visit(OpMultiplyGammaType op, const ParseTag & t) 
    { t.ossCode << "OpMultiplyGammaType()"; return true; }
  };


  //------------------------------------------------------------------------
  // Special optimizations
  //------------------------------------------------------------------------

  template <>
  struct OpVisitor<OpAdjMultiply, ParseTag> : public BracketPrinter<OpAdjMultiply>
  { 
    static bool visit(OpAdjMultiply op, const ParseTag & t) 
    { t.ossCode << "OpAdjMultiply()"; return true; }
  };

  template <>
  struct OpVisitor<OpMultiplyAdj, ParseTag> : public BracketPrinter<OpMultiplyAdj>
  { 
    static bool visit(OpMultiplyAdj op, const ParseTag & t) 
    { t.ossCode << "OpMultiplyAdj()"; return true; }
  };

  template <>
  struct OpVisitor<OpAdjMultiplyAdj, ParseTag> : public BracketPrinter<OpAdjMultiplyAdj>
  { 
    static bool visit(OpAdjMultiplyAdj op, const ParseTag & t) 
    { t.ossCode << "OpAdjMultiplyAdj()"; return true; }
  };

  template <>
  struct OpVisitor<FnTraceMultiply, ParseTag> : public BracketPrinter<FnTraceMultiply>
  { 
    static bool visit(FnTraceMultiply op, const ParseTag & t) 
    { t.ossCode << "FnTraceMultiply()"; return true; }
  };

  template <>
  struct OpVisitor<FnTraceColorMultiply, ParseTag> : public BracketPrinter<FnTraceColorMultiply>
  { 
    static bool visit(FnTraceColorMultiply op, const ParseTag & t) 
    { t.ossCode << "FnTraceColorMultiply()"; return true; }
  };

  template <>
  struct OpVisitor<FnTraceSpinMultiply, ParseTag> : public BracketPrinter<FnTraceSpinMultiply>
  { 
    static bool visit(FnTraceSpinMultiply op, const ParseTag & t) 
    { t.ossCode << "FnTraceSpinMultiply()"; return true; }
  };



  template <>
  struct OpVisitor<FnTraceSpinOuterProduct, ParseTag> : public BracketPrinter<FnTraceSpinOuterProduct>
  { 
    static bool visit(FnTraceSpinOuterProduct op, const ParseTag & t) 
    { t.ossCode << "FnTraceSpinOuterProduct()"; return true; }
  };


  template <>
  struct OpVisitor<FnTraceSpinQuarkContract13, ParseTag> : public BracketPrinter<FnTraceSpinQuarkContract13>
  { 
    static bool visit(FnTraceSpinQuarkContract13 op, const ParseTag & t) 
    { t.ossCode << "FnTraceSpinQuarkContract13()"; return true; }
  };


} 

#endif  
