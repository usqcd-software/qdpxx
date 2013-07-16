// -*- C++ -*-
/**
 * Header file containing definitions for pshift
 */
#pragma once

#include <memory>

namespace QDP {

  struct ShiftPhase1
  {
    // empty
  };

  struct ShiftPhase2
  {
    // empty
  };

  /**
   * default behaviour: no shift, return 0
   * this is used by BitOrCombine combiner
   */
  template<class T>
  struct LeafFunctor<OScalar<T>, ShiftPhase1>
  {
    typedef int Type_t;
    inline static Type_t apply(const OScalar<T> &a, const ShiftPhase1 &f) {
      return 0;
    }
  };

  /**
   * default behaviour: no shift, returns 0
   * again this is used by BitOrCombine combiner
   */
  template<class T>
  struct LeafFunctor<OLattice<T>, ShiftPhase1>
  {
    typedef int Type_t;
    inline static Type_t apply(const OLattice<T> &a, const ShiftPhase1 &f) {
      return 0;
    }
  };

  /**
   * default behaviour: no shift, returns 0
   * This is used by BitOrCombine combiner
   */
  template<class T, class C>
  struct LeafFunctor<QDPType<T,C>, ShiftPhase1>
  {
    typedef int Type_t;
    static int apply(const QDPType<T,C> &s, const ShiftPhase1 &f) {
      return 0;
    }
  };


  /**
   * default behaviour: no shift, returns 0
   * therefore this is used by BitOrCombiner
   */
  template<class T>
  struct LeafFunctor<OScalar<T>, ShiftPhase2>
  {
    typedef int Type_t;
    inline static Type_t apply(const OScalar<T> &a, const ShiftPhase2 &f) {
      return 0;
    }
  };


  /**
   * default behaviour: no shift, returns 0
   * this is used when BitOrCombine is called to find out
   * whether there is shift or not
   */
  template<class T>
  struct LeafFunctor<OLattice<T>, ShiftPhase2>
  {
    typedef int Type_t;
    inline static Type_t apply(const OLattice<T> &a, const ShiftPhase2 &f) {
      return 0;
    }
  };

  /**
   * default behaviour for phase2: no shift
   */
  template<class T, class C>
  struct LeafFunctor<QDPType<T,C>, ShiftPhase2>
  {
    typedef int Type_t;
    static int apply(const QDPType<T,C> &s, const ShiftPhase2 &f) {
      return 0;
    }
  };

  /**
   * Leaf functor of shift phases for GammaType
   */
  template<int N>
  struct LeafFunctor<GammaType<N>, ShiftPhase1>
  {
    typedef int Type_t;
    static int apply(const GammaType<N> &s, const ShiftPhase1 &f) { 
      return 0; 
    }
};

  template<int N, int m>
  struct LeafFunctor<GammaConst<N,m>, ShiftPhase1>
  {
    typedef int Type_t;
    static int apply(const GammaConst<N,m> &s, const ShiftPhase1 &f) {
      return 0; 
    }
};

  template<int N>
  struct LeafFunctor<GammaType<N>, ShiftPhase2>
  {
    typedef int Type_t;
    static int apply(const GammaType<N> &s, const ShiftPhase2 &f) { 
      return 0; 
    }
};

  template<int N, int m>
  struct LeafFunctor<GammaConst<N,m>, ShiftPhase2>
  {
    typedef int Type_t;
    static int apply(const GammaConst<N,m> &s, const ShiftPhase2 &f) {
      return 0; 
    }
  };
  
  /**
   * Define default behavior for nullcombine operator
   * Basically visiting every leaf node
   * This is a specialization of ForEach inside PETE
   */
  template<class Op, class A, class B, class FTag>
  struct ForEach<BinaryNode<Op, A, B>, FTag, NullCombine >
  {
    typedef int Type_t;
    inline static
    Type_t apply(const BinaryNode<Op, A, B> &expr, 
		 const FTag &f,
		 const NullCombine &c)
    {
      ForEach<A, FTag, NullCombine>::apply(expr.left(), f, c);
      ForEach<B, FTag, NullCombine>::apply(expr.right(), f, c);
      return 0;
    }
  };

  /**
   * Forward decleration of FnMap
   */
  struct FnMap;

  /**
   * Implementation of map. This should be the same for parcalar
   * and parscalarvec
   */
  //! General permutation map class for communications
  class Map
  {
  public:
    //! Constructor - does nothing really
#if defined (ARCH_PARSCALARVEC)
    Map (void) 
      :isign (1), dir(0)
    {
      // empty
    }
#else
    Map() {}
#endif

    //! Destructor
    ~Map() {}

    //! Constructor from a function object
    Map(const MapFunc& fn) {make(fn);}

    //! Actual constructor from a function object
    /*! The semantics are   source_site = func(dest_site,isign) */
    void make(const MapFunc& func);

#if defined (ARCH_PARSCALARVEC)
    // set sign and direction of this map
    // sign can be -1 or +1. d is direction : 0, 1, 2, 3
    void mapInfo (int sign, int d) 
    {
      isign = sign;
      dir = d;
    }

    int mapDir (void) const
    {
      return dir;
    }

    int mapSign (void) const
    {
      return isign;
    }
#endif

    template<class T1,class C1>
    inline typename MakeReturn<UnaryNode<FnMap,
	 typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>, C1>::Expression_t
    operator()(const QDPType<T1,C1> & l)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(*this),
	CreateLeaf<QDPType<T1,C1> >::make(l)));
    }


    template<class T1,class C1>
    inline typename MakeReturn<UnaryNode<FnMap,
	 typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
    operator()(const QDPExpr<T1,C1> & l)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(*this),
	CreateLeaf<QDPExpr<T1,C1> >::make(l)));
    }


  public:
    //! Accessor to offsets
    const multi1d<int>& goffset() const {return goffsets;}
    const multi1d<int>& soffset() const {return soffsets;}
    const multi1d<int>& roffset() const {return roffsets;}
    
    int getId() const {return myId;}
    bool hasOffnode() const { return offnodeP; }

  private:
    //! Hide copy constructor
    Map(const Map&) {}

    //! Hide operator=
    void operator=(const Map&) {}

  private:
    friend class FnMap;
    friend class FnMapRsrc;
    template<class E,class F,class C> friend class ForEach;

    //! Offset table used for communications. 
    /*! 
     * The direction is in the sense of the Map or Shift functions from QDP.
     * goffsets(position) 
     */ 
    multi1d<int> goffsets;
    multi1d<int> soffsets;
    multi1d<int> srcnode;
    multi1d<int> dstnode;

    multi1d<int> roffsets;

    int myId; // master map id

    multi1d<int> srcenodes;
    multi1d<int> destnodes;

    multi1d<int> srcenodes_num;
    multi1d<int> destnodes_num;
    
    // Indicate off-node communications is needed;
    bool offnodeP;

#if defined (ARCH_PARSCALARVEC)
    // Remember the direction and sign for this map
    // X direction is special 
    // isign is -1 or +1
    int isign, dir;
#endif
  };

  /**
   * Operator FnMap. This should be the same for parscalar and parscalarvec
   */
  struct FnMap
  {
  private:
    // hide assignment operator
    FnMap& operator=(const FnMap& f);

  public:
    const Map& map;
    // QDPHandle::Handle<RsrcWrapper> pRsrc;
    std::shared_ptr<RsrcWrapper> pRsrc;

    FnMap(const Map& m): map(m), pRsrc(new RsrcWrapper( m.destnodes , m.srcenodes )) 
    {
      // empty
    }

    FnMap(const FnMap& f) : map(f.map) , pRsrc(f.pRsrc) {}

    const FnMapRsrc& getResource(int srcnum_, int dstnum_) {
      // assert(pRsrc);
      return pRsrc->getResource( srcnum_ , dstnum_ );
    }
    
    const FnMapRsrc& getCached() const {
      // assert(pRsrc);
      return pRsrc->get();
    }
  
    template<class T>
    inline typename UnaryReturn<T, FnMap>::Type_t
    operator()(const T &a) const
    {
      return (a);
    }
    
  };  


  //-----------------------------------------------------------------------------
  //! Array of general permutation map class for communications
  class ArrayMap
  {
  public:
    //! Constructor - does nothing really
    ArrayMap() {}

    //! Destructor
    ~ArrayMap() {}

    //! Constructor from a function object
    ArrayMap(const ArrayMapFunc& fn) {make(fn);}

    //! Actual constructor from a function object
    /*! The semantics are   source_site = func(dest_site,isign,dir) */
    void make(const ArrayMapFunc& func);

    //! Function call operator for a shift
    /*! 
     * map(source,dir)
     *
     * Implements:  dest(x) = source(map(x,dir))
     *
     * Shifts on a OLattice are non-trivial.
     * Notice, there may be an ILattice underneath which requires shift args.
     * This routine is very architecture dependent.
     */
    template<class T1,class C1>
    inline typename MakeReturn<UnaryNode<FnMap,
					 typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>, C1>::Expression_t
    operator()(const QDPType<T1,C1> & l, int dir)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(&(mapsa[dir])),
						CreateLeaf<QDPType<T1,C1> >::make(l)));
    }


    template<class T1,class C1>
    inline typename MakeReturn<UnaryNode<FnMap,
					 typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
    operator()(const QDPExpr<T1,C1> & l, int dir)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(&(mapsa[dir])),
						CreateLeaf<QDPExpr<T1,C1> >::make(l)));
    }



  private:
    //! Hide copy constructor
    ArrayMap(const ArrayMap&) {}

    //! Hide operator=
    void operator=(const ArrayMap&) {}

  private:
    multi1d<Map> mapsa;
  
  };

  //-----------------------------------------------------------------------------
  //! BiDirectional of general permutation map class for communications
  class BiDirectionalMap
  {
  public:
    //! Constructor - does nothing really
    BiDirectionalMap() {}

    //! Destructor
    ~BiDirectionalMap() {}

    //! Constructor from a function object
    BiDirectionalMap(const MapFunc& fn) {make(fn);}

    //! Actual constructor from a function object
    /*! The semantics are   source_site = func(dest_site,isign) */
    void make(const MapFunc& func);

    //! Function call operator for a shift
    /*! 
     * map(source,isign)
     *
     * Implements:  dest(x) = source(map(x,isign))
     *
     * Shifts on a OLattice are non-trivial.
     * Notice, there may be an ILattice underneath which requires shift args.
     * This routine is very architecture dependent.
     */
    template<class T1,class C1>
    inline typename MakeReturn<UnaryNode<FnMap,
					 typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>, C1>::Expression_t
    operator()(const QDPType<T1,C1> & l, int isign)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(&(bimaps[(isign+1)>>1])),
						CreateLeaf<QDPType<T1,C1> >::make(l)));
    }


    template<class T1,class C1>
    inline typename MakeReturn<UnaryNode<FnMap,
					 typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
    operator()(const QDPExpr<T1,C1> & l, int isign)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(&(bimaps[(isign+1)>>1])),
						CreateLeaf<QDPExpr<T1,C1> >::make(l)));
    }

  private:
    //! Hide copy constructor
    BiDirectionalMap(const BiDirectionalMap&) {}

    //! Hide operator=
    void operator=(const BiDirectionalMap&) {}

  private:
    multi1d<Map> bimaps;
  
  };


  //-----------------------------------------------------------------------------
  //! ArrayBiDirectional of general permutation map class for communications
  class ArrayBiDirectionalMap
  {
  public:
    //! Constructor - does nothing really
    ArrayBiDirectionalMap() {}

    //! Destructor
    ~ArrayBiDirectionalMap() {}

    //! Constructor from a function object
    ArrayBiDirectionalMap(const ArrayMapFunc& fn) {make(fn);}

    //! Actual constructor from a function object
    /*! The semantics are   source_site = func(dest_site,isign,dir) */
    void make(const ArrayMapFunc& func);

    //! Function call operator for a shift
    /*! 
     * Implements:  dest(x) = source(map(x,isign,dir))
     *
     * Syntax:
     * map(source,isign,dir)
     *
     * isign = parity of direction (+1 or -1)
     * dir   = array index (could be direction in range [0,...,Nd-1])
     *
     * Implements:  dest(x) = s1(x+isign*dir)
     * There are cpp macros called  FORWARD and BACKWARD that are +1,-1 resp.
     * that are often used as arguments
     *
     * Shifts on a OLattice are non-trivial.
     * Notice, there may be an ILattice underneath which requires shift args.
     * This routine is very architecture dependent.
     */
    template<class T1,class C1>
    inline typename MakeReturn<UnaryNode<FnMap,
	 typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>, C1>::Expression_t
    operator()(const QDPType<T1,C1> & l, int isign, int dir)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(bimapsa((isign+1)>>1,dir)),
						CreateLeaf<QDPType<T1,C1> >::make(l)));
    }

    template<class T1,class C1>
    inline typename MakeReturn<UnaryNode<FnMap,
	 typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
    operator()(const QDPExpr<T1,C1> & l, int isign, int dir)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(bimapsa((isign+1)>>1,dir)),
						CreateLeaf<QDPExpr<T1,C1> >::make(l)));
    }


  private:
    //! Hide copy constructor
    ArrayBiDirectionalMap(const ArrayBiDirectionalMap&) {}

    //! Hide operator=
    void operator=(const ArrayBiDirectionalMap&) {}

  private:
    multi2d<Map> bimapsa;
  
  };

}
