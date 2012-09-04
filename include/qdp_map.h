// -*- C++ -*-

/*! @file
 * @brief Map classes
 *
 * Support classes for maps/shifts
 */

#ifndef QDP_MAP_H
#define QDP_MAP_H

namespace QDP {

// Helpful for communications
#define FORWARD 1
#define BACKWARD -1


/*! @defgroup map Maps and shifts
 *
 * Maps are the mechanism for communications. Under a map,
 * a data-parallel object is mapped uniquely from sites to
 * sites. Nearest neighbor shifts are an example of the more
 * generic map.
 *
 * @{
 */

//! MapFunc 
/*! Abstract base class used as a function object for constructing maps */
class MapFunc
{
public:
  //! Virtual destructor - no cleanup needed
  virtual ~MapFunc() {}
  //! Maps a lattice coordinate under a map to a new lattice coordinate
  /*! sign > 0 for map, sign < 0 for the inverse map */
  virtual multi1d<int> operator() (const multi1d<int>& coordinate, int sign) const = 0;
};
    

//! ArrayMapFunc 
/*! Abstract base class used as a function object for constructing maps */
class ArrayMapFunc
{
public:
  //! Virtual destructor - no cleanup needed
  virtual ~ArrayMapFunc() {}

  //! Maps a lattice coordinate under a map to a new lattice coordinate
  /*! sign > 0 for map, sign < 0 for the inverse map */
  virtual multi1d<int> operator() (const multi1d<int>& coordinate, int sign, int dir) const = 0;

  //! Returns the array size - the number of directions which are to be used
  virtual int numArray() const = 0;
};
    
/** @} */ // end of group map


//! General permutation map class for communications
class Map
{
public:
  //! Constructor - does nothing really
  Map() {}

  //! Destructor
  ~Map() {}

  //! Constructor from a function object
  Map(const MapFunc& fn) {make(fn);}

  //! Actual constructor from a function object
  /*! The semantics are   source_site = func(dest_site,isign) */
  void make(const MapFunc& func);


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
  int getRoffsetsId() const { return roffsetsId;}
  int getSoffsetsId() const { return soffsetsId;}
  int getGoffsetsId() const { return goffsetsId;}

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

  int roffsetsId;
  int soffsetsId;
  int goffsetsId;
  int myId; // master map id

  multi1d<int> srcenodes;
  multi1d<int> destnodes;

  multi1d<int> srcenodes_num;
  multi1d<int> destnodes_num;

  // Indicate off-node communications is needed;
  bool offnodeP;
};





struct FnMap
{
  //PETE_EMPTY_CONSTRUCTORS(FnMap)
private:
  FnMap& operator=(const FnMap& f);

public:
  const Map& map;
  //std::shared_ptr<RsrcWrapper> pRsrc;
  Handle<RsrcWrapper> pRsrc;

  FnMap(const Map& m): map(m), pRsrc(new RsrcWrapper( m.destnodes , m.srcenodes )) {}
  FnMap(const FnMap& f) : map(f.map) , pRsrc(f.pRsrc) {}

  const FnMapRsrc& getResource(int srcnum_, int dstnum_) {
    return (*pRsrc).getResource( srcnum_ , dstnum_ );
  }

  const FnMapRsrc& getCached() const {
    return (*pRsrc).get();
  }
  
  template<class T>
  inline typename UnaryReturn<T, FnMap>::Type_t
  operator()(const T &a) const
  {
    return (a);
  }

};








} // namespace QDP

#endif
