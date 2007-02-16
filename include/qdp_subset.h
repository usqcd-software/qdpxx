// -*- C++ -*-
// $Id: qdp_subset.h,v 1.10 2007-02-16 22:22:21 bjoo Exp $

/*! @file
 * @brief Sets and subsets
 */

QDP_BEGIN_NAMESPACE(QDP);

/*! @defgroup subsets Sets and Subsets
 *
 * Sets are the objects that facilitate colorings of the lattice.
 * Subsets are groups of sites that are all of one color.
 * Subsets (and in a few cases Sets) can be used to restrict operations
 * to only a particular coloring of the lattice.
 *
 * @{
 */

//-----------------------------------------------------------------------
//! SetMap
class SetFunc
{
public:
  virtual int operator() (const multi1d<int>& coordinate) const = 0;
  virtual int numSubsets() const = 0;
};

//-----------------------------------------------------------------------
//! Subsets - controls how lattices are looped
class Subset
{
public:
  //! There can be an empty constructor
  Subset() {}

  //! Copy constructor
  Subset(const Subset& s) {}

  //! Destructor for a subset
  virtual ~Subset() {}

  //! Access the coloring for this subset
  virtual int color() const = 0;

public:
  virtual bool hasOrderedRep() const = 0;
  virtual int start() const = 0;
  virtual int end() const = 0;

  virtual const multi1d<int>& siteTable() const = 0;
  virtual int numSiteTable() const = 0;
};

//-----------------------------------------------------------------------
//! Set - collection of subsets controlling which sites are involved in an operation
class Set
{
public:
  //! There can be an empty constructor
  Set() {}

  //! Constructor from a function object
  virtual void make(const SetFunc& fn) = 0;

  //! Index operator selects a subset from a set
  virtual const Subset& operator[](int subset_index) const = 0;

  //! Return number of subsets
  virtual int numSubsets() const = 0;

  //! Destructor for a set
  virtual ~Set() {}

public:
  //! The coloring of the lattice sites
  virtual const multi1d<int>& latticeColoring() const = 0;
};


//-----------------------------------------------------------------------
// Forward declaration
class UnOrderedSet;
class OrderedSet;


//-----------------------------------------------------------------------
//! UnorderedSubsets - controls how lattices are looped
class UnorderedSubset : public Subset
{
public:
  //! There can be an empty constructor
  UnorderedSubset() {}

  //! Copy constructor
  UnorderedSubset(const UnorderedSubset& s):
    ordRep(s.ordRep), startSite(s.startSite), endSite(s.endSite), 
    sub_index(s.sub_index), sitetable(s.sitetable)
    {}

  // Simple constructor
  void make(const UnorderedSubset& s);

  //! Destructor for a subset
  virtual ~UnorderedSubset() {}

  //! The = operator
  UnorderedSubset& operator=(const UnorderedSubset& s);

  //! Access the coloring for this subset
  int color() const {return sub_index;}

protected:
  // Simple constructor
  void make(bool rep, int start, int end, multi1d<int>* ind, int cb);

private:
  bool ordRep;
  int startSite;
  int endSite;
  int sub_index;

  //! Site lookup table
  multi1d<int>* sitetable;

public:
  inline bool hasOrderedRep() const {return ordRep;}
  inline int start() const {return startSite;}
  inline int end() const {return endSite;}

  const multi1d<int>& siteTable() const {return *sitetable;}
  inline int numSiteTable() const {return sitetable->size();}

  friend class UnorderedSet;
};


//-----------------------------------------------------------------------
//! Ordered Subsets - optimized subsets for lattice looping
class OrderedSubset : public Subset
{
public:
  //! There can be an empty constructor
  OrderedSubset() {}

  //! Copy constructor
  OrderedSubset(const OrderedSubset& s): 
    startSite(s.startSite), endSite(s.endSite), 
    sub_index(s.sub_index), sitetable(s.sitetable)
    {}

  // Simple constructor
  void make(const OrderedSubset& s);

  //! Destructor for a subset
  virtual ~OrderedSubset() {}

  //! The = operator
  OrderedSubset& operator=(const OrderedSubset& s);

  //! Access the coloring for this subset
  int color() const {return sub_index;}

protected:
  // Simple constructor
  void make(int start, int end, multi1d<int>* ind, int cb);


private:
  int startSite;
  int endSite;
  int sub_index;

  //! Site lookup table
  multi1d<int>* sitetable;

public:
  inline bool hasOrderedRep() const {return true;}
  inline int start() const {return startSite;}
  inline int end() const {return endSite;}

  const multi1d<int>& siteTable() const {return *sitetable;}
  inline int numSiteTable() const {return sitetable->size();}

  friend class OrderedSet;
};


//-----------------------------------------------------------------------
//! UnorderedSet - collection of subsets controlling which sites are involved in an operation
class UnorderedSet : public Set
{
public:
  //! There can be an empty constructor
  UnorderedSet() {}

  //! Constructor from a function object
  UnorderedSet(const SetFunc& fn) {make(fn);}

  //! Constructor from a function object
  void make(const SetFunc& fn);

  //! Index operator selects a subset from a set
  const UnorderedSubset& operator[](int subset_index) const {return sub[subset_index];}

  //! Return number of subsets
  int numSubsets() const {return sub.size();}

  //! Destructor for a set
  virtual ~UnorderedSet() {}

  //! The = operator
  UnorderedSet& operator=(const UnorderedSet& s);

protected:
  //! A set is composed of an array of subsets
  multi1d<UnorderedSubset> sub;

  //! Index or color array of lattice
  multi1d<int> lat_color;

  //! Array of sitetable arrays
  multi1d<multi1d<int> > sitetables;

public:
  //! The coloring of the lattice sites
  const multi1d<int>& latticeColoring() const {return lat_color;}
};


//-----------------------------------------------------------------------
//! OrderedSet - collection of subsets controlling which sites are involved in an operation
class OrderedSet : public Set
{
public:
  //! There can be an empty constructor
  OrderedSet() {}

  //! Constructor from a function object
  OrderedSet(const SetFunc& fn) {make(fn);}

  //! Constructor from a function object
  void make(const SetFunc& fn);

  //! Index operator selects a subset from a set
  const OrderedSubset& operator[](int subset_index) const {return sub[subset_index];}

  //! Return number of subsets
  int numSubsets() const {return sub.size();}

  //! Destructor for a set
  ~OrderedSet() {}

  //! The = operator
  OrderedSet& operator=(const OrderedSet& s);

protected:
  //! A set is composed of an array of subsets
  multi1d<OrderedSubset> sub;

  //! Index or color array of lattice
  multi1d<int> lat_color;

  //! Array of sitetable arrays
  multi1d<multi1d<int> > sitetables;

public:
  //! The coloring of the lattice sites
  const multi1d<int>& latticeColoring() const {return lat_color;}
};


//-----------------------------------------------------------------------

//! Default all subset
extern OrderedSubset all;


//! Experimental 3d checkerboarding for temp_precond
extern UnorderedSet rb3;

#if QDP_USE_CB2_LAYOUT == 1
//! Default 2-checkerboard (red/black) subset
extern OrderedSet rb;
#else
//! Default 2-checkerboard (red/black) subset
extern UnorderedSet rb;
#endif


#if QDP_USE_CB32_LAYOUT == 1
//! Default 2^{Nd+1}-checkerboard subset. Useful for pure gauge updating.
extern OrderedSet mcb;
#else
//! Default 2^{Nd+1}-checkerboard subset. Useful for pure gauge updating.
extern UnorderedSet mcb;
#endif
    

#if QDP_USE_CB2_LAYOUT == 1
//! Default even subset
extern OrderedSubset even;
#else
//! Default even subset
extern UnorderedSubset even;
#endif



#if QDP_USE_CB2_LAYOUT == 1
//! Default odd subset
extern OrderedSubset odd;
#else
//! Default odd subset
extern UnorderedSubset odd;
#endif

/** @} */ // end of group subsetss

QDP_END_NAMESPACE();
