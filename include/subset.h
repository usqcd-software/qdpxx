// -*- C++ -*-
// $Id: subset.h,v 1.9 2002-11-23 03:42:30 edwards Exp $

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

// Forward declaration
class Set;

//! Subsets - controls how lattices are looped
class Subset
{
public:
  //! There can be an empty constructor
  Subset() {}

  //! Copy constructor
  Subset(const Subset& s): startSite(s.startSite), endSite(s.endSite), 
    indexrep(s.indexrep), sitetable(s.sitetable) {}

  // Simple constructor
  void make(const Subset& s)
    {
      startSite = s.startSite;
      endSite = s.endSite;
      indexrep = s.indexrep;
      sitetable = s.sitetable;
    }

  //! Destructor for a subset
  ~Subset() {}

  //! Access the coloring for this subset
  int Index() const {return sub_index;}

protected:
  // Simple constructor
  void make(int start, int end, bool rep, multi1d<int>* ind, int cb);


private:
  int startSite;
  int endSite;
  int sub_index;
  bool indexrep;

  //! Site lookup table
  multi1d<int>* sitetable;

public:
  // These should be a no-no. Must fix the friend syntax
  /*! Is the representation a boolean mask? */
  bool IndexRep() const {return indexrep;}

  int Start() const {return startSite;}
  int End() const {return endSite;}
  const multi1d<int>* SiteTable() const {return sitetable;}
  int NumSiteTable() const {return sitetable->size();}

  friend class Set;
};


//! Set - collection of subsets controlling which sites are involved in an operation
class Set
{
public:
  //! There can be an empty constructor
  Set() {}

  //! Constructor from an int function
  void make(int (&func)(const multi1d<int>& coordinate), int nsubset_indices);

  //! Constructor from an int function
  Set(int (&func)(const multi1d<int>& coordinate), int nsubset_indices)
    {make(func,nsubset_indices);}

  //! Index operator selects a subset from a set
  const Subset& operator[](int subset_index) const {return sub[subset_index];}

  //! Return number of subsets
  int NumSubsets() const {return sub.size();}

  //! Destructor for a set
  ~Set() {}

protected:
  //! Initializer for sets
  void InitOffsets();
    

protected:
  //! A set is composed of an array of subsets
  multi1d<Subset> sub;

  //! Index or color array of lattice
  multi1d<int> lat_color;

  //! Array of sitetable arrays
  multi1d<multi1d<int> > sitetables;

public:
  // These should be a no-no. Must fix the friend syntax
  const multi1d<int>& LatticeColoring() const {return lat_color;}
};



//! Default all subset
extern Subset all;

//! Default 2-checkerboard (red/black) subset
extern Set rb;

//! Default 2^{Nd+1}-checkerboard subset. Useful for pure gauge updating.
extern Set mcb;
    
/** @} */ // end of group subsetss

QDP_END_NAMESPACE();
