// -*- C++ -*-
// $Id: subset.h,v 1.3 2002-09-26 21:30:07 edwards Exp $
//
// QDP data parallel interface
//
QDP_BEGIN_NAMESPACE(QDP);

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
    indexrep(s.indexrep), soffsets(s.soffsets), sitetable(s.sitetable) {}

  // Simple constructor
  void Make(const Subset& s)
    {
      startSite = s.startSite;
      endSite = s.endSite;
      indexrep = s.indexrep;
      soffsets = s.soffsets;
      sitetable = s.sitetable;
    }

  //! Destructor for a subset
  ~Subset() {}

  //! Access the coloring for this subset
  const int Index() const {return sub_index;}

protected:
  // Simple constructor
  void Make(int start, int end, bool rep, multi3d<int>* soff, multi1d<int>* ind, int cb);


private:
  int startSite;
  int endSite;
  int sub_index;
  bool indexrep;

  //! Site lookup table
  multi1d<int>* sitetable;

  //! Offset table used for communications. 
  /*! 
   * The direction is in the sense of the Map or Shift functions from QDP.
   * soffsets(direction,isign,position) 
   */ 
  multi3d<int>* soffsets;


public:
  // These should be a no-no. Must fix the friend syntax
  /*! Is the representation a boolean mask? */
  const bool IndexRep() const {return indexrep;}

  const int Start() const {return startSite;}
  const int End() const {return endSite;}
  const multi1d<int>* SiteTable() const {return sitetable;}
  const int NumSiteTable() const {return sitetable->size();}
  const multi3d<int>* Offsets() const {return soffsets;}

  friend class Set;
};


//! Set - collection of subsets controlling which sites are involved in an operation
class Set
{
public:
  //! There can be an empty constructor
  Set() {}

  //! Constructor from an int function
  void Make(int (&func)(const multi1d<int>& coordinate), int nsubset_indices);

  //! Constructor from an int function
  Set(int (&func)(const multi1d<int>& coordinate), int nsubset_indices)
    {Make(func,nsubset_indices);}

  //! Index operator selects a subset from a set
  const Subset& operator[](int subset_index) const {return sub[subset_index];}

  //! Return number of subsets
  const int NumSubsets() const {return sub.size();}

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

  //! Offset table used for communications. 
  /*! 
   * The direction is in the sense of the Map or Shift functions from QDP.
   * soffsets(direction,isign,position) 
   *
   * NOTE: this should be moved off to a shift class
   */ 
  multi3d<int> soffsets;



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
    
// Forward declaration
class Context;

//! Global context
extern Context *global_context;

/** @defgroup group3 Contexts and subsets
   *  
   *  Sets the context for all subsequent operations.
   *  The semantics is stack based and context are restored upon
   *  exiting an enclosing block of the declaration
   *
   *  @{
   */
  //! Context - simple class with a declaration having a side effect of setting default subset
  /*!
   *  Sets the context for all subsequent operations.
   *  The semantics is stack based and context are restored upon
   *  exiting an enclosing block of the declaration
   *
   *  To set a context, a user declares a Context like
   *  Context foo(rb[0]);
   *  which set the context to be the even of the red-black checkerboards
   */
class Context 
{
public:
  //! Constructor from a subset
  Context(const Subset& s): sub(s), prev(global_context)
    {global_context = this;}

  //! Destructor pops context
  ~Context() {global_context = prev;}

  //! Return the subset for this context
  const Subset& Sub() const {return sub;}

private:
  Context(): sub(all) {}

private:
  const Subset& sub;
  Context *prev;
};
/** @} */ // end of group3

QDP_END_NAMESPACE();
