/*! @file
 * @brief Sets and subsets
 */

#include "qdp.h"
#include "qdp_util.h"

namespace QDP 
{

  //---------------------------------------------------------------------
  //! Default all set
  Set set_all;

  //! Default all subset
  Subset all;

  //! Default rb3 subset -- Always unordered
  Set rb3;

  //! Default 2-checkerboard (red/black) set
  Set rb;

  //! Default 2^{Nd+1}-checkerboard set. Useful for pure gauge updating.
  Set mcb;

  //! Even subset
  Subset even;

  //! Odd subset
  Subset odd;

  Set::~Set() {
#ifdef QDP_IS_QDPJIT
    if (registered) {
      QDP_debug("Set::~Set: Strided:  Will sign off now...");
      QDPCache::Instance().signoff( idStrided );
    }
#endif
  }


#ifdef QDP_IS_QDPJIT
  Subset::Subset():registered(false) {
    QDPCache::Instance().sayHi();
  }
#else
  Subset::Subset() {}
#endif

  //! Copy constructor
#ifdef QDP_IS_QDPJIT
  Subset::Subset(const Subset& s):
    ordRep(s.ordRep), startSite(s.startSite), endSite(s.endSite), 
    sub_index(s.sub_index), sitetable(s.sitetable), set(s.set) , registered(false) { 
    QDPCache::Instance().sayHi();
  }
#else
  Subset::Subset(const Subset& s):
    ordRep(s.ordRep), startSite(s.startSite), endSite(s.endSite), 
    sub_index(s.sub_index), sitetable(s.sitetable), set(s.set)  { }
#endif


  Subset::~Subset()
  {
#ifdef QDP_IS_QDPJIT
    if (registered) {
      QDP_debug("Subet::~Subset: Will sign off now...");
      QDPCache::Instance().signoff( idStrided );
    }
#endif
  }

#ifdef QDP_IS_QDPJIT
  Set::Set(): registered(false) {
    QDPCache::Instance().sayHi();
  }
#else
  Set::Set() {}
#endif

#ifdef QDP_IS_QDPJIT
  //! Constructor from a function object
  Set::Set(const SetFunc& fn): registered(false) {
    QDPCache::Instance().sayHi();
    make(fn);    
  }
#else
  Set::Set(const SetFunc& fn) {
    make(fn);    
  }
#endif


  //! Function object used for constructing the all subset
  class SetAllFunc : public SetFunc
  {
  public:
    int operator() (const multi1d<int>& coordinate) const {return 0;}
    int numSubsets() const {return 1;}
  };

  
  //! Function object used for constructing red-black (2) checkerboard */
  class SetRBFunc : public SetFunc
  {
  public:
    int operator() (const multi1d<int>& coordinate) const
      {
	int sum = 0;
	for(int m=0; m < coordinate.size(); ++m)
	  sum += coordinate[m];

	return sum & 1;
      }

    int numSubsets() const {return 2;}
  };

  //! Function object used for constructing red-black (2) checkerboard in 3d
  class SetRB3Func : public SetFunc
  {
  public:
    int operator() (const multi1d<int>& coordinate) const
      {
	if (coordinate.size() < 3) { 
	  QDPIO::cerr << "Need at least 3d for 3d checkerboarding" << endl;
	  QDP_abort(1);
	}
	int sum = 0;
	for(int m=0; m < 3; ++m)
	  sum += coordinate[m];

	return sum & 1;
      }

    int numSubsets() const {return 2;}
  };

  
  //! Function object used for constructing 32 checkerboard. */
  class Set32CBFunc : public SetFunc
  {
  public:
    int operator() (const multi1d<int>& coordinate) const
      {
	int initial_color = 0;
	for(int m=Nd-1; m >= 0; --m)
	  initial_color = (initial_color << 1) + (coordinate[m] & 1);

	int cb = 0;
	for(int m=0; m < Nd; ++m)
	  cb += coordinate[m] >> 1;

	cb &= 1;
	return initial_color + (cb << Nd);
      }

    int numSubsets() const {return 1 << (Nd+1);}
  };


  //! Initializer for sets
  void initDefaultSets()
  {
    // Initialize the red/black checkerboard
    rb.make(SetRBFunc());

    // Initialize the 3d red/black checkerboard.
    rb3.make(SetRB3Func());

    // Initialize the 32-style checkerboard
    mcb.make(Set32CBFunc());

    // The all set
    set_all.make(SetAllFunc());

    // The all subset
    all.make(set_all[0]);

    // COPY the rb[0] to the even subset
    even = rb[0];

    // COPY the rb[1] to the odd subset
    odd = rb[1];
  }

	  
  //-----------------------------------------------------------------------------
  //! Simple constructor called to produce a Subset from inside a Set
  void Subset::make(bool _rep, int _start, int _end, multi1d<int>* ind, int cb, Set* _set)
  {
#ifdef QDP_IS_QDPJIT
    QDP_debug("Subset::make(...) Will reserve device memory now...");
#endif

    ordRep    = _rep;
    startSite = _start;
    endSite   = _end;
    sub_index = cb;
    sitetable = ind;
    set       = _set;

#ifdef QDP_IS_QDPJIT
    if (ind->size() == 0) 
      QDP_debug("At least one subset has zero size on at least one node. (rep=%d,start=%d,end=%d)",
	       (int)ordRep,(int)startSite,(int)endSite);
    else {
      if (registered) {
	QDP_info("Subset::make:  Already registered, will sign off the old memory ...");
	QDPCache::Instance().signoff( idStrided );
      }
      QDP_debug("Subset::make: Will register memory now...");
      idStrided = QDPCache::Instance().registrateOwnHostMem( ind->size() * sizeof(int) , (void*)ind->slice() );
      registered=true;
    }
#endif

  }

  //! Simple constructor called to produce a Subset from inside a Set
  void Subset::make(const Subset& s)
  {
#ifdef QDP_IS_QDPJIT
    QDP_debug("Subset::make(Subset) Will reserve device memory now...");
#endif

    ordRep    = s.ordRep;
    startSite = s.startSite;
    endSite   = s.endSite;
    sub_index = s.sub_index;
    sitetable = s.sitetable;
    set       = s.set;

#ifdef QDP_IS_QDPJIT
    if (s.sitetable->size() == 0)
      QDP_debug("At least one subset has zero size on at least one node. (Subset,rep=%d,start=%d,end=%d)",
	       (int)ordRep,(int)startSite,(int)endSite);
    else {
      if (registered) {
	QDP_info("Subset::make:  Already registered, will sign off the old memory ...");
	QDPCache::Instance().signoff( idStrided );
      }
      QDP_debug("Subset::make: Will register memory now...");
      idStrided = QDPCache::Instance().registrateOwnHostMem( s.sitetable->size() * sizeof(int) , (void*)s.sitetable->slice() );
      registered=true;
    }
#endif
  }

  //! Simple constructor called to produce a Subset from inside a Set
  Subset& Subset::operator=(const Subset& s)
  {
    make(s);
    return *this;
  }

  //-----------------------------------------------------------------------------
  // = operator
  Set& Set::operator=(const Set& s)
  {
    sub = s.sub;
    lat_color = s.lat_color;
    sitetables = s.sitetables;
#ifdef QDP_IS_QDPJIT
    QDP_error_exit("Sub::op= not yet implemented for GPU 3");
#endif

    return *this;
  }

  //-----------------------------------------------------------------------------

} // namespace QDP;
