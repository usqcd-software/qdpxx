// $Id: qdp_subset.cc,v 1.7 2007-02-16 22:22:21 bjoo Exp $
//
// QDP data parallel interface
//
// Sets and subsets

#include "qdp.h"
#include "qdp_util.h"

QDP_BEGIN_NAMESPACE(QDP);

//! Default all set
OrderedSet set_all;

//! Default all subset
OrderedSubset all;

//! Default rb3 subset -- Always unordered
UnorderedSet rb3;

#if QDP_USE_CB2_LAYOUT == 1
//! Default 2-checkerboard (red/black) set
OrderedSet rb;
#else
//! Default 2-checkerboard (red/black) set
UnorderedSet rb;
#endif


#if QDP_USE_CB32_LAYOUT == 1
//! Default 2^{Nd+1}-checkerboard set. Useful for pure gauge updating.
OrderedSet mcb;
#else
//! Default 2^{Nd+1}-checkerboard set. Useful for pure gauge updating.
UnorderedSet mcb;
#endif


#if QDP_USE_CB2_LAYOUT == 1
//! Even subset
OrderedSubset even;
#else
//! Even subset
UnorderedSubset even;
#endif


#if QDP_USE_CB2_LAYOUT == 1
//! Odd subset
OrderedSubset odd;
#else
//! Odd subset
UnorderedSubset odd;
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

	  

//! Simple constructor called to produce a Subset from inside a Set
void UnorderedSubset::make(bool _rep, int _start, int _end, multi1d<int>* ind, int cb)
{
  ordRep    = _rep;
  startSite = _start;
  endSite   = _end;
  sitetable = ind;
  sub_index = cb;
}

//! Simple constructor called to produce a Subset from inside a Set
void UnorderedSubset::make(const UnorderedSubset& s)
{
  ordRep    = s.ordRep;
  startSite = s.startSite;
  endSite   = s.endSite;
  sub_index = s.sub_index;
  sitetable = s.sitetable;
}

//! Simple constructor called to produce a Subset from inside a Set
UnorderedSubset& UnorderedSubset::operator=(const UnorderedSubset& s)
{
  make(s);
  return *this;
}

//! Simple constructor called to produce a Subset from inside a Set
void OrderedSubset::make(int _start, int _end, multi1d<int>* _ind, int _cb)
{
  startSite = _start;
  endSite   = _end;
  sitetable = _ind;
  sub_index = _cb;
}

//! Simple constructor called to produce a Subset from inside a Set
void OrderedSubset::make(const OrderedSubset& s)
{
  sub_index = s.sub_index;
  startSite = s.startSite;
  endSite   = s.endSite;
  sitetable = s.sitetable;
}

// = operator
OrderedSubset& OrderedSubset::operator=(const OrderedSubset& s)
{
  make(s);
  return *this;
}


// = operator
UnorderedSet& UnorderedSet::operator=(const UnorderedSet& s)
{
  sub = s.sub;
  lat_color = s.lat_color;
  sitetables = s.sitetables;
  return *this;
}

// = operator
OrderedSet& OrderedSet::operator=(const OrderedSet& s)
{
  sub = s.sub;
  lat_color = s.lat_color;
  sitetables = s.sitetables;
  return *this;
}

//-----------------------------------------------------------------------------

QDP_END_NAMESPACE();
