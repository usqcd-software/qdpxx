// $Id: qdp_subset.cc,v 1.3 2003-07-31 01:02:22 edwards Exp $
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

//! Default 2-checkerboard (red/black) set
UnorderedSet rb;

//! Default 2^{Nd+1}-checkerboard set. Useful for pure gauge updating.
UnorderedSet mcb;

//! Even subset
UnorderedSubset even;

//! Odd subset
UnorderedSubset odd;

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

QDP_END_NAMESPACE();
