// $Id: subset.cc,v 1.7 2002-12-14 01:13:56 edwards Exp $
//
// QDP data parallel interface
//
// Sets and subsets

#include "qdp.h"
#include "proto.h"

QDP_BEGIN_NAMESPACE(QDP);

//! Default all set
Set set_all;

//! Default all subset
Subset all;

//! Default 2-checkerboard (red/black) set
Set rb;

//! Default 2^{Nd+1}-checkerboard set. Useful for pure gauge updating.
Set mcb;

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
void InitDefaultSets()
{
  // Initialize the red/black checkerboard
  rb.make(SetRBFunc());

    // Initialize the 32-style checkerboard
  mcb.make(Set32CBFunc());

  // The all set
  set_all.make(SetAllFunc());

  // The all subset
  all.make(set_all[0]);
}


//! Simple constructor called to produce a Subset from inside a Set
void Subset::make(int start, int end, bool rep, multi1d<int>* ind, int cb)
{
  startSite = start;
  endSite = end;
  indexrep = rep;
  sitetable = ind;
  sub_index = cb;
}


//-----------------------------------------------------------------------------
// Find these in the respective  architecture  *_specific.cc  files
//! Constructor from an int function
//void Set::make(const SetFunc& fn);

//! Initializer for sets
//void Set::InitOffsets();

QDP_END_NAMESPACE();
