// $Id: subset.cc,v 1.5 2002-10-28 03:08:44 edwards Exp $
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

//! Function used for constructing the all subset
int subset_all_func(const multi1d<int>& coordinate) {return 0;}
  
//! Function used for constructing red-black (2) checkerboard */
int subset_rb_func(const multi1d<int>& coordinate)
{
  int sum = 0;
  for(int m=0; m < coordinate.size(); ++m)
    sum += coordinate[m];

  return sum & 1;
}
  
//! Function used for constructing 32 checkerboard. */
int subset_32cb_func(const multi1d<int>& coordinate)
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
  

//! Initializer for sets
void InitDefaultSets()
{
  // Initialize the red/black checkerboard
  rb.Make(subset_rb_func, 2);

    // Initialize the 32-style checkerboard
  mcb.Make(subset_32cb_func, 1 << (Nd+1));

  // The all set
  set_all.Make(subset_all_func, 1);

  // The all subset
  all.Make(set_all[0]);
}


//! Simple constructor called to produce a Subset from inside a Set
void Subset::Make(int start, int end, bool rep, multi3d<int>* soff, 
		  multi1d<int>* ind, int cb)
{
  startSite = start;
  endSite = end;
  indexrep = rep;
  soffsets = soff;
  sitetable = ind;
  sub_index = cb;
}


//-----------------------------------------------------------------------------
// Find these in the respective  architecture  *_specific.cc  files
//! Constructor from an int function
//void Set::Make(int (&func)(const multi1d<int>& coordinate), int nsubset_indices);

//! Initializer for sets
//void Set::InitOffsets();

QDP_END_NAMESPACE();
