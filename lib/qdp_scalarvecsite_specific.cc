/*! @file
 * @brief Scalarvec-like architecture specific routines
 * 
 * Routines common to all scalarvec-like architectures including 
 * scalarvec and parscalarvec
 */


#include "qdp.h"
#include "qdp_util.h"

namespace QDP {

//-----------------------------------------------------------------------------
namespace Layout
{
  //! coord[mu]  <- mu  : fill with lattice coord in mu direction
  /* Assumes no inner grid */
  LatticeInteger latticeCoordinate(int mu)
  {
    LatticeInteger d;

    if (mu < 0 || mu >= Nd)
      QDP_error_exit("dimension out of bounds");
    
    const int nodeSites = Layout::sitesOnNode();

    for(int i=0; i < nodeSites; ++i) 
    {
      Integer cc = Layout::siteCoords(Layout::nodeNumber(),i)[mu];
      int iouter = i >> INNER_LOG;
      int iinner = i & ((1 <<INNER_LOG)-1);
      copy_site(d.elem(iouter), iinner, cc.elem());
    }
    
    return d;
  }
}


//-----------------------------------------------------------------------------
// IO routine solely for debugging. Only defined here
template<class T>
ostream& operator<<(ostream& s, const multi1d<T>& s1)
{
  for(int i=0; i < s1.size(); ++i)
    s << " " << s1[i];

  return s;
}


//-----------------------------------------------------------------------------
//! Constructor from a function object
void Set::make(const SetFunc& func)
{
  // Number of subsets
  int nsubset_indices = func.numSubsets();

  // Number of sites on node
  const int nodeSites = Layout::sitesOnNode();

  // My node number 
  const int nodeNumber = Layout::nodeNumber();
	
#if QDP_DEBUG >= 2
  QDP_info("Set a subset: nsubset = %d",nsubset_indices);
#endif

  // This actually allocates the subsets
  sub.resize(nsubset_indices);

  // Create the space of the colorings of the lattice
  lat_color.resize(nodeSites);

  // Create the array holding the array of sitetable info
  sitetables.resize(nsubset_indices);

  // Create the array holding the array of membertable info
  membertables.resize(nsubset_indices);
  blocktables.resize(nsubset_indices);
  masktables.resize(nsubset_indices);

  // Loop over linear sites determining their color
  // This can be done in parallel I think
#pragma omp parallel for
  for(int linear=0; linear < nodeSites; ++linear) {
    multi1d<int> coord = Layout::siteCoords(Layout::nodeNumber(), linear);

    int node   = Layout::nodeNumber(coord);
    int lin    = Layout::linearSiteIndex(coord);
    int icolor = func(coord);

#if QDP_DEBUG >= 3
    cout << " coord="<<coord<<" node="<<node<<" linear="<<linear<<" col="<<icolor << endl;
#endif

    // Sanity checks
    if (node != nodeNumber)
      QDP_error_exit("Set: found site with node outside current node!");

    if (lin != linear)
      QDP_error_exit("Set: inconsistent linear sites");

    if (icolor < 0 || icolor >= nsubset_indices)
      QDP_error_exit("Set: coloring is outside legal range: color[%d]=%d",linear,icolor);

    // The coloring of this linear site
    lat_color[linear] = icolor;
  }


  /*
   * Loop over the lexicographic sites.
   * This implementation of the Set will always use a
   * sitetable.
   */
  
  // Loop over subsets
  for(int cb=0; cb < nsubset_indices; ++cb) {
    
    // Always construct the sitetables. 
    multi1d<bool>& membertable = membertables[cb];
    membertable.resize(nodeSites);
   
    
    // This loop goes through the lat_color table we constructed
    // previously. If the color matches the current working color (cb)
    // we increase the count for this color, and we set a 'true' flag in 
    // membertable
    int num_sitetable = 0;
    for(int linear=0; linear < nodeSites; ++linear) {
      if (lat_color[linear] == cb) {
      	++num_sitetable;	
	membertable[linear] = true;
      }
      else {
	membertable[linear] = false;
      }
    }

    // At this point: num_sitetable is the number of sites of this
    // color
    // membertable for the subset has a 'true' tick for every  
    // site that is in our subset.

    // Now we take the member table and we go through it finding all 
    // the true bits and putting it into the site table
    multi1d<int>& sitetable = sitetables[cb];

    sitetable.resize(num_sitetable); // resize the sitetable to the correct number of members.


    // Site ordering stuff for later
    bool ordRep;
    int start, end;
    int startBlock, endBlock;

    // Handle the case that there are no sites
    if (num_sitetable > 0) {


      // Initialize the sitetable to all -1s 
      for(int i=0; i < num_sitetable; ++i) {
	sitetable[i] = -1;
      }

      // Scan through lat_color table  adding in the linear sites
      // We could've also done this with the membertable I guess. 
      // NB: This step is not easy to parallelize since it essentially
      // A histogramming excercise. 
      for(int linear=0, j=0; linear < nodeSites; ++linear) {
	if (lat_color[linear] == cb) {
	  sitetable[j++] = linear;
	}
      }


      // Check *if* this coloring is contiguous and find the start
      // and ending sites
      ordRep = true;
      start = sitetable[0];   // this is the beginning
      end = sitetable[sitetable.size()-1];  // the absolute last site

      // Now look for a hole
      for(int prev=sitetable[0], i=0; i < sitetable.size(); ++i){
	if (sitetable[i] != prev++)
	  {
#if QDP_DEBUG >= 2
	    QDP_info("Set(%d): sitetable[%d]=%d",cb,i,sitetable[i]);
#endif
	    
	    // Found a hole. The rep is not ordered.
	    ordRep = false;
	    start = end = -1;
	    break;
	  }
      }
    }
    else { // num_sitetable == 0
      ordRep = false;
      start = end = -1;
    }

    // Scalarvecsite Specific things. 
    if ( ordRep ) {

      // If we are ordered, check we start and on vector boundaries.
      int startInner = start & (INNER_LEN - 1);
      int endInner = end & (INNER_LEN - 1);

      // If we dont start/end on vector boundaries throw a fit and barf
      if ( startInner != 0 ) { 
	QDP_info("Subset (%d) is contiguous, but does not start on vector boundary. Inner site of start is %d. INNER_LEN is %d\n", cb, startInner, INNER_LEN);
	QDP_abort(1);
      }

      if ( endInner != ( INNER_LEN - 1) ) {
	QDP_info("Subset (%d) is contiguous, but does not end on a vectory boundary. Inner site of end is %d. INNER_LEN is %d\n", cb, endInner, INNER_LEN);
	QDP_abort(1);
      }

      // Otherwise all is good...
      // Set start/end block members
      startBlock = start >> INNER_LOG;
      endBlock = end >> INNER_LOG;
    }
    else { 
      // Set these to some nonsense value
      startBlock = -1;
      endBlock = -1;
    }


    // Now make block and mask tables
    int numBlocks = 0; // we will count this.

    // The block coordinate of the last site. 
    int maxBlocks = Layout::sitesOnNode() >> INNER_LOG;

    // A temporary integer array. This will hold whether a block is in a subset or not.
    // Use an integer rather than a boolean so that we can sum the array at the end.
    multi1d<int> memberBlocks(maxBlocks);


    // This loop is over blocks, so each thread does its own block
    // It checks whether the block is a member of the subset
    // If it is it puts a '1' in the member blocks table otherwise that is initialized to '0'
    // Then a reduction over the memberBlocks should count the number of blocks
#pragma omp parallel for shared(memberBlocks, membertable) reduction(+:numBlocks)
    for(int block = 0; block < maxBlocks; block++) { 
      memberBlocks[block] = 0;

      // Loop through the inner sites and check if they are in the
      // subset. Leave the loop after finding the first one.
      // Also leave the loop if the linear index is bigger than the local
      // volume (in case the last vector is not filled).

      bool finished = false;
      for(int inner=0; inner < INNER_LEN && (!finished) ; inner++) { 
	
	int linear = block * INNER_LEN + inner;  // Linear Site

	if (linear < Layout::sitesOnNode()) {

	  // Check the site is a member. If it is set memberBlocks to '1'
	  if( membertable[linear] ) { 
	    finished = true;
	    memberBlocks[block] = 1;
	  }
	}
	else { 
	  // Site is 'out of range'. Since this happens on the inner 
	  // loop which is not parallel, all the previous sites have
	  // been checked if the block is a member we'd have exited already.
	  // while later sites would be even further out of range
	  finished = true;
	}
      } //for inner < INNER_LEN && (!found) 

      // Reduction: OpenMP can take care of this.
      numBlocks+= memberBlocks[block];

    } // parallel for blocks < maxBlocsk


    multi1d<int>& blocktable = blocktables[cb];
    multi1d<ILattice<bool,INNER_LEN> >& masktable = masktables[cb];

    // Resize the block and mask tables
    blocktable.resize(numBlocks); 
    masktable.resize(numBlocks);


    // Now this part sadly cannot be parallelized so easily, as it is a histogramming
    // thingie, so no OMP For here

    // We loop through the blocks, and if it is a member we add it to the block_table.
    // We work out the site-mask for it and add this to the block table also
    for(int block=0, j=0; block < maxBlocks; block++) {

      if ( memberBlocks[block] ) { 

	// Add block to block table at position 'j'
	blocktable[j] = block;

	// Now construct the mask for this block
	ILattice<bool, INNER_LEN> blockmask;

	// Initialize the mask to false.
	// Do this separately, since in the next loop
	// we will set mask bits depending on a linear site index
	// which may end up being bigger than the lattice volume if we 
	// don't exactly fill vectors. So we may not process the last
	// few bits, and then we should have them pre-set to false
	for(int inner=0; inner < INNER_LEN; inner++) { 
	  blockmask.elem(inner) = false;
	}

	// Set the mask bits as appropriate. 
	for(int inner=0; inner < INNER_LEN ; inner++) { 
	 
	  // Get the full linear address
	  int linear = inner + block*INNER_LEN;

	  if( linear < Layout::sitesOnNode() ) { 
	    if ( membertable[linear] ) { 
	      // If it is a member set the mask bit to true
	      blockmask.elem(inner) = true;
	    }
	  }

	}

	// Copy into masktable
	masktable[j] = blockmask;
	j++;
      }
    }
    
	

    sub[cb].make(ordRep, start, end, &(sitetables[cb]), cb, this, &(membertables[cb]), 
		 startBlock, 
		 endBlock,
		 &(blocktables[cb]),
		 &(masktables[cb]));

#if QDP_DEBUG >= 2
    QDP_info("Subset(%d): num_sitetable=%d  start=%d end=%d",cb,num_sitetable,start,end);
#endif
  }
}


} // namespace QDP;
