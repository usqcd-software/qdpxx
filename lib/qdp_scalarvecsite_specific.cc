// $Id: qdp_scalarvecsite_specific.cc,v 1.5 2003-08-29 04:13:13 edwards Exp $

/*! @file
 * @brief Scalarvec-like architecture specific routines
 * 
 * Routines common to all scalarvec-like architectures including 
 * scalarvec and parscalarvec
 */


#include "qdp.h"
#include "qdp_util.h"

QDP_BEGIN_NAMESPACE(QDP);

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

    for(int i=0; i < Layout::sitesOnNode(); ++i) 
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
void UnorderedSet::make(const SetFunc& func)
{
  int nsubset_indices = func.numSubsets();

#if QDP_DEBUG >= 2
  QDP_info("UnorderedSet a subset: nsubset = %d",nsubset_indices);
#endif

  // This actually allocates the subsets
  sub.resize(nsubset_indices);

  // Create the space of the colorings of the lattice
  lat_color.resize(Layout::sitesOnNode());

  // Create the array holding the array of sitetable info
  sitetables.resize(nsubset_indices);

  // Loop over linear sites determining their color
  for(int linear=0; linear < Layout::sitesOnNode(); ++linear)
  {
    multi1d<int> coord = Layout::siteCoords(Layout::nodeNumber(), linear);

    int node   = Layout::nodeNumber(coord);
    int lin    = Layout::linearSiteIndex(coord);
    int icolor = func(coord);

#if QDP_DEBUG >= 3
    cerr << " coord="<<coord<<" node="<<node<<" linear="<<linear<<" col="<<icolor << endl;
#endif

    // Sanity checks
    if (node != Layout::nodeNumber())
      QDP_error_exit("UnorderedSet: found site with node outside current node!");

    if (lin != linear)
      QDP_error_exit("UnorderedSet: inconsistent linear sites");

    if (icolor < 0 || icolor >= nsubset_indices)
      QDP_error_exit("UnorderedSet: coloring is outside legal range: color[%d]=%d",linear,icolor);

    // The coloring of this linear site
    lat_color[linear] = icolor;
  }


  /*
   * Loop over the lexicographic sites.
   * This implementation of the UnorderedSet will always use a
   * sitetable.
   */
  for(int cb=0; cb < nsubset_indices; ++cb)
  {
    // Always construct the sitetables. 

    // First loop and see how many sites are needed
    int num_sitetable = 0;
    for(int linear=0; linear < Layout::sitesOnNode(); ++linear)
      if (lat_color[linear] == cb)
	++num_sitetable;

    // Now take the inverse of the lattice coloring to produce
    // the site list
    multi1d<int>& sitetable = sitetables[cb];
    sitetable.resize(num_sitetable);

    for(int linear=0, j=0; linear < Layout::sitesOnNode(); ++linear)
      if (lat_color[linear] == cb)
	sitetable[j++] = linear;


    sub[cb].make(&(sitetables[cb]), cb);

#if QDP_DEBUG >= 2
    QDP_info("UnorderedSubset(%d)",cb);
#endif
  }
}
	  

//-----------------------------------------------------------------------------
//! Constructor from a function object
void OrderedSet::make(const SetFunc& func)
{
  int nsubset_indices = func.numSubsets();

#if QDP_DEBUG >= 2
  QDP_info("OrderedSet a subset: nsubset = %d",nsubset_indices);
#endif

  // This actually allocates the subsets
  sub.resize(nsubset_indices);

  // Create the space of the colorings of the lattice
  lat_color.resize(Layout::sitesOnNode());

  // Create the array holding the array of sitetable info
  sitetables.resize(nsubset_indices);

  // Loop over linear sites determining their color
  for(int linear=0; linear < Layout::sitesOnNode(); ++linear)
  {
    multi1d<int> coord = Layout::siteCoords(Layout::nodeNumber(), linear);

    int node   = Layout::nodeNumber(coord);
    int lin    = Layout::linearSiteIndex(coord);
    int icolor = func(coord);

#if QDP_DEBUG >= 3
    cerr << " coord="<<coord<<" node="<<node<<" linear="<<linear<<" col="<<icolor << endl;
#endif

    // Sanity checks
    if (node != Layout::nodeNumber())
      QDP_error_exit("OrderedSet: found site with node outside current node!");

    if (lin != linear)
      QDP_error_exit("OrderedSet: inconsistent linear sites");

    if (icolor < 0 || icolor >= nsubset_indices)
      QDP_error_exit("UnorderedSet: coloring is outside legal range: color[%d]=%d",linear,icolor);

    // The coloring of this linear site
    lat_color[linear] = icolor;
  }


  /*
   * Loop over the lexicographic sites.
   * Check if the linear sites are in a contiguous set.
   * This implementation only supports a single contiguous
   * block of sites.
   */
  for(int cb=0; cb < nsubset_indices; ++cb)
  {
    // Always construct the sitetables. 

    // First loop and see how many sites are needed
    int num_sitetable = 0;
    for(int linear=0; linear < Layout::sitesOnNode(); ++linear)
      if (lat_color[linear] == cb)
	++num_sitetable;

    // Now take the inverse of the lattice coloring to produce
    // the site list
    multi1d<int>& sitetable = sitetables[cb];
    sitetable.resize(num_sitetable);

    for(int linear=0, j=0; linear < Layout::sitesOnNode(); ++linear)
      if (lat_color[linear] == cb)
	sitetable[j++] = linear;

    // Now check that this coloring is contiguous and find the start
    // and ending sites
    int start = sitetable[0];   // this is the beginning
    int end = sitetable[sitetable.size()-1];  // the absolute last site

    // Now look for a hole
    for(int prev=sitetable[0], i=0; i < sitetable.size(); ++i)
      if (sitetable[i] != prev++)
      {
#if QDP_DEBUG >= 2
	QDP_info("OrderedSet(%d): sitetable[%d]=%d",cb,i,sitetable[i]);
#endif
	
	// This is a fatal run-time error. Found that sites are not contiguous
	QDP_error_exit("OrderedSet: found a subset with sites not contiguous in violation of the semantics imposed by this class. Maybe this object should have been an UnorderedSet.");
      }

    // Make the subset
    sub[cb].make(start, end, &(sitetables[cb]), cb);

#if QDP_DEBUG >= 2
    QDP_info("OrderedSubset(%d): start=%d end=%d",cb,start,end);
#endif
  }
}


//-----------------------------------------------------------------------------
//! Function overload read of  int
void read(NmlReader& nml, const string& s, int& d)
{
  if (Layout::primaryNode()) 
    param_int_array(&d, get_current_nml_section(), s.c_str(), 0);

  // Now broadcast back out to all nodes
  Internal::broadcast(d);
}

//! Function overload read of  float
void read(NmlReader& nml, const string& s, float& d)
{
  if (Layout::primaryNode()) 
    param_float_array(&d, get_current_nml_section(), s.c_str(), 0);

  // Now broadcast back out to all nodes
  Internal::broadcast(d);
}

//! Function overload read of  double
void read(NmlReader& nml, const string& s, double& d)
{
  if (Layout::primaryNode()) 
    param_double_array(&d, get_current_nml_section(), s.c_str(), 0);

  // Now broadcast back out to all nodes
  Internal::broadcast(d);
}

//! Function overload read of  bool
void read(NmlReader& nml, const string& s, bool& d)
{
  if (Layout::primaryNode()) 
  {
    int dd;
    param_bool_array(&dd, get_current_nml_section(), s.c_str(), 0);
    d = (dd == 0) ? false : true;
  }

  // Now broadcast back out to all nodes
  Internal::broadcast(d);
}

//! Function overload read of  string
void read(NmlReader& nml, const string& s, string& d)
{
  char *dd_tmp;
  int lleng;

  // Only primary node can grab string
  if (Layout::primaryNode()) 
  {
    dd_tmp = param_string_array(get_current_nml_section(), s.c_str(), 0);
    lleng = strlen(dd_tmp) + 1;
  }

  // First must broadcast size of string
  Internal::broadcast(lleng);

  // Now every node can alloc space for string
  if (! Layout::primaryNode()) 
    dd_tmp = new char[lleng];
  
  // Now broadcast char array out to all nodes
  Internal::broadcast((void *)dd_tmp, lleng);

  // All nodes can now grab char array and make a string
  d = dd_tmp;

  // Clean-up and boogie
  if (! Layout::primaryNode()) 
    delete[] dd_tmp;
}



//! Function overload read of  int  into element position n
void read(NmlReader& nml, const string& s, int& d, int n)
{
  if (Layout::primaryNode()) 
    param_int_array(&d, get_current_nml_section(), s.c_str(), 1, n);

  // Now broadcast back out to all nodes
  Internal::broadcast(d);
}

//! Function overload read of  float  into element position n
void read(NmlReader& nml, const string& s, float& d, int n)
{
  if (Layout::primaryNode()) 
    param_float_array(&d, get_current_nml_section(), s.c_str(), 1, n);

  // Now broadcast back out to all nodes
  Internal::broadcast(d);
}

//! Function overload read of  double  into element position n
void read(NmlReader& nml, const string& s, double& d, int n)
{
  if (Layout::primaryNode()) 
    param_double_array(&d, get_current_nml_section(), s.c_str(), 1, n);

  // Now broadcast back out to all nodes
  Internal::broadcast(d);
}

//! Function overload read of  bool  into element position n
void read(NmlReader& nml, const string& s, bool& d, int n)
{
  if (Layout::primaryNode()) 
  {
    int dd;
    param_bool_array(&dd, get_current_nml_section(), s.c_str(), 1, n);
    d = (dd == 0) ? false : true;
  }

  // Now broadcast back out to all nodes
  Internal::broadcast(d);
}


//! Function overload read of  Complex
void read(NmlReader& nml, const string& s, Complex& d)
{
  if (Layout::primaryNode()) 
  {
    WordType<Complex>::Type_t  dre, dim;
    param_complex_float_array(&dre, &dim, get_current_nml_section(), s.c_str(), 0);

    Real Dre(dre);
    Real Dim(dim);
    d = cmplx(Dre,Dim);
  }

  // Now broadcast back out to all nodes
  Internal::broadcast(d);
}


//! Function overload read of  Seed
void read(NmlReader& nml, const string& s, Seed& d)
{
  if (Layout::primaryNode()) 
  {
    int ss[4];

    // Snarf all 4 ints used to serialize a seed
    for(int n=0; n < 4; ++n)
      param_int_array(ss+n, get_current_nml_section(), s.c_str(), 1, n);

    // Taken from random.cc - a platform independent (peculiar) way to load up a seed
    Seed seed_tmp3;
    Seed seed_tmp2;
    Seed seed_tmp1;
    Seed seed_tmp0;

    seed_tmp3 = ss[3];
    seed_tmp2 = (seed_tmp3 << 12) | ss[2];
    seed_tmp1 = (seed_tmp2 << 12) | ss[1];
    seed_tmp0 = (seed_tmp1 << 12) | ss[0];

    d = seed_tmp0;
  }

  // Now broadcast back out to all nodes
  Internal::broadcast(d);
}


QDP_END_NAMESPACE();
