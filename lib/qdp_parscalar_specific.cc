// $Id: qdp_parscalar_specific.cc,v 1.3 2003-06-05 04:15:55 edwards Exp $

/*! @file
 * @brief Parscalar specific routines
 * 
 * Routines for parscalar implementation
 */


#include "qdp.h"
#include "qdp_util.h"
#include "qmp.h"

QDP_BEGIN_NAMESPACE(QDP);

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
  int nsubset_indices = func.numSubsets();

#if defined(DEBUG)
  QDP_info("Set a subset: nsubset = %d",nsubset_indices);
#endif

  // This actually allocates the subsets
  sub.resize(nsubset_indices);

  // Create the space of the colorings of the lattice
  lat_color.resize(Layout::sitesOnNode());

  // Create the array holding the array of sitetable info
  sitetables.resize(nsubset_indices);

  // For a sanity check, set to some invalid value
  for(int site=0; site < Layout::sitesOnNode(); ++site)
    lat_color[site] = -1;

  // Loop over all sites determining their color
  for(int site=0; site < Layout::sitesOnNode(); ++site)
  {
    multi1d<int> coord = crtesn(site, Layout::subgridLattSize());

    for(int m=0; m<Nd; ++m)
      coord[m] += Layout::nodeCoord()[m]*Layout::subgridLattSize()[m];

    int node   = Layout::nodeNumber(coord);
    int linear = Layout::linearSiteIndex(coord);
    int icolor = func(coord);

//  cerr << "site="<<site<<" coord="<<coord<<" node="<<node<<" linear="<<linear<<" col="<<icolor;
//  cerr << endl;

    if (node != Layout::nodeNumber())
      QDP_error_exit("Set: found site with node outside current node!");

    lat_color[linear] = icolor;
  }

  // Loop over all sites and see if coloring properly set
  for(int site=0; site < Layout::sitesOnNode(); ++site)
  {
    if (lat_color[site] == -1)
      QDP_error_exit("Set: found site with coloring not set");
  }


  /*
   * Loop over the lexicographic sites.
   * Check if the linear sites are in a contiguous set.
   * This implementation only supports a single contiguous
   * block of sites.
   */
  for(int cb=0; cb < nsubset_indices; ++cb)
  {
    bool indexrep = true;
    int start = 0;
    int end = -1;

    // Always construct the sitetables. This could be moved into
    // the found_gap and only initialized if the interval method 
    // was not possible

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


    sub[cb].make(start, end, indexrep, &(sitetables[cb]), cb);

#if defined(DEBUG)
    QDP_info("Subset(%d): indexrep=%d start=%d end=%d",cb,indexrep,start,end);
#endif
  }
}
	  

//-----------------------------------------------------------------------------
//! Initializer for generic map constructor
void Map::make(const MapFunc& func)
{
//  QDP_info("Map::make");

  //--------------------------------------
  // Setup the communication index arrays
  soffsets.resize(Layout::sitesOnNode());
  srcnode.resize(Layout::sitesOnNode());
  dstnode.resize(Layout::sitesOnNode());

  const int my_node = Layout::nodeNumber();

  // Loop over the sites on this node
  for(int linear=0; linear < Layout::sitesOnNode(); ++linear)
  {
    // Get the true lattice coord of this linear site index
    multi1d<int> coord = Layout::siteCoords(my_node, linear);

    // Source neighbor for this destination site
    multi1d<int> fcoord = func(coord,+1);

    // Destination neighbor receiving data from this site
    // This functions as the inverse map
    multi1d<int> bcoord = func(coord,-1);

    int fnode = Layout::nodeNumber(fcoord);
    int bnode = Layout::nodeNumber(bcoord);

    // Source linear site and node
    soffsets[linear] = Layout::linearSiteIndex(fcoord);
    srcnode[linear]  = fnode;

    // Destination node
    dstnode[linear]  = bnode;
  }

#if 0
//  extern NmlWriter nml;

//  Write(nml,srcnode);
//  Write(nml,dstnode);

  for(int linear=0; linear < Layout::sitesOnNode(); ++linear)
  {
    QDP_info("soffsets(%d) = %d",linear,soffsets(linear));
    QDP_info("srcnode(%d) = %d",linear,srcnode(linear));
    QDP_info("dstnode(%d) = %d",linear,dstnode(linear));
  }
#endif

  // Return a list of the unique nodes in the list
  // NOTE: my_node may be included as a unique node, so one extra
  multi1d<int> srcenodes_tmp = uniquify_list(srcnode);
  multi1d<int> destnodes_tmp = uniquify_list(dstnode);

  // To simplify life, remove a possible occurance of my_node
  // This may mean that the list could be empty afterwards, so that
  // means mark offnodeP as false
  int cnt_srcenodes = 0;
  for(int i=0; i < srcenodes_tmp.size(); ++i)
    if (srcenodes_tmp[i] != my_node)
      ++cnt_srcenodes;

  int cnt_destnodes = 0;
  for(int i=0; i < destnodes_tmp.size(); ++i)
    if (destnodes_tmp[i] != my_node)
      ++cnt_destnodes;

#if 0
  // Debugging
  for(int i=0; i < srcenodes_tmp.size(); ++i)
    QDP_info("srcenodes_tmp(%d) = %d",i,srcenodes_tmp[i]);

  for(int i=0; i < destnodes_tmp.size(); ++i)
    QDP_info("destnodes_tmp(%d) = %d",i,destnodes_tmp[i]);

  QDP_info("cnt_srcenodes = %d", cnt_srcenodes);
  QDP_info("cnt_destnodes = %d", cnt_destnodes);
#endif


  // A sanity check - both counts must be either 0 or non-zero
  if (cnt_srcenodes > 0 && cnt_destnodes == 0)
    QDP_error_exit("Map: some bizarre error - no dest nodes but have srce nodes");

  if (cnt_srcenodes == 0 && cnt_destnodes > 0)
    QDP_error_exit("Map: some bizarre error - no srce nodes but have dest nodes");

  // If no srce/dest nodes, then we know no off-node communications
  offnodeP = (cnt_srcenodes > 0) ? true : false;
  
  //
  // The rest of the routine is devoted to supporting off-node communications
  // If there is not any communications, then return
  //
  if (! offnodeP)
  {
//  QDP_info("exiting Map::make");
    return;
  }

  // Finally setup the srce and dest nodes now without my_node
  srcenodes.resize(cnt_srcenodes);
  destnodes.resize(cnt_destnodes);
  
  for(int i=0, j=0; i < srcenodes_tmp.size(); ++i)
    if (srcenodes_tmp[i] != my_node)
      srcenodes[j++] = srcenodes_tmp[i];

  for(int i=0, j=0; i < destnodes_tmp.size(); ++i)
    if (destnodes_tmp[i] != my_node)
      destnodes[j++] = destnodes_tmp[i];

#if 0
  // Debugging
  for(int i=0; i < srcenodes.size(); ++i)
    QDP_info("srcenodes(%d) = %d",i,srcenodes(i));

  for(int i=0; i < destnodes.size(); ++i)
    QDP_info("destnodes(%d) = %d",i,destnodes(i));
#endif


//  Write(nml,srcenodes);
//  Write(nml,destnodes);

  // Run through the lists and find the number of each unique node
  srcenodes_num.resize(srcenodes.size());
  destnodes_num.resize(destnodes.size());

  srcenodes_num = 0;
  destnodes_num = 0;

  for(int linear=0; linear < Layout::sitesOnNode(); ++linear)
  {
    int this_node = srcnode[linear];
    if (this_node != my_node)
      for(int i=0; i < srcenodes_num.size(); ++i)
      {
	if (srcenodes[i] == this_node)
	{
	  srcenodes_num[i]++;
	  break;
	}
      }

    int that_node = dstnode[linear];
    if (that_node != my_node)
      for(int i=0; i < destnodes_num.size(); ++i)
      {
	if (destnodes[i] == that_node)
	{
	  destnodes_num[i]++;
	  break;
	}
      }
  }
  
//  Write(nml,srcenodes_num);
//  Write(nml,destnodes_num);


#if 0
  for(int i=0; i < destnodes.size(); ++i)
  {
    QDP_info("srcenodes(%d) = %d",i,srcenodes(i));
    QDP_info("destnodes(%d) = %d",i,destnodes(i));
  }

  for(int i=0; i < destnodes_num.size(); ++i)
  {
    QDP_info("srcenodes_num(%d) = %d",i,srcenodes_num(i));
    QDP_info("destnodes_num(%d) = %d",i,destnodes_num(i));
  }
#endif

//  QDP_info("exiting Map::make");
}


//------------------------------------------------------------------------
// Message passing convenience routines
//------------------------------------------------------------------------

namespace Internal
{
  //! Send to another node (wait)
  void 
  sendToWait(void *send_buf, int dest_node, int count)
  {
#if defined(DEBUG)
    QDP_info("starting a sendToWait, count=%d, destnode=%d", count,dest_node);
#endif

    QMP_msgmem_t request_msg = QMP_declare_msgmem(send_buf, count);
    QMP_msghandle_t request_mh = QMP_declare_send_to(request_msg, dest_node, 0);

    if (QMP_start(request_mh) != QMP_SUCCESS)
      QMP_error_exit("sendToWait failed\n");

    QMP_wait(request_mh);

    QMP_free_msghandle(request_mh);
    QMP_free_msgmem(request_msg);

#if defined(DEBUG)
    QDP_info("finished a sendToWait");
#endif
  }

  //! Receive from another node (wait)
  void 
  recvFromWait(void *recv_buf, int srce_node, int count)
  {
#if defined(DEBUG)
    QDP_info("starting a recvFromWait, count=%d, srcenode=%d", count, srce_node);
#endif

    QMP_msgmem_t request_msg = QMP_declare_msgmem(recv_buf, count);
    QMP_msghandle_t request_mh = QMP_declare_receive_from(request_msg, srce_node, 0);

    if (QMP_start(request_mh) != QMP_SUCCESS)
      QMP_error_exit("recvFromWait failed\n");

    QMP_wait(request_mh);

    QMP_free_msghandle(request_mh);
    QMP_free_msgmem(request_msg);

#if defined(DEBUG)
    QDP_info("finished a recvFromWait");
#endif
  }

};


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
