// $Id: parscalar_specific.cc,v 1.9 2003-01-17 05:45:43 edwards Exp $

/*! @file
 * @brief Parscalar specific routines
 * 
 * Routines for parscalar implementation
 */


#include "qdp.h"
#include "proto.h"
#include "QMP.h"

QDP_BEGIN_NAMESPACE(QDP);

//! Definition of shift function object
NearestNeighborMap  shift;


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

#if 1
  fprintf(stderr,"Set a subset: nsubset = %d\n",nsubset_indices);
#endif

  // This actually allocates the subsets
  sub.resize(nsubset_indices);

  // Create the space of the colorings of the lattice
  lat_color.resize(Layout::subgridVol());

  // Create the array holding the array of sitetable info
  sitetables.resize(nsubset_indices);

  // For a sanity check, set to some invalid value
  for(int site=0; site < Layout::subgridVol(); ++site)
    lat_color[site] = -1;

  // Loop over all sites determining their color
  for(int site=0; site < Layout::subgridVol(); ++site)
  {
    multi1d<int> coord = crtesn(site, Layout::subgridLattSize());

    for(int m=0; m<Nd; ++m)
      coord[m] += Layout::nodeCoord()[m]*Layout::subgridLattSize()[m];

    int node   = Layout::nodeNumber(coord);
    int linear = Layout::linearSiteIndex(coord);
    int icolor = func(coord);

    cerr << "site="<<site<<" coord="<<coord<<" node="<<node<<" linear="<<linear<<" col="<<icolor;
    cerr << endl;

    if (node != Layout::nodeNumber())
      QDP_error_exit("Set: found site with node outside current node!");

    lat_color[linear] = icolor;
  }

  // Loop over all sites and see if coloring properly set
  for(int site=0; site < Layout::subgridVol(); ++site)
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
    for(int linear=0; linear < Layout::subgridVol(); ++linear)
      if (lat_color[linear] == cb)
	++num_sitetable;

    // Now take the inverse of the lattice coloring to produce
    // the site list
    multi1d<int>& sitetable = sitetables[cb];
    sitetable.resize(num_sitetable);

    for(int linear=0, j=0; linear < Layout::subgridVol(); ++linear)
      if (lat_color[linear] == cb)
	sitetable[j++] = linear;


    sub[cb].make(start, end, indexrep, &(sitetables[cb]), cb);

#if 1
    fprintf(stderr,"Subset(%d): indexrep=%d start=%d end=%d\n",cb,indexrep,start,end);
#endif
  }
}
	  

//-----------------------------------------------------------------------------
//! Initializer for generic map constructor
void Map::make(const MapFunc& func)
{
  QDP_info("Map::make");

  //--------------------------------------
  // Setup the communication index arrays
  soffsets.resize(Layout::subgridVol());
  srcnode.resize(Layout::subgridVol());
  dstnode.resize(Layout::subgridVol());

  const int my_node = Layout::nodeNumber();

  // Loop over the sites on this node
  for(int linear=0; linear < Layout::subgridVol(); ++linear)
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

#if 1
//  extern NmlWriter nml;

//  Write(nml,srcnode);
//  Write(nml,dstnode);

  for(int linear=0; linear < Layout::subgridVol(); ++linear)
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

#if 1
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
    QDP_info("exiting Map::make");
    return;
  }

  // Finally setup the srce and dest nodes now without my_node
  srcenodes.resize(cnt_srcenodes);
  destnodes.resize(cnt_destnodes);
  
  for(int i=0, j=0; i < srcenodes_tmp.size(); ++i)
    if (srcenodes_tmp[i] != my_node)
      srcenodes[j++] = srcenodes_tmp[i];

  for(int i=0, j=0; i < destnodes.size(); ++i)
    if (destnodes_tmp[i] != my_node)
      destnodes[j++] = destnodes_tmp[i];

#if 1
  // Debugging
  for(int i=0; i < srcenodes.size(); ++i)
    fprintf(stderr,"srcenodes(%d) = %d\n",i,srcenodes(i));

  for(int i=0; i < destnodes.size(); ++i)
    fprintf(stderr,"destnodes(%d) = %d\n",i,destnodes(i));
#endif


//  Write(nml,srcenodes);
//  Write(nml,destnodes);

  // Run through the lists and find the number of each unique node
  srcenodes_num.resize(srcenodes.size());
  destnodes_num.resize(destnodes.size());

  srcenodes_num = 0;
  destnodes_num = 0;

  for(int linear=0; linear < Layout::subgridVol(); ++linear)
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


#if 1
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

  QDP_info("exiting Map::make");
}


//-----------------------------------------------------------------------------
//! Initializer for nearest neighbor shift
void NearestNeighborMap::make()
{
  //--------------------------------------
  // Setup the communication index arrays
  soffsets.resize(Nd, 2, Layout::subgridVol());

  /* Get the offsets needed for neighbour comm.
   * soffsets(direction,isign,position)
   *  where  isign    = +1 : plus direction
   *                  =  0 : negative direction
   * the offsets cotain the current site, i.e the neighbour for site i
   * is  soffsets(i,dir,mu) and NOT  i + soffset(..) 
   */
  const multi1d<int>& nrow = Layout::lattSize();
  const multi1d<int>& subgrid = Layout::subgridLattSize();
  const multi1d<int>& node_coord = Layout::nodeCoord();
  multi1d<int> node_offset(Nd);

  for(int m=0; m<Nd; ++m)
    node_offset[m] = node_coord[m]*subgrid[m];

  for(int site=0; site < Layout::vol(); ++site)
  {
    // Get the true grid of this site
    multi1d<int> coord = crtesn(site, nrow);

    // Site and node for this lattice site within the machine
    int ipos = Layout::linearSiteIndex(coord);
    int node = Layout::nodeNumber(coord);

    // If this is my node, then add it to my list
    if (Layout::nodeNumber() == node)
    {
//      <must get a new ipos within a node>

      for(int m=0; m<Nd; ++m)
      {
	multi1d<int> tmpcoord = coord;

	/* Neighbor in backward direction */
	tmpcoord[m] = (coord[m] - 1 + nrow[m]) % nrow[m];
	soffsets(m,0,ipos) = Layout::linearSiteIndex(tmpcoord);

	/* Neighbor in forward direction */
	tmpcoord[m] = (coord[m] + 1) % nrow[m];
	soffsets(m,1,ipos) = Layout::linearSiteIndex(tmpcoord);
      }
    }
  }

#if 0
  for(int m=0; m < Nd; ++m)
    for(int s=0; s < 2; ++s)
      for(int ipos=0; ipos < Layout::subgridVol(); ++ipos)
	fprintf(stderr,"soffsets(%d,%d,%d) = %d\n",ipos,s,m,soffsets(m,s,ipos));
#endif
}


//----------------------------------------------------------------------------
// ArrayMap

// This class is is used for binding the direction index of an ArrayMapFunc
// so as to construct a MapFunc
struct PackageArrayMapFunc : public MapFunc
{
  PackageArrayMapFunc(const ArrayMapFunc& mm, int ddir): pmap(mm), dir(ddir) {}

  virtual multi1d<int> operator() (const multi1d<int>& coord, int sign) const
    {
      return pmap(coord, sign, dir);
    }

private:
  const ArrayMapFunc& pmap;
  int dir;
}; 


//! Initializer for generic map constructor
void ArrayMap::make(const ArrayMapFunc& func)
{
  // We are allowed to declare a mapsa, but not allocate one.
  // There is an empty constructor for Map. Hence, the resize will
  // actually allocate the space.
  mapsa.resize(func.numArray());

  // Loop over each direction making the Map
  for(int dir=0; dir < func.numArray(); ++dir)
  {
    PackageArrayMapFunc  my_local_map(func,dir);

    mapsa[dir].make(my_local_map);
  }
}



//------------------------------------------------------------------------
// Message passing convenience routines
//------------------------------------------------------------------------

namespace Internal
{
  // Nearest neighbor communication channels
  static QMP_msgmem_t request_msg[Nd][2];
  static QMP_msghandle_t request_mh[Nd][2];
  static QMP_msghandle_t mh_both[Nd];

  //! Slow send-receive (blocking)
  void
  sendRecvWait(void *send_buf, void *recv_buf, 
	       int count, int isign, int dir)
  {
#ifdef DEBUG
    QDP_info("starting a sendRecvWait, count=%d",count);
#endif

    QMP_msgmem_t msg[2] = {QMP_declare_msgmem(send_buf, count),
			   QMP_declare_msgmem(recv_buf, count)};
    QMP_msghandle_t mh_a[2] = {QMP_declare_send_relative(msg[0], dir, isign, 0),
			       QMP_declare_receive_relative(msg[1], dir, -isign, 0)};
    QMP_msghandle_t mh = QMP_declare_multiple(mh_a, 2);

    QMP_start(mh);
    QMP_wait(mh);

    QMP_free_msghandle(mh_a[1]);
    QMP_free_msghandle(mh_a[0]);
    QMP_free_msghandle(mh);
    QMP_free_msgmem(msg[1]);
    QMP_free_msgmem(msg[0]);

#ifdef DEBUG
    QDP_info("finished a sendRecvWait");
#endif
  }


  //! Fast send-receive (non-blocking)
  void
  sendRecv(void *send_buf, void *recv_buf, 
	   int count, int isign0, int dir)
  {
#ifdef DEBUG
    QDP_info("starting a sendRecv, count=%d, isign=%d dir=%d",
	     count,isign,dir);
#endif

    int isign = (isign0 > 0) ? 1 : -1;

    request_msg[dir][0] = QMP_declare_msgmem(send_buf, count);
    request_msg[dir][1] = QMP_declare_msgmem(recv_buf, count);
    request_mh[dir][1] = QMP_declare_send_relative(request_msg[dir][0], dir, isign, 0);
    request_mh[dir][0] = QMP_declare_receive_relative(request_msg[dir][1], dir, -isign, 0);
    mh_both[dir] = QMP_declare_multiple(request_mh[dir], 2);

    if (QMP_start(mh_both[dir]) != QMP_SUCCESS)
      QMP_error_exit("QMP_create_physical_topology failed\n");

#ifdef DEBUG
    QDP_info("finished a sendRecv");
#endif
  }

  //! Wait on send-receive (now blocks)
  void
  wait(int dir)
  {
#ifdef DEBUG
    QDP_info("starting a wait");
#endif
    
    QMP_wait(mh_both[dir]);

    QMP_free_msghandle(request_mh[dir][1]);
    QMP_free_msghandle(request_mh[dir][0]);
    QMP_free_msghandle(mh_both[dir]);
    QMP_free_msgmem(request_msg[dir][1]);
    QMP_free_msgmem(request_msg[dir][0]);

#ifdef DEBUG
    QDP_info("finished a wait");
#endif
  }


  //! Send to another node (wait)
  void 
  sendToWait(void *send_buf, int dest_node, int count)
  {
#ifdef DEBUG
    QDP_info("starting a sendToWait, count=%d, destnode=%d", count,dest_node);
#endif

    QMP_msgmem_t request_msg = QMP_declare_msgmem(send_buf, count);
    QMP_msghandle_t request_mh = QMP_declare_send_to(request_msg, dest_node, 0);

    if (QMP_start(request_mh) != QMP_SUCCESS)
      QMP_error_exit("sendToWait failed\n");

    QMP_wait(request_mh);

    QMP_free_msghandle(request_mh);
    QMP_free_msgmem(request_msg);

#ifdef DEBUG
    QDP_info("finished a sendToWait");
#endif
  }

  //! Receive from another node (wait)
  void 
  recvFromWait(void *recv_buf, int srce_node, int count)
  {
#ifdef DEBUG
    QDP_info("starting a recvFromWait, count=%d, srcenode=%d", count, srce_node);
#endif

    QMP_msgmem_t request_msg = QMP_declare_msgmem(recv_buf, count);
    QMP_msghandle_t request_mh = QMP_declare_receive_from(request_msg, srce_node, 0);

    if (QMP_start(request_mh) != QMP_SUCCESS)
      QMP_error_exit("recvFromWait failed\n");

    QMP_wait(request_mh);

    QMP_free_msghandle(request_mh);
    QMP_free_msgmem(request_msg);

#ifdef DEBUG
    QDP_info("finished a recvFromWait");
#endif
  }

};

QDP_END_NAMESPACE();
