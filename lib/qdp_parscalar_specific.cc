// $Id: qdp_parscalar_specific.cc,v 1.7 2003-07-26 04:01:54 edwards Exp $

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
//! Initializer for generic map constructor
void Map::make(const MapFunc& func)
{
#if QDP_DEBUG >= 3
  QDP_info("Map::make");
#endif

  //--------------------------------------
  // Setup the communication index arrays
  goffsets.resize(Layout::sitesOnNode());
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
    goffsets[linear] = Layout::linearSiteIndex(fcoord);
    srcnode[linear]  = fnode;

    // Destination node
    dstnode[linear]  = bnode;

#if QDP_DEBUG >= 3
    QDP_info("linear=%d  coord=%d %d %d %d   fcoord=%d %d %d %d    goffsets=%d", 
	     linear, 
	     coord[0], coord[1], coord[2], coord[3],
	     fcoord[0], fcoord[1], fcoord[2], fcoord[3],
	     goffsets[linear]);
#endif
  }

#if QDP_DEBUG >= 3
 {
   ostringstream foon;
   foon << "map." << Layout::nodeNumber();
   NmlWriter nml(foon.str());

   Write(nml,srcnode);
   Write(nml,dstnode);

   for(int linear=0; linear < Layout::sitesOnNode(); ++linear)
   {
     QDP_info("goffsets(%d) = %d",linear,goffsets(linear));
     QDP_info("srcnode(%d) = %d",linear,srcnode(linear));
     QDP_info("dstnode(%d) = %d",linear,dstnode(linear));
   }
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

#if QDP_DEBUG >= 3
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
#if QDP_DEBUG >= 3
    QDP_info("exiting Map::make");
#endif
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

#if QDP_DEBUG >= 3
  // Debugging
  for(int i=0; i < srcenodes.size(); ++i)
    QDP_info("srcenodes(%d) = %d",i,srcenodes(i));

  for(int i=0; i < destnodes.size(); ++i)
    QDP_info("destnodes(%d) = %d",i,destnodes(i));
#endif


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
  

#if QDP_DEBUG >= 3
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

  // Implementation limitation in the Map::operator(). Only support
  // a node sending data all to one node or no sending at all (offNodeP == false).
  if (srcenodes.size() != 1)
    QMP_error_exit("Map: for now only allow 1 destination node");
      
  if (destnodes.size() != 1)
    QMP_error_exit("Map: for now only allow receives from 1 node");


  // Now make a small scatter array for the dest_buf so that when data
  // is sent, it is put in an order the gather can pick it up
  // If we allow multiple dest nodes, then soffsets here needs to be
  // an array of arrays
  soffsets.resize(destnodes_num[0]);
  
  // Loop through sites on my *destination* node - here I assume all nodes have
  // the same number of sites. Mimic the gather pattern needed on that node and
  // set my scatter array to scatter into the correct site order
  for(int i=0, si=0; i < Layout::sitesOnNode(); ++i) 
  {
    // Get the true lattice coord of this linear site index
    multi1d<int> coord = Layout::siteCoords(destnodes[0], i);
    multi1d<int> fcoord = func(coord,+1);
    int fnode = Layout::nodeNumber(fcoord);
    int fline = Layout::linearSiteIndex(fcoord);

    if (fnode == my_node)
      soffsets[si++] = fline;
  }

#if QDP_DEBUG >= 3
  // Debugging
  for(int i=0; i < soffsets.size(); ++i)
    QDP_info("soffsets(%d) = %d",i,soffsets(i));
#endif


#if QDP_DEBUG >= 3
  QDP_info("exiting Map::make");
#endif
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
#if QDP_DEBUG >= 2
    QDP_info("starting a sendToWait, count=%d, destnode=%d", count,dest_node);
#endif

    QMP_msgmem_t request_msg = QMP_declare_msgmem(send_buf, count);
    QMP_msghandle_t request_mh = QMP_declare_send_to(request_msg, dest_node, 0);

    if (QMP_start(request_mh) != QMP_SUCCESS)
      QMP_error_exit("sendToWait failed\n");

    QMP_wait(request_mh);

    QMP_free_msghandle(request_mh);
    QMP_free_msgmem(request_msg);

#if QDP_DEBUG >= 2
    QDP_info("finished a sendToWait");
#endif
  }

  //! Receive from another node (wait)
  void 
  recvFromWait(void *recv_buf, int srce_node, int count)
  {
#if QDP_DEBUG >= 2
    QDP_info("starting a recvFromWait, count=%d, srcenode=%d", count, srce_node);
#endif

    QMP_msgmem_t request_msg = QMP_declare_msgmem(recv_buf, count);
    QMP_msghandle_t request_mh = QMP_declare_receive_from(request_msg, srce_node, 0);

    if (QMP_start(request_mh) != QMP_SUCCESS)
      QMP_error_exit("recvFromWait failed\n");

    QMP_wait(request_mh);

    QMP_free_msghandle(request_mh);
    QMP_free_msgmem(request_msg);

#if QDP_DEBUG >= 2
    QDP_info("finished a recvFromWait");
#endif
  }


  // A really stupid way to do broadcast
  void 
  stupidBroadcast(void* dest, unsigned int nbytes)
  {
    // Send to each node
    for(int node=1; node < Layout::numNodes(); ++node)
    {
      if (Layout::nodeNumber() == node)
	Internal::recvFromWait(dest, 0, nbytes);

      if (Layout::primaryNode())
	Internal::sendToWait(dest, node, nbytes);
    }
  }

};


//-----------------------------------------------------------------------------
// Write a lattice quantity
void writeOLattice(BinaryWriter& bin, 
		   const char* output, size_t size, size_t nmemb)
{
  size_t tot_size = size*nmemb;
  char *recv_buf = new char[tot_size];

  // Find the location of each site and send to primary node
  for(int site=0; site < Layout::vol(); ++site)
  {
    multi1d<int> coord = crtesn(site, Layout::lattSize());

    int node   = Layout::nodeNumber(coord);
    int linear = Layout::linearSiteIndex(coord);

    // Copy to buffer: be really careful since max(linear) could vary among nodes
    if (Layout::nodeNumber() == node)
      memcpy(recv_buf, output+linear*tot_size, tot_size);

    // Send result to primary node. Avoid sending prim-node sending to itself
    if (node != 0)
    {
      if (Layout::primaryNode())
	Internal::recvFromWait((void *)recv_buf, node, tot_size);

      if (Layout::nodeNumber() == node)
	Internal::sendToWait((void *)recv_buf, 0, tot_size);
    }

    if (Layout::primaryNode())
      bin.writeArray(recv_buf, size, nmemb);
  }

  delete[] recv_buf;
}


//! Read a lattice quantity
/*! This code assumes no inner grid */
void readOLattice(BinaryReader& bin, 
		  char* input, size_t size, size_t nmemb)
{
  size_t tot_size = size*nmemb;
  char *recv_buf = new char[tot_size];

  // Find the location of each site and send to primary node
  for(int site=0; site < Layout::vol(); ++site)
  {
    multi1d<int> coord = crtesn(site, Layout::lattSize());

    int node   = Layout::nodeNumber(coord);
    int linear = Layout::linearSiteIndex(coord);

    // Only on primary node read the data
    bin.readArrayPrimaryNode(recv_buf, size, nmemb);

    // Send result to destination node. Avoid sending prim-node sending to itself
    if (node != 0)
    {
      if (Layout::primaryNode())
	Internal::sendToWait((void *)recv_buf, node, tot_size);

      if (Layout::nodeNumber() == node)
	Internal::recvFromWait((void *)recv_buf, 0, tot_size);
    }

    if (Layout::nodeNumber() == node)
      memcpy(input+linear*tot_size, recv_buf, tot_size);
  }

  delete[] recv_buf;
}


QDP_END_NAMESPACE();
