// $Id: qdp_parscalar_specific.cc,v 1.36 2008-05-13 20:00:18 bjoo Exp $

/*! @file
 * @brief Parscalar specific routines
 * 
 * Routines for parscalar implementation
 */


#include "qdp.h"
#include "qdp_util.h"
#include "qmp.h"


namespace QDP {

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
  const int nodeSites = Layout::sitesOnNode();

  //--------------------------------------
  // Setup the communication index arrays
  goffsets.resize(nodeSites);
  srcnode.resize(nodeSites);
  dstnode.resize(nodeSites);

  const int my_node = Layout::nodeNumber();

  // Loop over the sites on this node
  for(int linear=0; linear < nodeSites; ++linear)
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
    QDP_info("linear=%d  coord=%d %d %d %d   fcoord=%d %d %d %d   bcoord=%d %d %d %d   goffsets=%d", 
	     linear, 
	     coord[0], coord[1], coord[2], coord[3],
	     fcoord[0], fcoord[1], fcoord[2], fcoord[3],
	     bcoord[0], bcoord[1], bcoord[2], bcoord[3],
	     goffsets[linear]);
#endif
  }

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
    QDP_info("no off-node communications: exiting Map::make");
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

  for(int linear=0; linear < nodeSites; ++linear)
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
    QDP_error_exit("Map: for now only allow 1 destination node");
      
  if (destnodes.size() != 1)
    QDP_error_exit("Map: for now only allow receives from 1 node");


  // Now make a small scatter array for the dest_buf so that when data
  // is sent, it is put in an order the gather can pick it up
  // If we allow multiple dest nodes, then soffsets here needs to be
  // an array of arrays
  soffsets.resize(destnodes_num[0]);
  
  // Loop through sites on my *destination* node - here I assume all nodes have
  // the same number of sites. Mimic the gather pattern needed on that node and
  // set my scatter array to scatter into the correct site order
  for(int i=0, si=0; i < nodeSites; ++i) 
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
  //! Broadcast a string from primary node to all other nodes
  void broadcast_str(std::string& result)
  {
    char *dd_tmp;
    int lleng;

    // Only primary node can grab string
    if (Layout::primaryNode()) 
    {
      lleng = result.length() + 1;
    }

    // First must broadcast size of string
    Internal::broadcast(lleng);

    // Now every node can alloc space for string
    dd_tmp = new(nothrow) char[lleng];
    if( dd_tmp == 0x0 ) { 
      QDP_error_exit("Unable to allocate dd_tmp\n");
    }

    if (Layout::primaryNode())
      memcpy(dd_tmp, result.c_str(), lleng);
  
    // Now broadcast char array out to all nodes
    Internal::broadcast((void *)dd_tmp, lleng);

    // All nodes can now grab char array and make a string
    result = dd_tmp;

    // Clean-up and boogie
    delete[] dd_tmp;
  }

  //! Is this a grid architecture
  bool gridArch()
  { 
    return (QMP_get_msg_passing_type() == QMP_GRID) ? true : false;
  }

  //! Send a clear-to-send
  void clearToSend(void *buffer, int count, int node)
  { 
    // On non-grid machines, use a clear-to-send like protocol
    if (! Internal::gridArch())
    {
      if(Layout::nodeNumber() == 0 && node != 0)
	Internal::sendToWait(buffer, node, count);
      if(Layout::nodeNumber() == node && node != 0)
	Internal::recvFromWait(buffer, 0, count);
    }
  }

  //! Route to another node (blocking)
  void route(void *buffer, int srce_node, int dest_node, int count)
  { 
#if QDP_DEBUG >= 2
    QDP_info("starting a route, count=%d, srcenode=%d destnode=%d", 
	     count,srce_node,dest_node);
#endif

//    QMP_route(buffer, count, srce_node, dest_node);
    DML_route_bytes((char*)buffer, count, srce_node, dest_node);

#if QDP_DEBUG >= 2
    QDP_info("finished a route");
#endif
  }

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
      QDP_error_exit("sendToWait failed\n");

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
      QDP_error_exit("recvFromWait failed\n");

    QMP_wait(request_mh);

    QMP_free_msghandle(request_mh);
    QMP_free_msgmem(request_msg);

#if QDP_DEBUG >= 2
    QDP_info("finished a recvFromWait");
#endif
  }

};


//-----------------------------------------------------------------------------
// Write a lattice quantity
void writeOLattice(BinaryWriter& bin, 
		   const char* output, size_t size, size_t nmemb)
{
  const int xinc = Layout::subgridLattSize()[0];

  size_t sizemem = size*nmemb;
  size_t tot_size = sizemem*xinc;
  char *recv_buf = new(nothrow) char[tot_size];
  if( recv_buf == 0x0 ) { 
    QDP_error_exit("Unable to allocate recv_buf\n");
  }

  // Find the location of each site and send to primary node
  int old_node = 0;

  for(int site=0; site < Layout::vol(); site += xinc)
  {
    // first site in each segment uniquely identifies the node
    int node = Layout::nodeNumber(crtesn(site, Layout::lattSize()));

    // Send nodes must wait for a ready signal from the master node
    // to prevent message pileups on the master node
    if (node != old_node)
    {
      // On non-grid machines, use a clear-to-send like protocol
      Internal::clearToSend(recv_buf,sizeof(int),node);
      old_node = node;
    }
    
    // Copy to buffer: be really careful since max(linear) could vary among nodes
    if (Layout::nodeNumber() == node)
    {
      for(int i=0; i < xinc; ++i)
      {
	int linear = Layout::linearSiteIndex(crtesn(site+i, Layout::lattSize()));
	memcpy(recv_buf+i*sizemem, output+linear*sizemem, sizemem);
      }
    }

    // Send result to primary node. Avoid sending prim-node sending to itself
    if (node != 0)
    {
#if 1
      // All nodes participate
      Internal::route((void *)recv_buf, node, 0, tot_size);
#else
      if (Layout::primaryNode())
	Internal::recvFromWait((void *)recv_buf, node, tot_size);

      if (Layout::nodeNumber() == node)
	Internal::sendToWait((void *)recv_buf, 0, tot_size);
#endif
    }

    bin.writeArrayPrimaryNode(recv_buf, size, nmemb*xinc);
  }

  delete[] recv_buf;

  // Keep the checksum in sync on all nodes. This only really
  // is needed if nodes do detailed checks on the checksums
  bin.syncChecksum();
}


// Write a single site of a lattice quantity
void writeOLattice(BinaryWriter& bin, 
		   const char* output, size_t size, size_t nmemb,
		   const multi1d<int>& coord)
{
  size_t tot_size = size*nmemb;
  char *recv_buf = new(nothrow) char[tot_size];
  if( recv_buf == 0x0 ) { 
    QDP_error_exit("Unable to allocate recvbuf\n");
  }


  // Send site to primary node
  int node   = Layout::nodeNumber(coord);
  int linear = Layout::linearSiteIndex(coord);

  // Send nodes must wait for a ready signal from the master node
  // to prevent message pileups on the master node
  Internal::clearToSend(recv_buf,sizeof(int),node);
  
  // Copy to buffer: be really careful since max(linear) could vary among nodes
  if (Layout::nodeNumber() == node)
    memcpy(recv_buf, output+linear*tot_size, tot_size);
  
  // Send result to primary node. Avoid sending prim-node sending to itself
  if (node != 0)
  {
#if 1
    // All nodes participate
    Internal::route((void *)recv_buf, node, 0, tot_size);
#else
    if (Layout::primaryNode())
      Internal::recvFromWait((void *)recv_buf, node, tot_size);

    if (Layout::nodeNumber() == node)
      Internal::sendToWait((void *)recv_buf, 0, tot_size);
#endif
  }

  bin.writeArray(recv_buf, size, nmemb);

  delete[] recv_buf;
}


//! Read a lattice quantity
/*! This code assumes no inner grid */
void readOLattice(BinaryReader& bin, 
		  char* input, size_t size, size_t nmemb)
{
  const int xinc = Layout::subgridLattSize()[0];

  size_t sizemem = size*nmemb;
  size_t tot_size = sizemem*xinc;
  char *recv_buf = new(nothrow) char[tot_size];
  if( recv_buf == 0x0 ) { 
    QDP_error_exit("Unable to allocate recvbuf\n");
  }

  // Find the location of each site and send to primary node
  for(int site=0; site < Layout::vol(); site += xinc)
  {
    // first site in each segment uniquely identifies the node
    int node = Layout::nodeNumber(crtesn(site, Layout::lattSize()));

    // Only on primary node read the data
    bin.readArrayPrimaryNode(recv_buf, size, nmemb*xinc);

    // Send result to destination node. Avoid sending prim-node sending to itself
    if (node != 0)
    {
#if 1
      // All nodes participate
      Internal::route((void *)recv_buf, 0, node, tot_size);
#else
      if (Layout::primaryNode())
	Internal::sendToWait((void *)recv_buf, node, tot_size);

      if (Layout::nodeNumber() == node)
	Internal::recvFromWait((void *)recv_buf, 0, tot_size);
#endif
    }

    if (Layout::nodeNumber() == node)
    {
      for(int i=0; i < xinc; ++i)
      {
	int linear = Layout::linearSiteIndex(crtesn(site+i, Layout::lattSize()));

	memcpy(input+linear*sizemem, recv_buf+i*sizemem, sizemem);
      }
    }
  }

  // Keep the checksum in sync on all nodes. This only really
  // is needed if nodes do detailed checks on the checksums
  bin.syncChecksum();

  delete[] recv_buf;
}

//! Read a single site worth of a lattice quantity
/*! This code assumes no inner grid */
void readOLattice(BinaryReader& bin, 
		  char* input, size_t size, size_t nmemb,
		  const multi1d<int>& coord)
{
  size_t tot_size = size*nmemb;
  char *recv_buf = new(nothrow) char[tot_size];
  if( recv_buf == 0x0 ) {
    QDP_error_exit("Unable to allocate recv_buf\n");
  }


  // Find the location of each site and send to primary node
  int node   = Layout::nodeNumber(coord);
  int linear = Layout::linearSiteIndex(coord);

  // Only on primary node read the data
  bin.readArrayPrimaryNode(recv_buf, size, nmemb);

  // Send result to destination node. Avoid sending prim-node sending to itself
  if (node != 0)
  {
#if 1
    // All nodes participate
    Internal::route((void *)recv_buf, 0, node, tot_size);
#else
    if (Layout::primaryNode())
      Internal::sendToWait((void *)recv_buf, node, tot_size);

    if (Layout::nodeNumber() == node)
      Internal::recvFromWait((void *)recv_buf, 0, tot_size);
#endif
  }

  if (Layout::nodeNumber() == node)
    memcpy(input+linear*tot_size, recv_buf, tot_size);

  // Keep the checksum in sync on all nodes. This only really
  // is needed if nodes do detailed checks on the checksums
  bin.syncChecksum();

  delete[] recv_buf;
}



//-----------------------------------------------------------------------
// Compute simple NERSC-like checksum of a gauge field
/*
 * \ingroup io
 *
 * \param u          gauge configuration ( Read )
 *
 * \return checksum
 */    

n_uint32_t computeChecksum(const multi1d<LatticeColorMatrix>& u,
			   int mat_size)
{
  size_t size = sizeof(REAL32);
  size_t su3_size = size*mat_size;
  n_uint32_t checksum = 0;   // checksum

  multi1d<multi1d<ColorMatrix> > sa(Nd);   // extract gauge fields
  const int nodeSites = Layout::sitesOnNode();

  for(int dd=0; dd<Nd; dd++)        /* dir */
  {
    sa[dd].resize(nodeSites);
    QDP_extract(sa[dd], u[dd], all);
  }

  char  *chk_buf = new(nothrow) char[su3_size];
  if( chk_buf == 0x0 ) { 
    QDP_error_exit("Unable to allocate chk_buf\n");
  }

  for(int linear=0; linear < nodeSites; ++linear)
  {
    for(int dd=0; dd<Nd; dd++)        /* dir */
    {
      switch (mat_size)
      {
      case 12:
      {
	REAL32 su3[2][3][2];

	for(int kk=0; kk<Nc; kk++)      /* color */
	  for(int ii=0; ii<2; ii++)    /* color */
	  {
	    Complex sitecomp = peekColor(sa[dd][linear],ii,kk);
	    su3[ii][kk][0] = toFloat(Real(real(sitecomp)));
	    su3[ii][kk][1] = toFloat(Real(imag(sitecomp)));
	  }

	memcpy(chk_buf, &(su3[0][0][0]), su3_size);
      }
      break;

      case 18:
      {
	REAL32 su3[3][3][2];

	for(int kk=0; kk<Nc; kk++)      /* color */
	  for(int ii=0; ii<Nc; ii++)    /* color */
	  {
	    Complex sitecomp = peekColor(sa[dd][linear],ii,kk);
	    su3[ii][kk][0] = toFloat(Real(real(sitecomp)));
	    su3[ii][kk][1] = toFloat(Real(imag(sitecomp)));
	  }

	memcpy(chk_buf, &(su3[0][0][0]), su3_size);
      }
      break;

      default:
	QDPIO::cerr << __func__ << ": unexpected size" << endl;
	QDP_abort(1);
      }

      // Compute checksum
      n_uint32_t* chk_ptr = (n_uint32_t*)chk_buf;
      for(unsigned int i=0; i < mat_size*size/sizeof(n_uint32_t); ++i)
	checksum += chk_ptr[i];
    }
  }

  delete[] chk_buf;

  // Get all nodes to contribute
  Internal::globalSumArray((unsigned int*)&checksum, 1);   // g++ requires me to narrow the type to unsigned int

  return checksum;
}


//-----------------------------------------------------------------------
// Read a QCD archive file
// Read a QCD (NERSC) Archive format gauge field
/*
 * \ingroup io
 *
 * \param cfg_in     binary writer object ( Modify )
 * \param u          gauge configuration ( Modify )
 */    

void readArchiv(BinaryReader& cfg_in, multi1d<LatticeColorMatrix>& u, 
		n_uint32_t& checksum, int mat_size, int float_size)
{
  size_t size = float_size;
  size_t su3_size = size*mat_size;
  size_t tot_size = su3_size*Nd;
  const int nodeSites = Layout::sitesOnNode();

  char  *input = new(nothrow) char[tot_size*nodeSites];  // keep another copy in input buffers
  if( input == 0x0 ) { 
    QDP_error_exit("Unable to allocate input\n");
  }

  char  *recv_buf = new(nothrow) char[tot_size];
  if( recv_buf == 0x0 ) { 
    QDP_error_exit("Unable to allocate recv_buf\n");
  }

  checksum = 0;

  // Find the location of each site and send to primary node
  for(int site=0; site < Layout::vol(); ++site)
  {
    multi1d<int> coord = crtesn(site, Layout::lattSize());

    int node   = Layout::nodeNumber(coord);
    int linear = Layout::linearSiteIndex(coord);

    // Only on primary node read the data
    cfg_in.readArrayPrimaryNode(recv_buf, size, mat_size*Nd);

    if (Layout::primaryNode()) 
    {
      // Compute checksum
      n_uint32_t* chk_ptr = (n_uint32_t*)recv_buf;
      for(unsigned int i=0; i < mat_size*Nd*size/sizeof(n_uint32_t); ++i)
	checksum += chk_ptr[i];
    }

    // Send result to destination node. Avoid sending prim-node sending to itself
    if (node != 0)
    {
#if 1
      // All nodes participate
      Internal::route((void *)recv_buf, 0, node, tot_size);
#else
      if (Layout::primaryNode())
	Internal::sendToWait((void *)recv_buf, node, tot_size);

      if (Layout::nodeNumber() == node)
	Internal::recvFromWait((void *)recv_buf, 0, tot_size);
#endif
    }

    if (Layout::nodeNumber() == node)
      memcpy(input+linear*tot_size, recv_buf, tot_size);
  }

  delete[] recv_buf;

  Internal::broadcast(checksum);

  // Reconstruct the gauge field
  ColorMatrix  sitefield;
  REAL su3[3][3][2];

  for(int linear=0; linear < nodeSites; ++linear)
  {
    for(int dd=0; dd<Nd; dd++)        /* dir */
    {
      // Transfer the data from input into SU3
      if (float_size == 4) 
      {
	REAL* su3_p = (REAL *)su3;
	REAL32* input_p = (REAL32 *)( input+su3_size*(dd+Nd*linear) );
	for(int cp_index=0; cp_index < mat_size; cp_index++) {
	  su3_p[cp_index] = (REAL)(input_p[cp_index]);
	}
      }
      else if (float_size == 8) 
      {
	// IEEE64BIT case
	REAL *su3_p = (REAL *)su3;
	REAL64 *input_p = (REAL64 *)( input+su3_size*(dd+Nd*linear) );
	for(int cp_index=0; cp_index < mat_size; cp_index++) { 
	  su3_p[cp_index] = (REAL)input_p[cp_index];
	}
      }
      else { 
	QDPIO::cerr << __func__ << ": Unknown mat size" << endl;
	QDP_abort(1);
      }

      /* Reconstruct the third column  if necessary */
      if (mat_size == 12) 
      {
	su3[2][0][0] = su3[0][1][0]*su3[1][2][0] - su3[0][1][1]*su3[1][2][1]
    	             - su3[0][2][0]*su3[1][1][0] + su3[0][2][1]*su3[1][1][1];
	su3[2][0][1] = su3[0][2][0]*su3[1][1][1] + su3[0][2][1]*su3[1][1][0]
	             - su3[0][1][0]*su3[1][2][1] - su3[0][1][1]*su3[1][2][0];

	su3[2][1][0] = su3[0][2][0]*su3[1][0][0] - su3[0][2][1]*su3[1][0][1]
	             - su3[0][0][0]*su3[1][2][0] + su3[0][0][1]*su3[1][2][1];
	su3[2][1][1] = su3[0][0][0]*su3[1][2][1] + su3[0][0][1]*su3[1][2][0]
	             - su3[0][2][0]*su3[1][0][1] - su3[0][2][1]*su3[1][0][0];
          
	su3[2][2][0] = su3[0][0][0]*su3[1][1][0] - su3[0][0][1]*su3[1][1][1]
	             - su3[0][1][0]*su3[1][0][0] + su3[0][1][1]*su3[1][0][1];
	su3[2][2][1] = su3[0][1][0]*su3[1][0][1] + su3[0][1][1]*su3[1][0][0]
	             - su3[0][0][0]*su3[1][1][1] - su3[0][0][1]*su3[1][1][0];
      }

      /* Copy into the big array */
      for(int kk=0; kk<Nc; kk++)      /* color */
      {
	for(int ii=0; ii<Nc; ii++)    /* color */
	{
	  Complex sitecomp = cmplx(Real(su3[ii][kk][0]), Real(su3[ii][kk][1]));
	  pokeColor(sitefield,sitecomp,ii,kk);
	}
      }
      
      u[dd].elem(linear) = sitefield.elem();
    }
  }
  
  delete[] input;
}


//-----------------------------------------------------------------------
// Write a QCD archive file
// Write a QCD (NERSC) Archive format gauge field
/*
 * \ingroup io
 *
 * \param cfg_out    binary writer object ( Modify )
 * \param u          gauge configuration ( Read )
 */    
void writeArchiv(BinaryWriter& cfg_out, const multi1d<LatticeColorMatrix>& u,
		 int mat_size)
{
  size_t size = sizeof(REAL32);
  size_t su3_size = size*mat_size;
  size_t tot_size = su3_size*Nd;
  char *recv_buf = new(nothrow) char[tot_size];
  if( recv_buf == 0x0 ) { 
    QDP_error_exit("Unable to allocate recv_buf\n");
  }

  const int nodeSites = Layout::sitesOnNode();

  multi1d<multi1d<ColorMatrix> > sa(Nd);   // extract gauge fields

  for(int dd=0; dd<Nd; dd++)        /* dir */
  {
    sa[dd].resize(nodeSites);
    QDP_extract(sa[dd], u[dd], all);
  }

  // Find the location of each site and send to primary node
  for(int site=0; site < Layout::vol(); ++site)
  {
    multi1d<int> coord = crtesn(site, Layout::lattSize());

    int node   = Layout::nodeNumber(coord);
    int linear = Layout::linearSiteIndex(coord);

    // Copy to buffer: be really careful since max(linear) could vary among nodes
    if (Layout::nodeNumber() == node)
    {
      char *recv_buf_tmp = recv_buf;

      for(int dd=0; dd<Nd; dd++)        /* dir */
      {
	if ( mat_size == 12 ) 
	{
	  REAL32 su3[2][3][2];

	  for(int kk=0; kk<Nc; kk++)      /* color */
	    for(int ii=0; ii<2; ii++)    /* color */
	    {
	      Complex sitecomp = peekColor(sa[dd][linear],ii,kk);
	      su3[ii][kk][0] = toFloat(Real(real(sitecomp)));
	      su3[ii][kk][1] = toFloat(Real(imag(sitecomp)));
	    }

	  memcpy(recv_buf_tmp, &(su3[0][0][0]), su3_size);
	}
	else
	{
	  REAL32 su3[3][3][2];

	  for(int kk=0; kk<Nc; kk++)      /* color */
	    for(int ii=0; ii<Nc; ii++)    /* color */
	    {
	      Complex sitecomp = peekColor(sa[dd][linear],ii,kk);
	      su3[ii][kk][0] = toFloat(Real(real(sitecomp)));
	      su3[ii][kk][1] = toFloat(Real(imag(sitecomp)));
	    }

	  memcpy(recv_buf_tmp, &(su3[0][0][0]), su3_size);
	}

	recv_buf_tmp += su3_size;
      }
    }

    // Send result to primary node. Avoid sending prim-node sending to itself
    if (node != 0)
    {
#if 1
      // All nodes participate
      Internal::route((void *)recv_buf, node, 0, tot_size);
#else
      if (Layout::primaryNode())
	Internal::recvFromWait((void *)recv_buf, node, tot_size);

      if (Layout::nodeNumber() == node)
	Internal::sendToWait((void *)recv_buf, 0, tot_size);
#endif
    }

    cfg_out.writeArrayPrimaryNode(recv_buf, size, mat_size*Nd);
  }

  // Keep the checksum in sync on all nodes. This only really
  // is needed if nodes do detailed checks on the checksums
  cfg_out.syncChecksum();

  delete[] recv_buf;

  if (cfg_out.fail())
  {
    QDPIO::cerr << __func__ << ": error writing configuration" << endl;
    QDP_abort(1);
  }
}

//-----------------------------------------------------------------------------
//Write a lattice quantity using parallel I/O. This routine is a
//temporary solution for writing parallel I/O on QCDOC. fwrite is
//called directly and hence the chroma BinaryWriter mechanism
//for I/O is bypassed. This routine takes as its first argument 
//the filename of the file to be written (cf writeOLattice which 
// takes the BinaryWriter object). The last argument is the ratio of nodes to 
//  concurrent writers. The routine returns a checksum of the data.
QDPUtil::n_uint32_t writeOLattice_parallel(const std::string& p, 
		   const char* output, size_t size, size_t nmemb,
					   int rw_ratio)
{
  const int xinc = Layout::subgridLattSize()[0];

  size_t sizemem = size*nmemb;
  size_t tot_size = sizemem*xinc;


  //First, need to compute a checksum since we are bypassing 
  //the BinaryWriter mechanism. This involves routing all data to 
  //master node who will do this computation. No writing is done at this stage

  QDPUtil::n_uint32_t  my_checksum = 0;
  char *recv_buf = new(nothrow) char[tot_size];

  if( recv_buf == 0x0 ) { 
    QDP_error_exit("Unable to allocate recv_buf\n");
  }

  // Find the location of each site and send to primary node
  int old_node = 0;

  for(int site=0; site < Layout::vol(); site += xinc)
  {
    // first site in each segment uniquely identifies the node
    int node = Layout::nodeNumber(crtesn(site, Layout::lattSize()));

    // Send nodes must wait for a ready signal from the master node
    // to prevent message pileups on the master node
    if (node != old_node)
    {
      // On non-grid machines, use a clear-to-send like protocol
      Internal::clearToSend(recv_buf,sizeof(int),node);
      old_node = node;
    }
    
    // Copy to buffer: be really careful since max(linear) could vary among nodes
    if (Layout::nodeNumber() == node)
    {
      for(int i=0; i < xinc; ++i)
      {
	int linear = Layout::linearSiteIndex(crtesn(site+i, Layout::lattSize()));
	memcpy(recv_buf+i*sizemem, output+linear*sizemem, sizemem);
      }
    }

    // Send result to primary node. Avoid sending prim-node sending to itself
    if (node != 0)
    {
#if 1
      // All nodes participate
      Internal::route((void *)recv_buf, node, 0, tot_size);
#else
      if (Layout::primaryNode())
	Internal::recvFromWait((void *)recv_buf, node, tot_size);

      if (Layout::nodeNumber() == node)
	Internal::sendToWait((void *)recv_buf, 0, tot_size);
#endif
    }

    if (Layout::primaryNode())

      //master node computes checksum
      my_checksum = QDPUtil::crc32(my_checksum, recv_buf, size*nmemb*xinc);
  }

  delete[] recv_buf;
  // End of checksum section


  //Now write data to disk in parallel


  //File for direct writing of OLattice
  FILE *fp;

  // first primaryNode creates an empty output file by opening and closing
  if (Layout::primaryNode())
    {
      if((fp=fopen(p.c_str(), "w")) == NULL) {
   	printf("Cannot open file %s.\n",p.c_str() );
  	exit(1);
     }
    fclose(fp);
  }

  // synchronise nodes
  //QMP_barrier();
  //dummy integer for global sum barrier
  int dummy_sum_int = 1;
  QMP_sum_int(&dummy_sum_int);

  // All nodes open output file for updating
  if((fp=fopen(p.c_str(), "r+")) == NULL) {
  printf("Cannot open file %s.\n",p.c_str() );
  exit(1);
  }

  //loop over cuncurrent writers
  for(int i=0; i < rw_ratio; i++)
    {

      //is this node writing this time round?
      if (Layout::nodeNumber()%rw_ratio == i)
	{
	  //Loop over number of sites on this node
	  for(int site=0; site < Layout::sitesOnNode(); site ++)
	    {

	      // get global site index from node site index via global coords
	      int global_site_index = local_site(Layout::siteCoords(Layout::nodeNumber(),site), Layout::lattSize());
      

	      // write site data to correct global place in file
	      fseek(fp,global_site_index*sizemem,SEEK_SET);
	      fwrite(output+sizemem*site,sizemem,1,fp);

	    }
	  //QMP_barrier();
	  QMP_sum_int(&dummy_sum_int);
	}
      else
	//wait in barrier for writing nodes to finish
	QMP_sum_int(&dummy_sum_int);
	//QMP_barrier();
    }

  //close output file
  fclose(fp);

  return my_checksum;

}


//-----------------------------------------------------------------------------
//Read a lattice quantity using parallel I/O. This routine is a
//temporary solution for reading parallel I/O on QCDOC. fread is
//called directly and hence the chroma BinaryReader mechanism
//for I/O is bypassed. This routine takes as its first argument 
//the filename of the file to be read (cf readOLattice which 
// takes the BinaryReader object). The last argument is the ratio of nodes to 
//  concurrent writers. The routine returns a checksum of the data.
QDPUtil::n_uint32_t readOLattice_parallel(const std::string& p, 
			   char* input, size_t size, size_t nmemb,
					  int rw_ratio)
{
  const int xinc = Layout::subgridLattSize()[0];

  size_t sizemem = size*nmemb;
  size_t tot_size = sizemem*xinc;

  //dummy integer for global sum barrier
  int dummy_sum_int = 1;

  //File for direct reading of OLattice
  FILE *fp;

  // All nodes open input file for reading
  if((fp=fopen(p.c_str(), "r")) == NULL) {
  printf("Cannot open file %s.\n",p.c_str() );
  exit(1);
  }

  //loop over cuncurrent readers
  for(int i=0; i < rw_ratio; i++)
    {

      //is this node reading this time round?
      if (Layout::nodeNumber()%rw_ratio == i)
	{
	  //Loop over number of sites on this node
	  for(int site=0; site < Layout::sitesOnNode(); site ++)
	    {

	      // get global site index from node site index via global coords
	      int global_site_index = local_site(Layout::siteCoords(Layout::nodeNumber(),site), Layout::lattSize());
      

	      // read site data from correct global place in file
	      fseek(fp,global_site_index*sizemem,SEEK_SET);
	      fread(input+sizemem*site,sizemem,1,fp);

	    }
	  QMP_sum_int(&dummy_sum_int);
	  //QMP_barrier();
	}
      else
	//wait in barrier for writing nodes to finish
	QMP_sum_int(&dummy_sum_int);
      //QMP_barrier();
    }

  //close input file
  fclose(fp);
   

  //Now compute a checksum. This involves routing all data to 
  //master node who will do this computation. No writing is done at this stage

  QDPUtil::n_uint32_t  my_checksum = 0;
  char *recv_buf = new(nothrow) char[tot_size];

  if( recv_buf == 0x0 ) { 
    QDP_error_exit("Unable to allocate recv_buf\n");
  }

  // Find the location of each site and send to primary node
  int old_node = 0;

  for(int site=0; site < Layout::vol(); site += xinc)
  {
    // first site in each segment uniquely identifies the node
    int node = Layout::nodeNumber(crtesn(site, Layout::lattSize()));

    // Send nodes must wait for a ready signal from the master node
    // to prevent message pileups on the master node
    if (node != old_node)
    {
      // On non-grid machines, use a clear-to-send like protocol
      Internal::clearToSend(recv_buf,sizeof(int),node);
      old_node = node;
    }
    
    // Copy to buffer: be really careful since max(linear) could vary among nodes
    if (Layout::nodeNumber() == node)
    {
      for(int i=0; i < xinc; ++i)
      {
	int linear = Layout::linearSiteIndex(crtesn(site+i, Layout::lattSize()));
	memcpy(recv_buf+i*sizemem, input+linear*sizemem, sizemem);
      }
    }

    // Send result to primary node. Avoid sending prim-node sending to itself
    if (node != 0)
    {
#if 1
      // All nodes participate
      Internal::route((void *)recv_buf, node, 0, tot_size);
#else
      if (Layout::primaryNode())
	Internal::recvFromWait((void *)recv_buf, node, tot_size);

      if (Layout::nodeNumber() == node)
	Internal::sendToWait((void *)recv_buf, 0, tot_size);
#endif
    }

    if (Layout::primaryNode())

      //master node computes checksum
      my_checksum = QDPUtil::crc32(my_checksum, recv_buf, size*nmemb*xinc);
  }

  delete[] recv_buf;
  // End of checksum section

  return my_checksum;

 
}

} // namespace QDP;
