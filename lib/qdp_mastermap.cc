// -*- C++ -*-

#include "qdp.h"
#include "qdp_util.h"

namespace QDP {


  MasterMap& MasterMap::Instance()
  {
    static MasterMap singleton;
    return singleton;
  }


  void MasterMap::remove_neg(multi1d<int>& out, const multi1d<int>& orig) const {
    multi1d<int> c(Layout::sitesOnNode());
    int num=0;
    for(int i=0 ; i<Layout::sitesOnNode() ; ++i) 
      if (orig[i] >= 0)
	c[num++]=i;
    out.resize( num );
    for(int i=0; i < num; ++i)
      out[i] = c[i];
  }



  void MasterMap::uniquify_list_inplace(multi1d<int>& out , const multi1d<int>& ll) const
  {
    multi1d<int> d(ll.size());

    // Enter the first element as unique to prime the search
    int ipos = 0;
    int num = 0;
    int prev_node;
  
    d[num++] = prev_node = ll[ipos++];

    // Find the unique source nodes
    while (ipos < ll.size())
      {
	int this_node = ll[ipos++];

	if (this_node != prev_node)
	  {
	    // Has this node occured before?
	    bool found = false;
	    for(int i=0; i < num; ++i)
	      if (d[i] == this_node)
		{
		  found = true;
		  break;
		}

	    // If this is the first time this value has occurred, enter it
	    if (! found)
	      d[num++] = this_node;
	  }

	prev_node = this_node;
      }

    // Copy into a compact size array
    out.resize(num);
    for(int i=0; i < num; ++i) {
      out[i] = d[i];
    }

  }





  int MasterMap::registrate(const Map& map) {
    //QDP_info("Map registered id=%d (total=%d)",1 << vecPMap.size(),vecPMap.size()+1 );
    int id = 1 << vecPMap.size();
    vecPMap.push_back(&map);

    //QDP_info("Resizing power set to %d", id << 1 );
    powerSet.resize( id << 1 );
    powerSetC.resize( id << 1 );
    idInner.resize( id << 1 );
    idFace.resize( id << 1 );

    for (int i = 0 ; i < id ; ++i ) {

      multi1d<int> ct(Layout::sitesOnNode());
      for(int q=0 ; q<Layout::sitesOnNode() ; ++q) 
	ct[q]=q;

      multi1d<int> join(powerSet[i]->size() + map.roffset().size());
      for (int q = 0 ; q < powerSet[i]->size() ; ++q ) {
	join[q]=(*powerSet[i])[q];
	ct[ (*powerSet[i])[q] ] = -1;
      }

      for (int q = 0, pos = powerSet[i]->size() ; q < map.roffset().size() ; ++q ) {
	join[pos++]=map.roffset()[q];
	ct[ map.roffset()[q] ] = -1;
      }

      powerSet[i|id] = new multi1d<int>;
      powerSetC[i|id]= new multi1d<int>;

      uniquify_list_inplace( *powerSet[i|id] , join );

      remove_neg( *powerSetC[i|id] , ct );

      idFace[i|id] = QDPCache::Instance().registrateOwnHostMem( powerSet[i|id]->size() * sizeof(int) , (void*)powerSet[i|id]->slice() );
      idInner[i|id] = QDPCache::Instance().registrateOwnHostMem( powerSetC[i|id]->size() * sizeof(int) , (void*)powerSetC[i|id]->slice() );

    }
    return id;
  }


  int MasterMap::getIdInner(int bitmask) const {
    if ( bitmask < 0 || bitmask > powerSetC.size()-1 )
      QDP_error_exit("internal error: get id inner");
    return idInner[bitmask]; 
  }
  int MasterMap::getIdFace(int bitmask) const {
    if ( bitmask < 0 || bitmask > powerSetC.size()-1 )
      QDP_error_exit("internal error: get id face");
    return idFace[bitmask];
  }
  int MasterMap::getCountInner(int bitmask) const {
    if ( bitmask < 0 || bitmask > powerSet.size()-1 )
      QDP_error_exit("internal error: get count inner");
    return powerSetC[bitmask]->size(); 
  }
  int MasterMap::getCountFace(int bitmask) const {
    if ( bitmask < 0 || bitmask > powerSetC.size()-1 )
      QDP_error_exit("internal error: get count face");
    return powerSet[bitmask]->size(); 
  }


} // namespace QDP


