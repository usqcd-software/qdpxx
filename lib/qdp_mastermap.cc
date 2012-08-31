// -*- C++ -*-

#include "qdp.h"
#include "qdp_util.h"

namespace QDP {


  MasterMap& MasterMap::Instance()
  {
    static MasterMap singleton;
    return singleton;
  }

  const multi1d<int>& MasterMap::getInnerSites(int bitmask) const { 

    if ( bitmask < 0 || bitmask > powerSetC.size()-1 )
      QDP_error_exit("internal error: getInnerSites");

    return *powerSetC[bitmask]; 
  }

  const multi1d<int>& MasterMap::getFaceSites(int bitmask) const { 

    if ( bitmask < 0 || bitmask > powerSet.size()-1 )
      QDP_error_exit("internal error: getInnerSites");

    return *powerSet[bitmask]; 
  }


  void MasterMap::complement(multi1d<int>& out, const multi1d<int>& orig) const {
    std::vector<int> c;
    c.reserve(Layout::sitesOnNode());
    for(int i=0 ; i<Layout::sitesOnNode() ; ++i) {

      bool found = false;
      for(int q=0; q < orig.size(); ++q)
	if (orig[q] == i)
	{
	  found = true;
	  break;
	}
      
      if (!found)
	c.push_back(i);

    }

    out.resize( c.size() );
    for(int i=0; i < c.size(); ++i)
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
    for(int i=0; i < num; ++i)
      out[i] = d[i];

  }





  int MasterMap::registrate(const Map& map) {
    //QDP_info("Map registered id=%d (total=%d)",1 << vecPMap.size(),vecPMap.size()+1 );
    int id = 1 << vecPMap.size();
    vecPMap.push_back(&map);

    //QDP_info("Resizing power set to %d", id << 1 );
    powerSet.resize( id << 1 );
    powerSetC.resize( id << 1 );

    for (int i = 0 ; i < id ; ++i ) {
      //QDP_info("Processing set %d   id=%d",i,id);
      //QDPIO::cout << i << " powerSet[] size = " << powerSet[i]->size() << "\n";
      //QDPIO::cout << i << " roffset size = " << map.roffset().size() << "\n";

      multi1d<int> join(powerSet[i]->size() + map.roffset().size());
      for (int q = 0 ; q < powerSet[i]->size() ; ++q ) 
	join[q]=(*powerSet[i])[q];
      for (int q = 0, pos = powerSet[i]->size() ; q < map.roffset().size() ; ++q ) 
	join[pos++]=map.roffset()[q];
      //QDP_info("before uniquify: join_size = %d",join.size() );
      //join = uniquify_list(join);
      //QDP_info("after uniquify: join_size = %d",join.size() );

      powerSet[i|id] = new multi1d<int>;
      powerSetC[i|id]= new multi1d<int>;
      uniquify_list_inplace( *powerSet[i|id] , join );
      complement( *powerSetC[i|id] , *powerSet[i|id] );

      // for (int q = 0 ; q < powerSet[i|id].size() ; ++q )
      //  	QDP_info("powerSet[%d][%d]=%d",i|id,q,powerSet[i|id][q]);

      // for (int q = 0 ; q < powerSetC[i|id].size() ; ++q )
      //  	QDP_info("powerSetC[%d][%d]=%d",i|id,q,powerSetC[i|id][q]);
    }

    return id;
  }



} // namespace QDP


