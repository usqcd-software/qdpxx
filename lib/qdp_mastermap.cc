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

    return powerSetC[bitmask]; 
  }

  const multi1d<int>& MasterMap::getFaceSites(int bitmask) const { 

    if ( bitmask < 0 || bitmask > powerSet.size()-1 )
      QDP_error_exit("internal error: getInnerSites");

    return powerSet[bitmask]; 
  }


  multi1d<int> MasterMap::complement(const multi1d<int>& orig) const {
    std::vector<int> c;
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

    multi1d<int> dd(c.size());
    for(int i=0; i < c.size(); ++i)
      dd[i] = c[i];

    return dd;
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
      multi1d<int> join(powerSet[i].size() + map.roffset().size());
      for (int q = 0 ; q < powerSet[i].size() ; ++q ) 
	join[q]=powerSet[i][q];
      for (int q = 0, pos = powerSet[i].size() ; q < map.roffset().size() ; ++q ) 
	join[pos++]=map.roffset()[q];
      //QDP_info("before uniquify: join_size = %d",join.size() );
      join = uniquify_list(join);
      //QDP_info("after uniquify: join_size = %d",join.size() );

      powerSet[i|id] = join;
      powerSetC[i|id] = complement(join);

      // for (int q = 0 ; q < powerSet[i|id].size() ; ++q )
      //  	QDP_info("powerSet[%d][%d]=%d",i|id,q,powerSet[i|id][q]);

      // for (int q = 0 ; q < powerSetC[i|id].size() ; ++q )
      //  	QDP_info("powerSetC[%d][%d]=%d",i|id,q,powerSetC[i|id][q]);
    }

    return id;
  }



} // namespace QDP


