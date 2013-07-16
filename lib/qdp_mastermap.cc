// -*- C++ -*-

#include "qdp.h"
#include "qdp_util.h"

namespace QDP {

  SiteTypeInfo::SiteTypeInfo (int numsites)
    :types(numsites)
  {
#pragma omp parallel for
    for (int i = 0; i < numsites; i++) {
      types[i] = 0; 
    }
  }

  SiteTypeInfo::SiteTypeInfo (const SiteTypeInfo& info)
    :types(info.types.size())
  {
#pragma omp parallel for
    for (int i = 0; i < types.size(); i++) {
      types[i] = info.types[i]; 
    }
  }

  SiteTypeInfo&
  SiteTypeInfo::operator = (const SiteTypeInfo& info)
  {
    if (this != &info) {
      types.resize (info.types.size());
#pragma omp parallel for
      for (int i = 0; i < types.size(); i++) {
	types[i] = info.types[i]; 
      }
    }
    return *this;
  }


  /**
   * Set a site to be a face site
   */
  void 
  SiteTypeInfo::setFaceSite (int site)
  {
    if (site < types.size())
      types[site] = 1;
    else
      QDP_error_exit("internal error: site index %d greater than total number of sites %d\n",
		     site, types.size());
  }

  /**
   * Check a site is a face site or not
   */
  int 
  SiteTypeInfo::isFaceSite (int site) const
  {
    if (site < types.size())
      return types[site];
    else {
      QDP_error_exit("internal error: site index %d greater than total number of sites %d\n",
		     site, types.size());
      return 0; // make compiler happy
    }
  }

  int
  SiteTypeInfo::numberSites (void) const
  {
    return types.size();
  }

  SiteTypeInfo::~SiteTypeInfo (void)
  {
    // empty
  }

  //-------- Master map Implementation --------------------
  MasterMap& MasterMap::Instance()
  {
    static MasterMap singleton;
    return singleton;
  }

  const multi1d<int>& MasterMap::getInnerSites(int bitmask) const 
  { 
    if ( bitmask < 0 || bitmask > powerSetC.size()-1 )
      QDP_error_exit("internal error: getInnerSites");

    return *powerSetC[bitmask]; 
  }
  

  const multi1d<int>& MasterMap::getFaceSites(int bitmask) const { 

    if ( bitmask < 0 || bitmask > powerSet.size()-1 )
      QDP_error_exit("internal error: getInnerSites");

    return *powerSet[bitmask]; 
  }


#if 0
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
#endif

  void MasterMap::complement(multi1d<int>& out, const multi1d<int>& orig,
			     const SiteTypeInfo& tinfo) const 
  {
    int numsites = tinfo.numberSites();
    std::vector<int> c;
    c.reserve(numsites);

    for(int i=0 ; i< numsites; ++i) {
      if (!tinfo.isFaceSite (i))
	c.push_back(i);
    }

    out.resize(c.size());
#pragma omp parallel for
    for(int i=0; i < c.size(); ++i)
      out[i] = c[i];
  }



  void MasterMap::uniquify_list_inplace(multi1d<int>& out , const multi1d<int>& ll,
					SiteTypeInfo& tinfo) const
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

#pragma omp parallel for
    for(int i=0; i < num; ++i) {
      out[i] = d[i];
      tinfo.setFaceSite (out[i]);
    }
  }


  /**
   * Note: For parscalarvec, when a shift happens along x direction,
   * we need to put sites that are on the same vector unit into the surface
   * sites. By doing this, all interior sites are aligned with INNER_LEN
   */
  int MasterMap::registrate(const Map& map) {
    QDP_info ("register map for dir %d sign %d\n",
	      map.mapDir(), map.mapSign());
    int id = 1 << vecPMap.size();
    vecPMap.push_back(&map);

    powerSet.resize( id << 1 );
    powerSetC.resize( id << 1 );

    for (int i = 0 ; i < id ; ++i ) {
      //QDP_info("Processing set %d   id=%d",i,id);
      //QDPIO::cout << i << " powerSet[] size = " << powerSet[i]->size() << "\n";
      //QDPIO::cout << i << " roffset size = " << map.roffset().size() << "\n";


#if defined(ARCH_PARSCALARVEC)  
      multi1d<int> join;
      if (map.mapDir() != 0) 
	join.resize (powerSet[i]->size() + map.roffset().size());
      else
	join.resize (powerSet[i]->size() +  (map.roffset().size() << INNER_LOG));
#else
      multi1d<int> join(powerSet[i]->size() + map.roffset().size());
#endif

#pragma omp parallel for
      for (int q = 0 ; q < powerSet[i]->size() ; ++q ) 
	join[q]=(*powerSet[i])[q];

#if defined(ARCH_PARSCALARVEC)
      if (map.mapDir() != 0) {
	// shift not along x direction, all neighbors are packed and are surface sites
	int pos = powerSet[i]->size();

#pragma omp parallel for
	for (int q = 0; q < map.roffset().size() ; ++q) 
	  join[pos + q]=map.roffset()[q];
      }
      else {
	// shift along x direction. For one surface site there are multiple INNER_LEN - 1
	// sites need to be together with this site. So we count those non-surface sites
	// as surface sites
	int pos = powerSet[i]->size();

#pragma omp parallel for
	for (int q = 0 ; q < map.roffset().size() ; ++q ) {
	  if (map.mapSign() < 0) { 
	    // shift to the right
	    if ((map.roffset()[q] & (INNER_LEN - 1)) != 0) {
	      QDP_error_exit ("Surface receiving offset %d is not multiple of %d. \n",
			      map.roffset()[q], INNER_LEN);
	    }
	    join[pos + q * INNER_LEN] = map.roffset()[q];
	    for (int nq = 1; nq < INNER_LEN; nq++) // assume x direction are packed
	      join[pos + q * INNER_LEN + nq] = map.roffset()[q] + nq;
	  }
	  else {
	    // shift to the left
	    if ((map.roffset()[q] + 1) & (INNER_LEN - 1) != 0) {
	      QDP_error_exit ("Surface receiving offset %d is incompatible with %d. \n",
			      map.roffset()[q], INNER_LEN - 1);
	    }
	    join[pos + q * INNER_LEN] = map.roffset()[q];
	    for (int nq = 1; nq < INNER_LEN; nq++) // assume x direction are packed
	      join[pos + q * INNER_LEN + nq] = map.roffset()[q] - nq;
	  }
	}
      }
#else
      for (int q = 0, pos = powerSet[i]->size() ; q < map.roffset().size() ; ++q ) 
	join[pos++]=map.roffset()[q];
#endif
      //QDP_info("before uniquify: join_size = %d",join.size() );
      //join = uniquify_list(join);
      //QDP_info("after uniquify: join_size = %d",join.size() );

      powerSet[i|id] = new multi1d<int>;
      powerSetC[i|id]= new multi1d<int>;
      
      // create a site type information holder
      SiteTypeInfo tinfo (Layout::sitesOnNode());
      uniquify_list_inplace( *powerSet[i|id] , join, tinfo);
      complement( *powerSetC[i|id] , *powerSet[i|id], tinfo );
      
#if 0
      for (int q = 0 ; q < powerSet[i|id]->size() ; ++q )
	QDP_info("powerSet[%d][%d]=%d",i|id,q,powerSet[i|id][q]);

      for (int q = 0 ; q < powerSetC[i|id]->size() ; ++q )
	QDP_info("powerSetC[%d][%d]=%d",i|id,q,powerSetC[i|id][q]);
#endif
    }

    return id;
  }

} // namespace QDP


