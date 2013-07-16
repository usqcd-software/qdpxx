// -*- C++ -*-

#ifndef QDP_MASTERMAP_H
#define QDP_MASTERMAP_H

namespace QDP {

  /**
   * This class keep tracks the type of each site.
   * Some sites are on face, and some sites are interior sites
   */
  class SiteTypeInfo {
  public:
    /**
     * Constructor. Build the information from total number of sites
     */
    SiteTypeInfo (int num_sites);

    /**
     * Copy construtor
     */
    SiteTypeInfo (const SiteTypeInfo& info);

    /**
     * Assignment operator
     */
    SiteTypeInfo &
    operator = (const SiteTypeInfo& info);

    /**
     * Destructor
     */
    ~SiteTypeInfo (void);

    /**
     * Total number of sites
     */
    int numberSites (void) const;

    /**
     * Set a site to be a face site
     */
    void setFaceSite (int site);

    /**
     * Check a site is a face site or not
     */
    int isFaceSite (int site) const;

  private:
    multi1d<int> types; // 1 means it is a face site
  };

  class MasterMap {
  public:
    static MasterMap& Instance();
    int registrate(const Map& map);
    // return all inner sites
    const multi1d<int>& getInnerSites(int bitmask) const;

    // return all face sites
    const multi1d<int>& getFaceSites(int bitmask) const;

  private:
#if 0
    void complement(multi1d<int>& out, const multi1d<int>& orig) const;
    void uniquify_list_inplace(multi1d<int>& out , const multi1d<int>& ll) const;
#endif

    void complement(multi1d<int>& out, 
		    const multi1d<int>& orig,
		    const SiteTypeInfo& tinfo) const;

    void uniquify_list_inplace(multi1d<int>& out , 
			       const multi1d<int>& ll, 
			       SiteTypeInfo& tinfo) const;

    MasterMap() {
      //QDP_info("MasterMap() reserving");
      powerSet.reserve(2048);
      powerSetC.reserve(2048);
      powerSet.resize(1);
      powerSet[0] = new multi1d<int>;
      ////QDPIO::cout << "powerSet[0] size = " << powerSet[0]->size() << "\n";
    }

    std::vector<const Map*> vecPMap;
    std::vector< multi1d<int>* > powerSet; // Power set of roffsets
    std::vector< multi1d<int>* > powerSetC; // Power set of complements
  };

} // namespace QDP

#endif
