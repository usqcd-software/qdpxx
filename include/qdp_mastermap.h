// -*- C++ -*-

#ifndef QDP_MASTERMAP_H
#define QDP_MASTERMAP_H

namespace QDP {

  class MasterMap {
  public:
    static MasterMap& Instance();
    int registrate(const Map& map);
    const multi1d<int>& getInnerSites(int bitmask) const;
    const multi1d<int>& getFaceSites(int bitmask) const;

  private:
    multi1d<int> complement(const multi1d<int>& orig) const;
    MasterMap() {
      //QDP_info("MasterMap() reserving");
      powerSet.reserve(2048);
      powerSetC.reserve(2048);
    }

    std::vector<const Map*> vecPMap;
    std::vector< multi1d<int> > powerSet; // Power set of roffsets
    std::vector< multi1d<int> > powerSetC; // Power set of complements
  };

} // namespace QDP

#endif
