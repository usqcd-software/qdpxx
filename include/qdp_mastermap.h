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
    void complement(multi1d<int>& out, const multi1d<int>& orig) const;
    void uniquify_list_inplace(multi1d<int>& out , const multi1d<int>& ll) const;

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
