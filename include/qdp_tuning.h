#ifndef QDP_TUNING_H
#define QDP_TUNING_H


#include <string>
#include <map>

using namespace std;


namespace QDP {

  typedef map<int,int>             MapVolumes; // local Vol -> threads/block
  typedef map<string, MapVolumes > MapTuning;  // qdp_exptr -> MapVolumes



  class TuningLocalVol {
  public:
    TuningLocalVol(){}
    TuningLocalVol(const int& vol,const int& threads): vol(vol), threads(threads) {}
    int vol;
    int threads;
  };

  class TuningHash {
  public:
    TuningHash(){}
    TuningHash(const string& s , const MapVolumes& m): hash(s), volumes(m.size()) {
      int pos=0;
      for (MapVolumes::const_iterator i = m.begin() ; i != m.end() ; ++i ) {
	TuningLocalVol localVol(i->first,i->second);
	volumes[pos++] = localVol;
      }
    }

    string hash;
    multi1d<TuningLocalVol> volumes;
  };

  class TuningDB {
  public:
    TuningDB(){}
    TuningDB(const MapTuning& mt): hashs(mt.size()) {
      int pos=0;
      for (MapTuning::const_iterator i = mt.begin() ; i != mt.end() ; ++i ) {
	TuningHash hash( i->first , i->second );
	hashs[pos++]=hash;
      }
    }

    multi1d<TuningHash> hashs;
  };

  void write(QDP::XMLWriter& xml, const string& str, const TuningLocalVol& lvol);
  void write(QDP::XMLWriter& xml, const string& str, const TuningHash& hash);
  void write(QDP::XMLFileWriter& xml, const string& str, const TuningDB& db);

  void read(QDP::XMLReader& xml0, const string& str, TuningLocalVol& lvol);
  void read(QDP::XMLReader& xml0, const string& str, TuningHash& hash);
  void read(QDP::XMLReader& xml0, const string& str, TuningDB& db);



  class JitTuning {
  public:
    static JitTuning& Instance() {
      static JitTuning singleton;
      return singleton;
    }

    void setResourcePath( string path0 );
    string getResourcePath();
    MapTuning& getMapTuning();    
    void load(const string& fname);
    void save(const string& fname);
    void save_all(const string& fname);

  private:
    JitTuning();                                 // Private constructor
    JitTuning(const JitTuning&);                 // Prevent copy-construction
    JitTuning& operator=(const JitTuning&);

    void createMapTuning(MapTuning& mt,const TuningDB& db);
    void mergeMapTuning(MapTuning& dest,const MapTuning& src);

    string path;
    MapTuning mapTuning;
  };









}




#endif
