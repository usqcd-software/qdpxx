#include "qdp.h"

namespace QDP {



  void JitTuning::setResourcePath( string path0 ) { 
    QDP_info("Setting JIT tuning DB file to %s",path0.c_str());
    path=path0; 
  } 

  string JitTuning::getResourcePath() { return path; }

  MapTuning& JitTuning::getMapTuning() { return mapTuning; }
    
  void JitTuning::load(const string& fname) {
    QDP_info("Reading tuning DB %s",fname.c_str());
    TuningDB db_in;
    XMLReader xml_in(fname);
    read(xml_in,"/db",db_in);
    createMapTuning(mapTuning,db_in);
    xml_in.close();
  }

  void JitTuning::save(const string& fname) {
    TuningDB db(mapTuning);
    QDP_info("Writing tuning DB %s",fname.c_str());
    XMLFileWriter xml(fname);
    write(xml,"db",db);
    xml.close();
  }


  JitTuning::JitTuning() {};                                 // Private constructor

  void JitTuning::createMapTuning(MapTuning& mt,const TuningDB& db)
  {
    for (int i=0;i<db.hashs.size();++i) {
      for (int q=0;q<db.hashs[i].volumes.size();++q) {
	mt[db.hashs[i].hash][db.hashs[i].volumes[q].vol]=db.hashs[i].volumes[q].threads;
      }
    }
  }




  void write(QDP::XMLWriter& xml, const string& str, const TuningLocalVol& lvol) {
    push(xml,str);
    write(xml,"volume",lvol.vol);
    write(xml,"threads",lvol.threads);
    pop(xml);
  }

  void write(QDP::XMLWriter& xml, const string& str, const TuningHash& hash) {
    push(xml,str);
    write(xml,"hash",hash.hash);
    write(xml,"volumes",hash.volumes);
    pop(xml);
  }

  void write(QDP::XMLFileWriter& xml, const string& str, const TuningDB& db) {
    push(xml,str);
    write(xml,"DB",db.hashs);
    pop(xml);
  }



  void read(QDP::XMLReader& xml0, const string& str, TuningLocalVol& lvol) {
    XMLReader xml( xml0 , str );
    read(xml,"volume",lvol.vol);
    read(xml,"threads",lvol.threads);
  }

  void read(QDP::XMLReader& xml0, const string& str, TuningHash& hash) {
    XMLReader xml( xml0 , str );
    read(xml,"hash",hash.hash);
    read(xml,"volumes",hash.volumes);
  }

  void read(QDP::XMLReader& xml0, const string& str, TuningDB& db) {
    XMLReader xml( xml0 , str );
    read(xml,"DB",db.hashs);
  }


}

