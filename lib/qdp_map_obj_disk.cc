/*! \file
 *  \brief A Map Object that works lazily from Disk
 */

#include "qdp_map_obj_disk.h"
#include <string>

namespace QDP 
{ 
  namespace MapObjDiskEnv 
  {
    // Anonymous namespace
    namespace {
      const std::string file_magic="XXXXQDPLazyDiskMapObjFileXXXX";
    };

    // Magic string at start of file.
    const std::string& getFileMagic() {return file_magic;}
  };


  std::string
  peekMapObjectDiskTypeCode(const std::string& filename)
  {
    BinaryFileReader reader;
    reader.open(filename);
    std::string read_magic;
    reader.readDesc(read_magic);
    
    // Check magic
    if (read_magic != MapObjDiskEnv::file_magic) { 
      QDPIO::cerr << "Magic String Wrong: Expected: " << MapObjDiskEnv::file_magic << " but read: " << read_magic << endl;
      QDP_abort(1);
    }

#ifdef DISK_OBJ_DEBUGGING
    QDPIO::cout << "Read File Magic. Current Position: " << reader.currentPosition() << endl;
#endif
      
    MapObjDiskEnv::file_version_t read_version;
    read(reader, read_version);
      
#ifdef DISK_OBJ_DEBUGGING
    QDPIO::cout << "Read File Verion. Current Position: " << reader.currentPosition() << endl;
#endif
      
    // Check version
    QDPIO::cout << "MapObjectDisk: file has version: " << read_version << endl;
    
    std::string user_data;
    readDesc(reader, user_data);
#ifdef DISK_OBJ_DEBUGGING
    QDPIO::cout << "Read File Type String. Code=" << user_data << ". Current Position: " << reader.currentPosition() << endl;
#endif
    
    reader.close();
    return user_data;
  }

}
    
