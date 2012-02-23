// -*- C++ -*-
/*! \file
 *  \brief A Map Object that works lazily from Disk
 */


#ifndef __qdp_map_obj_disk_multiple_h__
#define __qdp_map_obj_disk_multiple_h__

#include "qdp_map_obj_disk.h"
#include <vector>

namespace QDP
{

  //----------------------------------------------------------------------------
  //! Class that holds multiple DBs. Can only be used in a read-only mode.
  template<typename K, typename V>
  class MapObjectDiskMultiple
  {
  public:
    //! Empty constructor
    MapObjectDiskMultiple() {}

    //! Finalizes object
    ~MapObjectDiskMultiple() {}

    //! Set debugging level
    void setDebug(int level)
    {
      for(int i=0; i < dbs_.size(); ++i)
	dbs_[i].setDebug(level);
    }

    //! Get debugging level
    int getDebug() const {return dbs_[0].getDebug();}

    //! Open files
    void open(const std::vector<std::string>& files)
    {
      dbs_.resize(files.size());

      for(int i=0; i < dbs_.size(); ++i)
      {
//	dbs_[i].open(files[i], std::ios_base::in);
	dbs_[i].open(files[i]);
      }
    }


    //! Check if a DB file exists before opening.
    bool fileExists(const std::vector<std::string>& files) const
    {
      bool ret = true;
      for(int i=0; i < dbs_.size(); ++i)
      {
	ret = dbs_[i].fileExists(files[i]);
	if (! ret)
	  return ret;
      }

      return ret;
    }


    //! Close the files
    void close()
    {
      for(int i=0; i < dbs_.size(); ++i)
      {
	dbs_[i].close();
      }
    }


    /**
     * Get val for a given key
     * @param key user supplied key
     * @param val after the call val will be populated
     * @return 0 on success, otherwise the key not found
     */
    int get(const K& key, V& val) const
    {
      int ret = -1;

      for(int i=0; i < dbs_.size(); ++i) 
      {
	ret = dbs_[i].get(key, val);
	if (ret == 0)
	  break;
      }
      return ret;
    }


    /**
     * Flush database in memory to disk
     */
    void flush()
    {
      for(int i=0; i < dbs_.size(); ++i)
      {
	dbs_[i].flush();
      }
    }


    /**
     * Does this key exist in the store
     * @param key a key object
     * @return true if the answer is yes
     */
    bool exist(const K& key) const
    {
      bool ret = false;
      
      for(int i=0; i < dbs_.size(); ++i)
      {
	ret = dbs_[i].exist(key);
	if (ret)
	  break;
      }
      
      return ret;
    }

    /**
     * Get user user data from the metadata database
     *
     * @param user_data user supplied buffer to store user data
     * @return returns 0 if success. Otherwise failure.
     */
    int getUserdata(std::string& user_data) const
    {
      int ret = -1;

      if (dbs_.size() == 0)
	return ret;

      // Grab the first db
      ret = dbs_[0].getUserdata(user_data);

      return ret;
    }

  private:
    //! Hide
    MapObjectDiskMultiple(const MapObjectDiskMultiple&) {}

    //! Hide
    void operator=(const MapObjectDiskMultiple&) {}

  private:
    //! Array of read-only maps
    std::vector< MapObjectDisk<K,V> > dbs_;
  };

} // namespace Chroma

#endif
