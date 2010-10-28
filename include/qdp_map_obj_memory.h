// -*- C++ -*-
/*! \file
 *  \brief A memory based map object 
 */

#ifndef __qdp_map_obj_memory_h__
#define __qdp_map_obj_memory_h__

#include "qdp_map_obj.h"
#include <vector>

namespace QDP
{

  //----------------------------------------------------------------------------
  //! A wrapper over maps
  template<typename K, typename V>
  class MapObjectMemory : public MapObject<K,V>
  {
  public:
    //! Default constructor
    MapObjectMemory() {}

    //! Destructor
    ~MapObjectMemory() {}

    //! Insert
    int insert(const K& key, const V& val) {
      int ret = 0;
      if( exist(key) ) {
	src_map[key] = val; // Update
      }
      else
      {
	src_map.insert(std::make_pair(key,val));
      }

      return ret;
    }

    //! Accessor
    int get(const K& key, V& val) const { 
      int ret = 0;
      if ( exist(key) ) {
	val = src_map.find(key)->second;
      }
      else {
	ret = 1;
      }
      return ret;
    }

    //! Flush out state of object
    void flush() {}

    //! Exists?
    bool exist(const K& key) const {
      return (src_map.find(key) == src_map.end()) ? false : true;
    }
			
    //! The number of elements
    unsigned int size() const {return static_cast<unsigned long>(src_map.size());}

    //! Dump keys
    void keys(std::vector<K>& keys_) const {
      typename MapType_t::const_iterator iter;
      for(iter  = src_map.begin();
	  iter != src_map.end();
	  ++iter) { 
	keys_.push_back(iter->first);
      }
    }


    //! Insert user data into the metadata database
    int insertUserdata(const std::string& user_data_) {
      user_data = user_data_;
      return 0;
    }
    
    //! Get user user data from the metadata database
    int getUserdata(std::string& user_data_) const {
      user_data_ = user_data;
      return 0;
    }

    /*! 
     * These extend the bacis MapObject Interface. 
     * The iterators are used stream through the object
     * Need to be public for now 
     */

    //! Usual begin iterator
    //! Map type convenience
    typedef std::map<K,V> MapType_t;
    
 
    typename MapType_t::const_iterator begin() const {return src_map.begin();}
    
    //! Usual end iterator
    typename MapType_t::const_iterator  end() const {return src_map.end();}

  private:
    //! Map of objects
    mutable MapType_t src_map;
    string user_data;
  };

} // namespace QDP

#endif
