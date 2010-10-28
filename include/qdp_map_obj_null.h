// -*- C++ -*-
/*! \file
 *  \brief A null map object 
 */

#ifndef __qdp_map_obj_null_h__
#define __qdp_map_obj_null_h__

#include "qdp_map_obj.h"
#include <vector>

namespace QDP
{

  //----------------------------------------------------------------------------
  //! A wrapper over maps
  template<typename K, typename V>
  class MapObjectNull : public MapObject<K,V>
  {
  public:
    //! Default constructor
    MapObjectNull() {}

    //! Destructor
    ~MapObjectNull() {}

    //! Insert
    int insert(const K& key, const V& val) {return 1;}

    //! Accessor
    int get(const K& key, V& val) {return 1;}
    
    //! Flush out state of object
    void flush() {}

    //! Exists?
    bool exist(const K& key) const {return false;}
			
    //! The number of elements
    unsigned int size() const {return 0;}

    //! Dump keys
    void keys(std::vector<K>& _keys) const {}

    //! Insert user data into the metadata database
    int insertUserdata(const std::string& user_data) {}
    
    //! Get user user data from the metadata database
    int getUserdata(std::string& user_data) {}

    /*! 
     * These extend the bacis MapObject Interface. 
     * The iterators are used to QIO the object
     * Need to be public for now 
     */

    //! Usual begin iterator
    //! Map type convenience
    typedef std::map<K,V> MapType_t;
    

    //! Annoying, need these to satisfy the map
    typename MapType_t::const_iterator begin() const {return src_map.begin();}
    
    //! Usual end iterator
    typename MapType_t::const_iterator  end() const {return src_map.end();}

  private:
    //! Map of objects
    mutable MapType_t src_map;
  };

} // namespace QDP

#endif
