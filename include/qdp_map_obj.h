// -*- C++ -*-
/*! \file
 * \brief Wrapper over maps
 */

#ifndef __qdp_map_obj_h__
#define __qdp_map_obj_h__

#include <map>
#include <vector>

#include "qdp.h"

namespace QDP
{

  //----------------------------------------------------------------------------
  //! A wrapper over maps
  template<typename K, typename V>
  class MapObject
  {
  public:
    //! Virtual Destructor
    virtual	
    ~MapObject() {}

    //! Open write mode (inserts)
    virtual
    void openWrite(void) = 0;

    //! Open object in read mode (lookups)
    virtual
    void openRead(void) = 0;

    //! Open
    virtual 
    void openUpdate(void) = 0;

    //! Exists?
    virtual
    bool exist(const K& key) const = 0;
			
    //! Insert
    virtual
    void insert(const K& key, const V& val) = 0;
 
    //! Update
    virtual 
    void update(const K& key, const V& val) = 0;

    //! Other accessor
    virtual
    void lookup(const K& key, V& val) const = 0;

    //! Size of Map
    virtual
    unsigned int size() const = 0;

    //! Dump keys
    virtual
    std::vector<K> keys() const = 0;

  };

} // namespace QDP

#endif
