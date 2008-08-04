// -*- C++ -*-
// $Id: qdp_db.h,v 1.1 2008-08-04 01:54:28 edwards Exp $
/*! @file
 * @brief Support for ffdb-lite - a wrapper over Berkeley DB
 */

#ifndef QDP_DB_H
#define QDP_DB_H

#include "qdp_layout.h"
#include "ConfDataStoreDB.h"

namespace QDP
{
  using namespace FFDB;

  /*! @defgroup io IO
   *
   * Berkeley DB support
   *
   * @{
   */

  //--------------------------------------------------------------------------------
  //!  DB Base class
  /*!
    This class is used for writing of user data (most usefully measurements)
    into a Berkeley DB with a key/value semantics. 
  */
  template<typename K, typename D>
  class BinaryStoreDB : protected FFDB::ConfDataStoreDB<K,D>
  {
  public:
    /**
     * Constructor for a DB
     *
     * @param DB file filename holding keys and data.
     */
    BinaryStoreDB (const std::string& file,
		   int max_cache_size = 50000000) 
      : FFDB::ConfDataStoreDB<K,D>(file, max_cache_size) {}

    /*!
      Destroy the object
    */
    virtual ~BinaryStoreDB() {}

    /**
     * Insert a pair of data and key into the database
     * data is not ensemble, but a vector of complex.
     * @param key a key
     * @param data a user provided data
     */
    virtual void insert (K& key, D& data) = 0;

    /**
     * Get data for a given key
     * @param key user supplied key
     * @param data after the call data will be populated
     * @return 0 on success, otherwise the key not found
     */
    virtual int get (K& key, D& data) = 0;

    /**
     * Return all available keys to user
     * @param keys user suppled an empty vector which is populated
     * by keys after this call.
     */
    virtual void keys (std::vector<K>& keys) = 0;

    /**
     * Return all pairs of keys and data
     * @param keys user supplied empty vector to hold all keys
     * @param data user supplied empty vector to hold data
     * @return keys and data in the vectors having the same size
     */
    virtual void keysAndData (std::vector<K>& keys, std::vector<D>& values) = 0;

    /**
     * Flush database in memory to disk
     */
    virtual void flush (void) = 0;


    /**
     * Name of database associated with this Data store
     *
     * @return database name
     */
    const std::string
    storageName (void) const 
    {
      std::string filename_;
      if (Layout::primaryNode())
	filename_ = FFDB::ConfDataStoreDB<K,D>::storeageName();
      
      // broadcast string
      Internal::broadcast_str(filename_);
    }

    
    /**
     * Insert user data into the  metadata database
     *
     * @param user_data user supplied data
     * @return returns 0 if success, else failure
     */
    int insertUserdata (const std::string& user_data)
    {
      int ret;
      if (Layout::primaryNode())
	ret = FFDB::ConfDataStoreDB<K,D>::insertUserdata(user_data);

      Internal::broadcast(ret);
      return ret;
    }
    
    /**
     * Get user user data from the metadata database
     *
     * @param user_data user supplied buffer to store user data
     * @return returns 0 if success. Otherwise failure.
     */
    int getUserdata (std::string& user_data)
    {
      int ret;
      if (Layout::primaryNode())
	ret = FFDB::ConfDataStoreDB<K,D>::getUserdata(user_data);

      Internal::broadcast(ret);
      return ret;
    }
  };


}  // namespace QDP

#endif
