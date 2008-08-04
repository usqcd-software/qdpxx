// -*- C++ -*-
// $Id: qdp_db_fx.h,v 1.2 2008-08-04 04:08:51 edwards Exp $
/*! @file
 * @brief Support for ffdb-lite - a wrapper over Berkeley DB
 */

#ifndef QDP_DB_FX_H
#define QDP_DB_FX_H

#include "qdp_db.h"
#include "ConfFxDSizeStoreDB.h"

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
  class BinaryFxStoreDB : protected FFDB::ConfFxDSizeStoreDB<K,D>
  {
  public:
    /**
     * Empty constructor for a DB
     */
    BinaryFxStoreDB () : FFDB::ConfFxDSizeStoreDB<K,D>() {}

    /**
     * Constructor for a DB
     *
     * @param DB file filename holding keys and data.
     */
    BinaryFxStoreDB (const std::string& file,
		      int max_cache_size = 50000000) 
      : FFDB::ConfFxDSizeStoreDB<K,D>()
    {
      open(file, max_cache_size);
    }

    /*!
      Destroy the object
    */
    ~BinaryFxStoreDB() 
    {
      close();
    }

    /**
     * Open a DB
     *
     * @param DB file filename holding keys and data.
     */
    void open (const std::string& file,
	       int max_cache_size = 50000000) 
    {
      if (Layout::primaryNode())
	FFDB::ConfFxDSizeStoreDB<K,D>::open(file, max_cache_size);
    }

    /*!
      Close the object
    */
    void close () 
    {
      if (Layout::primaryNode())
	FFDB::ConfFxDSizeStoreDB<K,D>::close();
    }

    /**
     * Insert a pair of data and key into the database
     * data is not ensemble, but a vector of complex.
     * @param key a key
     * @param data a user provided data
     */
    void insert (K& key, D& data) 
    {
      if (Layout::primaryNode())
	FFDB::ConfFxDSizeStoreDB<K,D>::insert(key, data);
    }

    /**
     * Get data for a given key
     * @param key user supplied key
     * @param data after the call data will be populated
     * @return 0 on success, otherwise the key not found
     */
    int get (K& key, D& data)
    {
      int ret;
      if (Layout::primaryNode())
	ret = FFDB::ConfFxDSizeStoreDB<K,D>::get(key, data);

      Internal::broadcast(ret);
      return ret;
    }

    /**
     * Return all available keys to user
     * @param keys user suppled an empty vector which is populated
     * by keys after this call.
     */
    void keys (std::vector<K>& kys)
    {
      if (Layout::primaryNode())
	FFDB::ConfFxDSizeStoreDB<K,D>::keys(kys);
    }

    /**
     * Return all pairs of keys and data
     * @param keys user supplied empty vector to hold all keys
     * @param data user supplied empty vector to hold data
     * @return keys and data in the vectors having the same size
     */
    void keysAndData (std::vector<K>& kys, std::vector<D>& vals)
    {
      if (Layout::primaryNode())
	FFDB::ConfFxDSizeStoreDB<K,D>::keysAndData(kys, vals);
    }

    /**
     * Flush database in memory to disk
     */
    void flush (void)
    {
      if (Layout::primaryNode())
	FFDB::ConfFxDSizeStoreDB<K,D>::flush();
    }


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
