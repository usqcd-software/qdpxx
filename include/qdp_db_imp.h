// -*- C++ -*-
// $Id: qdp_db_imp.h,v 1.2 2008-08-26 18:44:50 edwards Exp $
/*! @file
 * @brief Support for ffdb-lite - a wrapper over Berkeley DB
 */

#ifndef QDP_DB_IMP_H
#define QDP_DB_IMP_H

#include "qdp_layout.h"
#include "ConfDataStoreDB.h"
#include "ConfVarDSizeStoreDB.h"
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
  class BinaryStoreDB : protected FFDB::ConfDataStoreDB<K,D>
  {
  public:
    /**
     * Empty constructor for a DB
     */
    BinaryStoreDB() {}

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
    virtual std::string storageName (void) const = 0;

    
    /**
     * Insert user data into the  metadata database
     *
     * @param user_data user supplied data
     * @return returns 0 if success, else failure
     */
    virtual int insertUserdata (const std::string& user_data)
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
    virtual int getUserdata (std::string& user_data)
    {
      int ret;
      if (Layout::primaryNode())
	ret = FFDB::ConfDataStoreDB<K,D>::getUserdata(user_data);

      Internal::broadcast(ret);
      return ret;
    }
  };


  //--------------------------------------------------------------------------------
  //!  DB Base class using DBTree
  /*!
    This class is used for writing of user data (most usefully measurements)
    into a Berkeley DB with a key/value semantics. 
  */
  template<typename K, typename D>
  class BinaryVarStoreDB : protected FFDB::ConfVarDSizeStoreDB<K,D>
  {
  public:
    /**
     * Empty constructor for a DB
     */
    BinaryVarStoreDB () : FFDB::ConfVarDSizeStoreDB<K,D>() {}

    /**
     * Constructor for a DB
     *
     * @param DB file filename holding keys and data.
     */
    BinaryVarStoreDB (const std::string& file,
		      int max_cache_size = 50000000,
		      int pagesize = db_pagesize)
      : FFDB::ConfVarDSizeStoreDB<K,D>()
    {
      open(file, max_cache_size, pagesize);
    }

    /*!
      Destroy the object
    */
    ~BinaryVarStoreDB() 
    {
      close();
    }

    /**
     * Open a DB
     *
     * @param DB file filename holding keys and data.
     */
    void open (const std::string& file,
	       int max_cache_size = 50000000,
	       int pagesize = db_pagesize)
    {
      if (Layout::primaryNode())
	FFDB::ConfVarDSizeStoreDB<K,D>::open(file, max_cache_size, pagesize);
    }

    /*!
      Close the object
    */
    void close () 
    {
      if (Layout::primaryNode())
	FFDB::ConfVarDSizeStoreDB<K,D>::close();
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
	FFDB::ConfVarDSizeStoreDB<K,D>::insert(key, data);
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
	ret = FFDB::ConfVarDSizeStoreDB<K,D>::get(key, data);
      else
	notImplemented();

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
	FFDB::ConfVarDSizeStoreDB<K,D>::keys(kys);
      else
	notImplemented();
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
	FFDB::ConfVarDSizeStoreDB<K,D>::keysAndData(kys, vals);
      else
	notImplemented();
    }

    /**
     * Flush database in memory to disk
     */
    void flush (void)
    {
      if (Layout::primaryNode())
	FFDB::ConfVarDSizeStoreDB<K,D>::flush();
    }


    /**
     * Name of database associated with this Data store
     *
     * @return database name
     */
    std::string storageName (void) const 
    {
      std::string filename_;
      if (Layout::primaryNode())
	filename_ = FFDB::ConfVarDSizeStoreDB<K,D>::storageName();
      
      // broadcast string
      Internal::broadcast_str(filename_);

      return filename_;
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
	ret = FFDB::ConfVarDSizeStoreDB<K,D>::insertUserdata(user_data);

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
	ret = FFDB::ConfVarDSizeStoreDB<K,D>::getUserdata(user_data);
      else
	notImplemented();

      Internal::broadcast(ret);
      return ret;
    }

  private:
    void notImplemented() const
    {
      QDP_error_exit("Berkeley DB read routines do not work (yet) in parallel - only single node");
    }

  };


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
		     int max_cache_size = 50000000,
		     int pagesize = db_pagesize)
      : FFDB::ConfFxDSizeStoreDB<K,D>()
    {
      open(file, max_cache_size, pagesize);
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
	       int max_cache_size = 50000000,
	       int pagesize = db_pagesize)
    {
      if (Layout::primaryNode())
	FFDB::ConfFxDSizeStoreDB<K,D>::open(file, max_cache_size, pagesize);
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
      int ret=0;
      if (Layout::primaryNode())
	ret = FFDB::ConfFxDSizeStoreDB<K,D>::get(key, data);
      else
	notImplemented();

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
      else
	notImplemented();
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
      else
	notImplemented();
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
    std::string storageName (void) const 
    {
      std::string filename_;
      if (Layout::primaryNode())
	filename_ = FFDB::ConfFxDSizeStoreDB<K,D>::storageName();
      
      // broadcast string
      Internal::broadcast_str(filename_);

      return filename_;
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
	ret = FFDB::ConfFxDSizeStoreDB<K,D>::insertUserdata(user_data);

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
	ret = FFDB::ConfFxDSizeStoreDB<K,D>::getUserdata(user_data);
      else
	notImplemented();

      Internal::broadcast(ret);
      return ret;
    }

  private:
    void notImplemented() const
    {
      QDP_error_exit("Berkeley DB read routines do not work (yet) in parallel - only single node");
    }

  };


}  // namespace QDP

#endif
