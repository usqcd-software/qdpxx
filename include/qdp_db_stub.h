// -*- C++ -*-
// $Id: qdp_db_stub.h,v 1.3 2008-08-07 16:18:52 edwards Exp $
/*! @file
 * @brief Stubs of wrappers over Berkeley DB
 */

#ifndef QDP_DB_STUB_H
#define QDP_DB_STUB_H

#include <vector>
#include <string>

namespace QDP
{
  namespace FFDB
  {
     // empty - just for making the compilers happy.
  }

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
  class BinaryStoreDB
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
    virtual int insertUserdata (const std::string& user_data) = 0;
    
    /**
     * Get user user data from the metadata database
     *
     * @param user_data user supplied buffer to store user data
     * @return returns 0 if success. Otherwise failure.
     */
    virtual int getUserdata (std::string& user_data) = 0;
  };


  //--------------------------------------------------------------------------------
  //!  DB Base class using DBTree
  /*!
    This class is used for writing of user data (most usefully measurements)
    into a Berkeley DB with a key/value semantics. 
  */
  template<typename K, typename D>
  class BinaryVarStoreDB : BinaryStoreDB<K,D>
  {
  private:
    void notImplemented() const
    {
      QDPIO::cerr << "BinaryVarStoreDB: not implemented - this is a stub version. You must --enable-db-lite in qdp++" << endl;
      QDP_abort(1);
    }

  public:
    /**
     * Empty constructor for a DB
     */
    BinaryVarStoreDB () {}

    /**
     * Constructor for a DB
     *
     * @param DB file filename holding keys and data.
     */
    BinaryVarStoreDB (const std::string& file,
		      int max_cache_size = 50000000) {notImplemented();}

    /*!
      Destroy the object
    */
    ~BinaryVarStoreDB() {notImplemented();}

    /**
     * Open a DB
     *
     * @param DB file filename holding keys and data.
     */
    void open (const std::string& file,
	       int max_cache_size = 50000000) {notImplemented();}

    /*!
      Close the object
    */
    void close () {notImplemented();}

    /**
     * Insert a pair of data and key into the database
     * data is not ensemble, but a vector of complex.
     * @param key a key
     * @param data a user provided data
     */
    void insert (K& key, D& data) {notImplemented();}

    /**
     * Get data for a given key
     * @param key user supplied key
     * @param data after the call data will be populated
     * @return 0 on success, otherwise the key not found
     */
    int get (K& key, D& data) {notImplemented();}

    /**
     * Return all available keys to user
     * @param keys user suppled an empty vector which is populated
     * by keys after this call.
     */
    void keys (std::vector<K>& kys) {notImplemented();}

    /**
     * Return all pairs of keys and data
     * @param keys user supplied empty vector to hold all keys
     * @param data user supplied empty vector to hold data
     * @return keys and data in the vectors having the same size
     */
    void keysAndData (std::vector<K>& kys, std::vector<D>& vals) {notImplemented();}

    /**
     * Flush database in memory to disk
     */
    void flush (void) {notImplemented();}


    /**
     * Name of database associated with this Data store
     *
     * @return database name
     */
    std::string storageName (void) const {notImplemented();}
    
    /**
     * Insert user data into the  metadata database
     *
     * @param user_data user supplied data
     * @return returns 0 if success, else failure
     */
    int insertUserdata (const std::string& user_data) {notImplemented();}
    
    /**
     * Get user user data from the metadata database
     *
     * @param user_data user supplied buffer to store user data
     * @return returns 0 if success. Otherwise failure.
     */
    int getUserdata (std::string& user_data) {notImplemented();}
  };


  //--------------------------------------------------------------------------------
  //!  DB Base class
  /*!
    This class is used for writing of user data (most usefully measurements)
    into a Berkeley DB with a key/value semantics. 
  */
  template<typename K, typename D>
  class BinaryFxStoreDB : BinaryStoreDB<K,D>
  {
  private:
    void notImplemented() const
    {
      QDPIO::cerr << "BinaryVarStoreDB: not implemented - this is a stub version. You must --enable-db-lite in qdp++" << endl;
      QDP_abort(1);
    }

  public:
    /**
     * Empty constructor for a DB
     */
    BinaryFxStoreDB () {}

    /**
     * Constructor for a DB
     *
     * @param DB file filename holding keys and data.
     */
    BinaryFxStoreDB (const std::string& file,
		     int max_cache_size = 50000000) {notImplemented();}

    /*!
      Destroy the object
    */
    ~BinaryFxStoreDB() {notImplemented();}

    /**
     * Open a DB
     *
     * @param DB file filename holding keys and data.
     */
    void open (const std::string& file,
	       int max_cache_size = 50000000) {notImplemented();}

    /*!
      Close the object
    */
    void close () {notImplemented();}

    /**
     * Insert a pair of data and key into the database
     * data is not ensemble, but a vector of complex.
     * @param key a key
     * @param data a user provided data
     */
    void insert (K& key, D& data) {notImplemented();}

    /**
     * Get data for a given key
     * @param key user supplied key
     * @param data after the call data will be populated
     * @return 0 on success, otherwise the key not found
     */
    int get (K& key, D& data) {notImplemented();}

    /**
     * Return all available keys to user
     * @param keys user suppled an empty vector which is populated
     * by keys after this call.
     */
    void keys (std::vector<K>& kys) {notImplemented();}

    /**
     * Return all pairs of keys and data
     * @param keys user supplied empty vector to hold all keys
     * @param data user supplied empty vector to hold data
     * @return keys and data in the vectors having the same size
     */
    void keysAndData (std::vector<K>& kys, std::vector<D>& vals) {notImplemented();}

    /**
     * Flush database in memory to disk
     */
    void flush (void) {notImplemented();}

    /**
     * Name of database associated with this Data store
     *
     * @return database name
     */
    std::string storageName (void) const {notImplemented();}
    
    /**
     * Insert user data into the  metadata database
     *
     * @param user_data user supplied data
     * @return returns 0 if success, else failure
     */
    int insertUserdata (const std::string& user_data) {notImplemented();}
    
    /**
     * Get user user data from the metadata database
     *
     * @param user_data user supplied buffer to store user data
     * @return returns 0 if success. Otherwise failure.
     */
    int getUserdata (std::string& user_data) {notImplemented();}
  };


}  // namespace QDP

#endif
