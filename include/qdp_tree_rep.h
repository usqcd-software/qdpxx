// -*- C++ -*-
// $Id: qdp_tree_rep.h,v 1.1.2.1 2008-03-15 14:28:56 edwards Exp $
/*! @file
 * @brief Tree representation of data
 *
 * Support for Tree representation of data
 */

#ifndef QDP_TREE_REP_H
#define QDP_TREE_REP_H

#include <string>
#include <list>

namespace QDP 
{
  /*! @ingroup io
   * @{
   */

  //! This is a place-holder 
  class TreeRep
  {
  public:
    //! Default constructor
    TreeRep() {}
  
    //! Destructor
    virtual ~TreeRep() {}
  };


#if 0
  //--------------------------------------------------------------------------------
  //! Tree representation
  /*!
    Holds a tree representation of data
  */
  class TreeRep
  {
  public:
    //! Default constructor
    TreeRep();
  
    //! Destructor
    virtual ~TreeReader();

    /* So should these, there is just a lot of overloading */
    //! Xpath query
    virtual void get(const std::string& xpath, std::string& result);
    //! Xpath query
    virtual void get(const std::string& xpath, int& result);
    //! Xpath query
    virtual void get(const std::string& xpath, unsigned int& result);
    //! Xpath query
    virtual void get(const std::string& xpath, short int& result);
    //! Xpath query
    virtual void get(const std::string& xpath, unsigned short int& result);
    //! Xpath query
    virtual void get(const std::string& xpath, long int& result);
    //! Xpath query
    virtual void get(const std::string& xpath, unsigned long int& result);
    //! Xpath query
    virtual void get(const std::string& xpath, float& result);
    //! Xpath query
    virtual void get(const std::string& xpath, double& result);
    //! Xpath query
    virtual void get(const std::string& xpath, bool& result);

    //! Count the number of occurances from the Xpath query
    /*! THIS IS NEEDED. DUNNO YET HOW TO DO THIS. HOPEFULLY ONLY NEED SIMPLE COUNTS */
    virtual bool exist(const std::string& xpath) const;

    //! Count the number of occurances from the Xpath query
    /*! THIS IS NEEDED. DUNNO YET HOW TO DO THIS. HOPEFULLY ONLY NEED SIMPLE COUNTS */
    virtual int count(const std::string& xpath) const;

  private:
    //! Hide the = operator
    void operator=(const TreeReader&) {}

  private:
    TreeReader* tree_use; /*!< Possible pointer to another TreeReader in case this one is derived */
    bool        derivedP;  /*!< is this reader derived from another reader? */
  };

#endif

  /*! @} */   // end of group io

} // namespace QDP

#endif
