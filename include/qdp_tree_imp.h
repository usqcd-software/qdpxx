// -*- C++ -*-
// $Id: qdp_tree_imp.h,v 1.1.2.3 2008-03-17 03:55:36 edwards Exp $
/*! @file
 * @brief Tree IO support
 *
 * Support for parent class Tree representation 
 */

#ifndef QDP_TREE_IMP_H
#define QDP_TREE_IMP_H

#include <string>

#include "qdp_tree_types.h"
#include "qdp_tree_rep.h"

namespace QDP 
{

  // Forward declarations
  class TreeReaderImp;
  class TreeWriterImp;

  /*! @ingroup io
   * @{
   */

  //--------------------------------------------------------------------------------
  //! Tree reader class
  /*!
    This is used to read data from a Tree file using Xpath.

    Note that only the primary node opens and reads Tree files. Results from
    Xpath queries are broadcast to all nodes.
    
    This object is meant to serve as a base class. It cannot be created
    with a default constructor. However, it can be created by a constructor
    using another (concrete) TreeReader as an argument. This is meant to
    serve like a cloning function. This new TreeReader can be a subdirectory
    of the original (derived) TreeReader.
  */
  class TreeReaderImp
  {
  public:
    //! Destructor
    virtual ~TreeReaderImp() {}

    //! Clone a reader from a different path
    virtual TreeReaderImp* clone(const std::string& xpath) = 0;

    /* The meaning of these should be clear to you */
    //! Xpath query
    virtual void read(const std::string& xpath, std::string& result) = 0;
    //! Xpath query
    virtual void read(const std::string& xpath, int& result) = 0;
    //! Xpath query
    virtual void read(const std::string& xpath, unsigned int& result) = 0;
    //! Xpath query
    virtual void read(const std::string& xpath, short int& result) = 0;
    //! Xpath query
    virtual void read(const std::string& xpath, unsigned short int& result) = 0;
    //! Xpath query
    virtual void read(const std::string& xpath, long int& result) = 0;
    //! Xpath query
    virtual void read(const std::string& xpath, unsigned long int& result) = 0;
    //! Xpath query
    virtual void read(const std::string& xpath, float& result) = 0;
    //! Xpath query
    virtual void read(const std::string& xpath, double& result) = 0;
    //! Xpath query
    virtual void read(const std::string& xpath, bool& result) = 0;

    /* Overloadings of primitive(elemental) array objects */
    //! Xpath query
    virtual void read(const std::string& xpath, multi1d<int>& result) = 0;
    //! Xpath query
    virtual void read(const std::string& xpath, multi1d<unsigned int>& result) = 0;
    //! Xpath query
    virtual void read(const std::string& xpath, multi1d<short int>& result) = 0;
    //! Xpath query
    virtual void read(const std::string& xpath, multi1d<unsigned short int>& result) = 0;
    //! Xpath query
    virtual void read(const std::string& xpath, multi1d<long int>& result) = 0;
    //! Xpath query
    virtual void read(const std::string& xpath, multi1d<unsigned long int>& result) = 0;
    //! Xpath query
    virtual void read(const std::string& xpath, multi1d<float>& result) = 0;
    //! Xpath query
    virtual void read(const std::string& xpath, multi1d<double>& result) = 0;
    //! Xpath query
    virtual void read(const std::string& xpath, multi1d<bool>& result) = 0;

    //! Return the entire contents of the Reader as a TreeRep
    virtual void treeRep(TreeRep& output) = 0;
        
    //! Return the current context as a TreeRep
    virtual void treeRepCurrentContext(TreeRep& output) = 0;
        
    //! Count the number of occurances from the Xpath query
    /*! THIS IS NEEDED. PROBABLY WILL NOT SUPPORT GENERIC XPATH */
    virtual bool exist(const std::string& xpath) = 0;

    //! Count the array element entries
    virtual int countArrayElem() = 0;

    //! Return tag for array element n
    virtual std::string arrayElem(int n) const = 0;

  protected:
    //! Hide default constructor
    TreeReaderImp() {}
  
  private:
    //! Hide the = operator
    void operator=(const TreeReaderImp&) {}
  
    //! The needed tree reader
//    virtual TreeReaderImp& getTreeReader() const = 0;
  };


  //--------------------------------------------------------------------------------
  //! Metadata output class
  /*!
    Use this to write Trees. When closing tags, you do not have to specify which
    tag to close since this class will remember the order in which you opened
    the tags and close them in reverse order to ensure well-formed Tree.

    Note that only the primary node writes Trees.
  */
  class TreeWriterImp
  {
  public:
    //! Virtual destructor
    virtual ~TreeWriterImp() {}

    //! Writes an opening Tree tag    
    /*!
      \param tagname The name of the tag
    */
    virtual void openStruct(const std::string& tagname) = 0;
    virtual void closeStruct() = 0;

    //! Writes an opening Tree tag    
    /*!
      \param tagname The name of the tag
    */
    virtual void openTag(const std::string& tagname) = 0;

    //! Closes a tag
    virtual void closeTag() = 0;

    //! Return tag for array element n
    virtual std::string arrayElem(int n) const = 0;

    //! Write the number of array elements written
    virtual void writeArraySize(int size) = 0;

    //! Flush the output. Maybe a nop.
    virtual void flush() = 0;

    // Overloaded Writer Functions
    //! Write tag and contents
    virtual void write(const std::string& tagname, const std::string& output) = 0;
    //! Write tag and contents
    virtual void write(const std::string& tagname, const int& output) = 0;
    //! Write tag contents
    virtual void write(const std::string& tagname, const unsigned int& output) = 0;
    //! Write tag and contents
    virtual void write(const std::string& tagname, const short int& output) = 0;
    //! Write tag and contents
    virtual void write(const std::string& tagname, const unsigned short int& output) = 0;
    //! Write tag and contents
    virtual void write(const std::string& tagname, const long int& output) = 0;
    //! Write tag and contents
    virtual void write(const std::string& tagname, const unsigned long int& output) = 0;
    //! Write tag and contents
    virtual void write(const std::string& tagname, const float& output) = 0;
    //! Write tag and contents
    virtual void write(const std::string& tagname, const double& output) = 0;
    //! Write tag and contents
    virtual void write(const std::string& tagname, const bool& output) = 0;

    // Overloaded array (elemental list) Writer Functions
    //! Write tag and contents
    virtual void write(const std::string& tagname, const multi1d<int>& output) = 0;
    //! Write tag contents
    virtual void write(const std::string& tagname, const multi1d<unsigned int>& output) = 0;
    //! Write tag and contents
    virtual void write(const std::string& tagname, const multi1d<short int>& output) = 0;
    //! Write tag and contents
    virtual void write(const std::string& tagname, const multi1d<unsigned short int>& output) = 0;
    //! Write tag and contents
    virtual void write(const std::string& tagname, const multi1d<long int>& output) = 0;
    //! Write tag and contents
    virtual void write(const std::string& tagname, const multi1d<unsigned long int>& output) = 0;
    //! Write tag and contents
    virtual void write(const std::string& tagname, const multi1d<float>& output) = 0;
    //! Write tag and contents
    virtual void write(const std::string& tagname, const multi1d<double>& output) = 0;
    //! Write tag and contents
    virtual void write(const std::string& tagname, const multi1d<bool>& output) = 0;

    // Write all the Tree to std::string
    // NEEDS FIXING - WANT A TRUE TREE HERE AND NOT A STRING VERSION
//    void writeTree(const std::string& output) = 0;

  protected:
    //! Hide default constructor
    TreeWriterImp() {}
  
  private:
    //! Hide the = operator
    void operator=(const TreeWriterImp&) {}
  };

  /*! @} */   // end of group io

} // namespace QDP

#endif
