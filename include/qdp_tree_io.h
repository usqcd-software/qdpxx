// -*- C++ -*-
// $Id: qdp_tree_io.h,v 1.1.2.2 2008-03-16 02:40:04 edwards Exp $
/*! @file
 * @brief Tree IO support
 *
 * Support for parent class Tree representation
 */

#ifndef QDP_TREE_IO_H
#define QDP_TREE_IO_H

#include <string>
#include <sstream>

#include "qdp_tree_rep.h"

namespace QDP 
{

  // Forward declarations
  class TreeReader;
  class TreeWriter;
  class TreeBufferWriter;
  class TreeArrayWriter;

  template<class T> inline void write(TreeWriter& tree, const std::string& s, const multi1d<T>& s1);

  class TreeReaderImp;


  /*! @ingroup io
   * @{
   */

  //--------------------------------------------------------------------------------
  //! Tree reader class
  /*!
    This is used to read data from an Tree file using Xpath.

    Note that only the primary node opens and reads Tree files. Results from
    Xpath queries are broadcast to all nodes.
    
    This object is meant to serve as a base class. It cannot be created
    with a default constructor. However, it can be created by a constructor
    using another (concrete) TreeReader as an argument. This is meant to
    serve like a cloning function. This new TreeReader can be a subdirectory
    of the original (derived) TreeReader.
  */
  class TreeReader
  {
  public:
    //! Clone a reader but with a possibly different path
    explicit TreeReader(TreeReader& old, const std::string& xpath);
  
    //! Destructor
    virtual ~TreeReader();

    /* The meaning of these should be clear to you */

    /* So should these, there is just a lot of overloading */
    //! Xpath query
    virtual void read(const std::string& xpath, std::string& result);
    //! Xpath query
    virtual void read(const std::string& xpath, int& result);
    //! Xpath query
    virtual void read(const std::string& xpath, unsigned int& result);
    //! Xpath query
    virtual void read(const std::string& xpath, short int& result);
    //! Xpath query
    virtual void read(const std::string& xpath, unsigned short int& result);
    //! Xpath query
    virtual void read(const std::string& xpath, long int& result);
    //! Xpath query
    virtual void read(const std::string& xpath, unsigned long int& result);
    //! Xpath query
    virtual void read(const std::string& xpath, float& result);
    //! Xpath query
    virtual void read(const std::string& xpath, double& result);
    //! Xpath query
    virtual void read(const std::string& xpath, bool& result);

    /* Overloadings of primitive(elemental) array objects */
    //! Xpath query
    virtual void read(const std::string& xpath, multi1d<int>& result);
    //! Xpath query
    virtual void read(const std::string& xpath, multi1d<unsigned int>& result);
    //! Xpath query
    virtual void read(const std::string& xpath, multi1d<short int>& result);
    //! Xpath query
    virtual void read(const std::string& xpath, multi1d<unsigned short int>& result);
    //! Xpath query
    virtual void read(const std::string& xpath, multi1d<long int>& result);
    //! Xpath query
    virtual void read(const std::string& xpath, multi1d<unsigned long int>& result);
    //! Xpath query
    virtual void read(const std::string& xpath, multi1d<float>& result);
    //! Xpath query
    virtual void read(const std::string& xpath, multi1d<double>& result);
    //! Xpath query
    virtual void read(const std::string& xpath, multi1d<bool>& result);

    /* More overloadings of primitive(elemental) array objects */
    //! Xpath query
    virtual void read(const std::string& xpath, multi1d<Integer>& result);
    //! Xpath query
    virtual void read(const std::string& xpath, multi1d<Real32>& result);
    //! Xpath query
    virtual void read(const std::string& xpath, multi1d<Real64>& result);
    //! Xpath query
    virtual void read(const std::string& xpath, multi1d<Boolean>& result);

    //! Return the entire contents of the Reader as a TreeRep
    virtual void treeRep(TreeRep& output);
        
    //! Return the current context as a TreeRep
    virtual void treeRepCurrentContext(TreeRep& output);
        
    //! Does the result of this Xpath query exist
    /*! THIS IS NEEDED. PROBABLY WILL NOT SUPPORT GENERIC XPATH */
    virtual bool exist(const std::string& xpath);

    //! Count the number of occurances from the Xpath query
    /*! PROBABLY WILL NOT SUPPORT GENERIC XPATH */
//    virtual int count(const std::string& xpath);

    //! Count the array element entries
    virtual int countArrayElem();

    //! Return tag for array element n
    virtual std::string arrayElem(int n) const;

  protected:
    //! Hide default constructor
    TreeReader();
  
    //! Hide copy constructor
    TreeReader(const TreeReader&) {}
  
    //! Internal cloning function
    virtual TreeReaderImp* clone(const std::string& xpath);

    //! Register object with base class
    void registerObject(TreeReaderImp* obj);

    //! Return the implementation
    TreeReaderImp& getTreeReader() const;

  private:
    //! Hide the = operator
    void operator=(const TreeReader&) {}
  
  private:
    bool           initP;    /*!< Indicates whether this class has been initialized */
    TreeReaderImp* tree_use; /*!< Pointer to TreeReaderImp used for actual implementation */
  };


  // Time to build a telephone book of basic primitives
  //! Xpath query
  void read(TreeReader& tree, const std::string& s, std::string& input);
  //! Xpath query
  void read(TreeReader& tree, const std::string& s, int& input);
  //! Xpath query
  void read(TreeReader& tree, const std::string& s, unsigned int& input);
  //! Xpath query
  void read(TreeReader& tree, const std::string& s, short int& input);
  //! Xpath query
  void read(TreeReader& tree, const std::string& s, unsigned short int& input);
  //! Xpath query
  void read(TreeReader& tree, const std::string& s, long int& input);
  //! Xpath query
  void read(TreeReader& tree, const std::string& s, unsigned long int& input);
  //! Xpath query
  void read(TreeReader& tree, const std::string& s, float& input);
  //! Xpath query
  void read(TreeReader& tree, const std::string& s, double& input);
  //! Xpath query
  void read(TreeReader& tree, const std::string& s, bool& input);


  //! Read a multi1d object from a tree
  template<class T>
  inline
  void read(TreeReader& tree, const std::string& s, multi1d<T>& input)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    TreeReader arraytop(tree, s);
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;

    std::ostringstream error_message;
  
    int array_size;
    try 
    {
      array_size = arraytop.countArrayElem();
    }
    catch(const std::string& e) 
    { 
      error_message << "Exception occurred while counting array elements during array read of " 
		    << s << endl
		    << "Query returned error: " << e;
      throw error_message.str();
    }
      
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    // Now resize the array to hold the no of elements.
    input.resize(array_size);

    // Get the elements one by one
    for(int i=0; i < input.size(); ++i) 
    {
      // recursively try and read the element.
      try 
      {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
	read(arraytop, arraytop.arrayElem(i), input[i]);
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
      } 
      catch (const std::string& e) 
      {
	error_message << "Failed to match element " << i
		      << " of array " << s
		      << endl
		      << "Query returned error: " << e;
	throw error_message.str();
      }
    }
  }


  // Specialized versions for basic types
  template<>
  void read(TreeReader& tree, const std::string& s, multi1d<int>& input);
  template<>
  void read(TreeReader& tree, const std::string& s, multi1d<unsigned int>& input);
  template<>
  void read(TreeReader& tree, const std::string& s, multi1d<short int>& input);
  template<>
  void read(TreeReader& tree, const std::string& s, multi1d<unsigned short int>& input);
  template<>
  void read(TreeReader& tree, const std::string& s, multi1d<long int>& input);
  template<>
  void read(TreeReader& tree, const std::string& s, multi1d<unsigned long int>& input);
  template<>
  void read(TreeReader& tree, const std::string& s, multi1d<float>& input);
  template<>
  void read(TreeReader& tree, const std::string& s, multi1d<double>& input);
  template<>
  void read(TreeReader& tree, const std::string& s, multi1d<bool>& input);

  // More specialized versions for basic types
  template<>
  void read(TreeReader& tree, const std::string& s, multi1d<Integer>& input);
  template<>
  void read(TreeReader& tree, const std::string& s, multi1d<Real32>& input);
  template<>
  void read(TreeReader& tree, const std::string& s, multi1d<Real64>& input);
  template<>
  void read(TreeReader& tree, const std::string& s, multi1d<Boolean>& input);


  //--------------------------------------------------------------------------------
  //! Metadata output class
  /*!
    Use this to write Trees. When closing tags, you do not have to specify which
    tag to close since this class will remember the order in which you opened
    the tags and close them in reverse order to ensure well-formed Tree.

    Note that only the primary node writes Trees.
  */
  class TreeWriter
  {
  public:
    // Virtual destructor
    virtual ~TreeWriter();

    //! Writes an opening Tree tag    
    /*!
      \param tagname The name of the tag
    */
    virtual void openTag(const std::string& tagname) = 0;

    //! Closes a tag
    virtual void closeTag() = 0;

    //! Writes an opening Tree tag    
    /*!
      \param tagname The name of the tag
    */
    virtual void openStruct(const std::string& tagname) = 0;
    virtual void closeStruct() = 0;

    //! Return tag for array element n
    virtual std::string arrayElem(int n) const = 0;

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

    // More overloaded array (elemental list) Writer Functions
    //! Write tag and contents
    virtual void write(const std::string& tagname, const multi1d<Integer>& output) = 0;
    //! Write tag and contents
    virtual void write(const std::string& tagname, const multi1d<Real32>& output) = 0;
    //! Write tag and contents
    virtual void write(const std::string& tagname, const multi1d<Real64>& output) = 0;
    //! Write tag and contents
    virtual void write(const std::string& tagname, const multi1d<Boolean>& output) = 0;

    // Write all the Tree to std::string
    // NEEDS FIXING - WANT A TRUE TREE HERE AND NOT A STRING VERSION
//    void writeTree(const std::string& output) = 0;
  };


  //! Push a group name
  /*! Write an opening tag
    \param tree The writer
    \param s the name of the tag
  */
  void push(TreeWriter& tree, const std::string& s);

  //! Pop a group name
  /*! Write an closing tag */

  void pop(TreeWriter& tree);

  //! Write an object with the next array element name as the tag
  /*!
    \param tree The writer
    \param output The  contents
  */
  void writeElem(TreeArrayWriter& tree, const int& output);

  //! Write something from a reader
//  void write(TreeWriter& tree, const std::string& s, const TreeReader& d);
//  TreeWriter& operator<<(TreeWriter& tree, const TreeReader& d);

  //! Write something from a TreeBufferWriter
//  void write(TreeWriter& tree, const std::string& s, const TreeBufferWriter& d);
//  TreeWriter& operator<<(TreeWriter& tree, const TreeBufferWriter& d);

  //! Write a TreeRep
//  void write(TreeWriter& tree, const std::string& s, const TreeRep& d);
//  TreeWriter& operator<<(TreeWriter& tree, const TreeRep& d);

  // Time to build a telephone book of basic primitives
  //! Write a opening tag, contents and a closing tag
  /*!
    \param tree The writer
    \param s the tag name
    \param output The  contents
  */
  void write(TreeWriter& tree, const std::string& s, const std::string& output);
  //! Write a opening tag, contents and a closing tag
  /*!
    \param tree The writer
    \param s the tag name
    \param output The  contents
  */
  void write(TreeWriter& tree, const std::string& s, const char* output);
  //! Write a opening tag, contents and a closing tag
  /*!
    \param tree The writer
    \param s the tag name
    \param output The  contents
  */
  void write(TreeWriter& tree, const std::string& s, const char& output);
  //! Write a opening tag, contents and a closing tag
  /*!
    \param tree The writer
    \param s the tag name
    \param output The  contents
  */
  void write(TreeWriter& tree, const std::string& s, const int& output);
  //! Write a opening tag, contents and a closing tag
  /*!
    \param tree The writer
    \param s the tag name
    \param output The  contents
  */
  void write(TreeWriter& tree, const std::string& s, const unsigned int& output);
  //! Write a opening tag, contents and a closing tag
  /*!
    \param tree The writer
    \param s the tag name
    \param output The  contents
  */
  void write(TreeWriter& tree, const std::string& s, const short int& output);
  //! Write a opening tag, contents and a closing tag
  /*!
    \param tree The writer
    \param s the tag name
    \param output The  contents
  */
  void write(TreeWriter& tree, const std::string& s, const unsigned short int& output);
  //! Write a opening tag, contents and a closing tag
  /*!
    \param tree The writer
    \param s the tag name
    \param output The  contents
  */
  void write(TreeWriter& tree, const std::string& s, const long int& output);
  //! Write a opening tag, contents and a closing tag
  /*!
    \param tree The writer
    \param s the tag name
    \param output The  contents
  */
  void write(TreeWriter& tree, const std::string& s, const unsigned long int& output);
  //! Write a opening tag, contents and a closing tag
  /*!
    \param tree The writer
    \param s the tag name
    \param output The  contents
  */
  void write(TreeWriter& tree, const std::string& s, const float& output);
  //! Write a opening tag, contents and a closing tag
  /*!
    \param tree The writer
    \param s the tag name
    \param output The  contents
  */
  void write(TreeWriter& tree, const std::string& s, const double& output);
  //! Write a opening tag, contents and a closing tag
  /*!
    \param tree The writer
    \param s the tag name
    \param output The  contents
  */
  void write(TreeWriter& tree, const std::string& s, const bool& output);


  // Writers for arrays of basic types
  //! Writes an array of data
  /*!
    All the data are written inside a single tag pair
    \param tree The writer
    \param s the tag name
    \param s1 The array of contents
  */
  template<>
  void write(TreeWriter& tree, const std::string& s, const multi1d<int>& s1);
  //! Writes an array of data
  /*!
    All the data are written inside a single tag pair
    \param tree The writer
    \param s the tag name
    \param s1 The array of contents
  */
  template<>
  void write(TreeWriter& tree, const std::string& s, const multi1d<unsigned int>& s1);
  //! Writes an array of data
  /*!
    All the data are written inside a single tag pair
    \param tree The writer
    \param s the tag name
    \param s1 The array of contents
  */
  template<>
  void write(TreeWriter& tree, const std::string& s, const multi1d<short int>& s1);
  //! Writes an array of data
  /*!
    All the data are written inside a single tag pair
    \param tree The writer
    \param s the tag name
    \param s1 The array of contents
  */
  template<>
  void write(TreeWriter& tree, const std::string& s, const multi1d<unsigned short int>& s1);
  //! Writes an array of data
  /*!
    All the data are written inside a single tag pair
    \param tree The writer
    \param s the tag name
    \param s1 The array of contents
  */
  template<>
  void write(TreeWriter& tree, const std::string& s, const multi1d<long int>& s1);
  //! Writes an array of data
  /*!
    All the data are written inside a single tag pair
    \param tree The writer
    \param s the tag name
    \param s1 The array of contents
  */
  template<>
  void write(TreeWriter& tree, const std::string& s, const multi1d<unsigned long int>& s1);
  //! Writes an array of data
  /*!
    All the data are written inside a single tag pair
    \param tree The writer
    \param s the tag name
    \param s1 The array of contents
  */
  template<>
  void write(TreeWriter& tree, const std::string& s, const multi1d<float>& s1);
  //! Writes an array of data
  /*!
    All the data are written inside a single tag pair
    \param tree The writer
    \param s the tag name
    \param s1 The array of contents
  */
  template<>
  void write(TreeWriter& tree, const std::string& s, const multi1d<double>& s1);
  //! Writes an array of data
  /*!
    All the data are written inside a single tag pair
    \param tree The writer
    \param s the tag name
    \param s1 The array of contents
  */
  template<>
  void write(TreeWriter& tree, const std::string& s, const multi1d<bool>& s1);

  //! Write an expression
  template<class RHS, class C>
  void write(TreeWriter& tree, const std::string& s, const QDPExpr<RHS,C>& d)
  {
    return write(tree, s, C(d));
  }

#if 0
  // NEED TO FIX THIS
  //! Write a Tree multi2d element
  template<class T> 
  inline
  void write(TreeWriter& tree, const std::string& s, const multi2d<T>& s1)
  {
    for(int j=0; j < s1.size1(); ++j)
      for(int i=0; i < s1.size2(); ++i)
      {
	std::ostringstream ost;
	if (Layout::primaryNode()) 
	  ost << s << "[ " << i << " ][ " << j << " ]";
	write(tree, ost.str(), s1[i][j]);
      }
  }

#endif

  // Tree writers
  //! Writes an array of data
  /*!
    All the data are written inside a single tag pair
    \param tree The writer
    \param s the tag name
    \param s1 The array of contents
  */
  template<>
  void write(TreeWriter& tree, const string& s, const multi1d<Integer>& s1);
  /*!
    All the data are written inside a single tag pair
    \param tree The writer
    \param s the tag name
    \param s1 The array of contents
  */
  template<>
  void write(TreeWriter& tree, const string& s, const multi1d<Real32>& s1);
  /*!
    All the data are written inside a single tag pair
    \param tree The writer
    \param s the tag name
    \param s1 The array of contents
  */
  template<>
  void write(TreeWriter& tree, const string& s, const multi1d<Real64>& s1);
  /*!
    All the data are written inside a single tag pair
    \param tree The writer
    \param s the tag name
    \param s1 The array of contents
  */
  template<>
  void write(TreeWriter& tree, const string& s, const multi1d<Boolean>& s1);




#if 0
  class TreeBufferWriter : public TreeWriter
  {
  public:
  };

#else
  //--------------------------------------------------------------------------------
  //! Writes Tree metadata to a buffer
  /*! 
    This class provides an internal tree representation of the data
    that is held in memory. It is useful for working on snippets
    of a bigger tree, and then serializing this bit of a tree out 
    as part of a bigger tree.
  */
  class TreeBufferWriter : public TreeWriter
  {
  public:
    /*! No prologue written */
    TreeBufferWriter();
  
    //! Construct from a TreeRep
    explicit TreeBufferWriter(const TreeRep& s);

    //! Destroy
    ~TreeBufferWriter();

    //! Construct from a TreeRep
    void open(const TreeRep& s);

    //! Flush the output. Maybe a nop.
    void flush();

    //! Return entire buffer as a TreeRep
    /*! SEE IF WE CAN JUST WRITE IN PLACE RATHER THAN RETURNING BIG THING */
//    std::string str() const;
//    TreeRep treeRep() const;
    void treeRep(TreeRep&) const;
        
    // Return root element as a TreeRep
//    std::string printCurrentContext() const;
//    TreeRep treeRepCurrentContext() const;
    void treeRepCurrentContext(TreeRep&) const;
        
    //! Open an array element struct
    void openStruct(const std::string& tagname);
    //! Close an array element struct
    void closeStruct();

    // These are all forwardings to the underlying TreeWriter

    //! Writes an opening Tree tag    
    /*!
      \param tagname The name of the tag
    */
    void openTag(const std::string& tagname);

    //! Closes a tag
    void closeTag();

    //! Return tag for array element n
    virtual std::string arrayElem(int n) const;

    // Overloaded Writer Functions
    //! Write tag and contents
    void write(const std::string& tagname, const std::string& output);
    //! Write tag and contents
    void write(const std::string& tagname, const int& output);
    //! Write tag contents
    void write(const std::string& tagname, const unsigned int& output);
    //! Write tag and contents
    void write(const std::string& tagname, const short int& output);
    //! Write tag and contents
    void write(const std::string& tagname, const unsigned short int& output);
    //! Write tag and contents
    void write(const std::string& tagname, const long int& output);
    //! Write tag and contents
    void write(const std::string& tagname, const unsigned long int& output);
    //! Write tag and contents
    void write(const std::string& tagname, const float& output);
    //! Write tag and contents
    void write(const std::string& tagname, const double& output);
    //! Write tag and contents
    void write(const std::string& tagname, const bool& output);

    // Overloaded array (elemental list) Writer Functions
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<int>& output);
    //! Write tag contents
    void write(const std::string& tagname, const multi1d<unsigned int>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<short int>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<unsigned short int>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<long int>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<unsigned long int>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<float>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<double>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<bool>& output);

    // More overloaded array (elemental list) Writer Functions
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<Integer>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<Real32>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<Real64>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<Boolean>& output);

  protected:
    //! Return the implementation
    TreeWriter& getTreeWriter() const;

  private:
    // The output TreeRep
    TreeRep tree_rep;

    // The function that supplies the TreeRep
    const TreeRep& getTreeRep() const {return tree_rep;}
  };
#endif


  //--------------------------------------------------------------------------------
  //! Writes metadata to an array which serves as a handle for another Tree object
  class TreeArrayWriter : public TreeWriter
  {
  public:
    /*! No prologue written
     * @param tree_out  previous TreeWriter object - used for handle source
     * @param size      size of array - default unbounded
     */
    explicit TreeArrayWriter(TreeWriter& tree_out, int size = -1);

    //! Destructor
    ~TreeArrayWriter();

    //! Flush the output. Maybe a nop.
    void flush();

    //! Closes the array writer
    /*! It is an error to close before all data is written, unless unbounded */
    void close();
       
    //! Returns the number of array elements written
    int size() const;

    //! Open the array
    void openArray(const std::string& tagname);
    //! Close the array
    void closeArray();

    //! Open an array element struct
    void openStruct(const std::string& tagname);
    //! Close an array element struct
    void closeStruct();

    //! Get next array element name
    std::string nextElem();

    // These are all forwardings to the underlying TreeWriter

    //! Writes an opening Tree tag    
    /*!
      \param tagname The name of the tag
    */
    void openTag(const std::string& tagname);

    //! Closes a tag
    void closeTag();

    //! Return tag for array element n
    virtual std::string arrayElem(int n) const;

    // Overloaded Writer Functions
    //! Write tag and contents
    void write(const std::string& tagname, const std::string& output);
    //! Write tag and contents
    void write(const std::string& tagname, const int& output);
    //! Write tag contents
    void write(const std::string& tagname, const unsigned int& output);
    //! Write tag and contents
    void write(const std::string& tagname, const short int& output);
    //! Write tag and contents
    void write(const std::string& tagname, const unsigned short int& output);
    //! Write tag and contents
    void write(const std::string& tagname, const long int& output);
    //! Write tag and contents
    void write(const std::string& tagname, const unsigned long int& output);
    //! Write tag and contents
    void write(const std::string& tagname, const float& output);
    //! Write tag and contents
    void write(const std::string& tagname, const double& output);
    //! Write tag and contents
    void write(const std::string& tagname, const bool& output);

    // Overloaded array (elemental list) Writer Functions
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<int>& output);
    //! Write tag contents
    void write(const std::string& tagname, const multi1d<unsigned int>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<short int>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<unsigned short int>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<long int>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<unsigned long int>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<float>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<double>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<bool>& output);

    // More overloaded array (elemental list) Writer Functions
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<Integer>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<Real32>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<Real64>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<Boolean>& output);

  private:
    bool usedP;             /*!< set if the array is ever opened */
    bool initP;             /*!< set once array tag is written */
    int  open_elem;         /*!< counts number of open push-es */
    int  array_size;        /*!< total array element size */
    int  elements_written;  /*!< elements written so far */

    // output stream is actually the original stream
    TreeWriter& output_tree; 

    TreeWriter& getTreeWriter() const {return output_tree;}
  };


  //! Push an array group name
  /*! Write an opening tag
    \param tree The writer
    \param s the name of the tag
  */
  void push(TreeArrayWriter& tree, const std::string& s);

  //! Pop an array group name
  /*! Write an closing tag */
  void pop(TreeArrayWriter& tree);

  //! Push an element group name
  void pushElem(TreeArrayWriter& tree);

  //! Pop an element group name
  void popElem(TreeArrayWriter& tree);


  //! Write a opening tag, array contents and a closing tag
  /*!
    Each element of the array is written in a "elem" tag.
    \param tree The writer
    \param s the tag name
    \param s1 The array of contents
  */
  template<class T>
  inline
  void write(TreeWriter& tree, const std::string& s, const multi1d<T>& s1)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    // Open an array writer for this object
    TreeArrayWriter tree_array(tree);

    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    // Start the array
    push(tree_array, s);

    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    for(int index=0; index < s1.size(); ++index)
    {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
      try 
      {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
	write(tree_array, tree_array.nextElem(), s1[index]);  // Possibly grab user defines here
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
      } 
      catch (const std::string& e) 
      {
	std::ostringstream error_message;
  	error_message << "Error writing element " << index
		      << " of array " << s
		      << endl
		      << "Error returned: " << e;
	throw error_message.str();
      }
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    }

    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    pop(tree_array); // Close the array
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
  }


  /*! @} */   // end of group io

} // namespace QDP

#endif
