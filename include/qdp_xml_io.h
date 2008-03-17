// -*- C++ -*-
// $Id: qdp_xml_io.h,v 1.1.2.4 2008-03-17 03:55:36 edwards Exp $
/*! @file
 * @brief XML IO support via trees
 */

#ifndef QDP_XML_IO_H
#define QDP_XML_IO_H

#include <string>
#include <sstream>

#include "qdp_tree_io.h"

namespace QDP 
{

  // Forward declarations
  class XMLBufferWriter;

  class XMLReaderImp;
  class XMLWriterImp;
  class XMLBufferWriterImp;
  class XMLFileWriterImp;

  /*! @ingroup io
   * @{
   */

  //--------------------------------------------------------------------------------
  //! XML reader class
  /*!
    This is used to read data from an XML file using Xpath.

    Note that only the primary node opens and reads XML files. Results from
    Xpath queries are broadcast to all nodes.
  */
  class XMLReader : public TreeReader
  {
  public:
    //! Empty constructor
    XMLReader();

    //! Construct from contents of file
    /*!
      Opens and reads an XML file.
      \param filename The name of the file.
    */
    XMLReader(const std::string& filename);

    //! Construct from contents of stream
    XMLReader(std::istream& is);

    //! Construct from contents of a XMLBufferWriter
    XMLReader(const XMLBufferWriter& mw);

    //! Clone a reader but with a possibly different path
    XMLReader(XMLReader& old, const std::string& xpath);

    //! Destructor
    ~XMLReader();

    /* The meaning of these should be clear to you */

    //! Opens and reads an XML file.
    /*!
      \param filename The name of the file
      \post Any previously opened file is closed.
    */
    void open(const std::string& filename);

    //! Opens and reads an XML file.
    /*!
      \param id The input stream of the file
      \post Any previously opened file is closed      
    */
    void open(std::istream& is);

    //! Reads content of a  XMLBufferWriter
    void open(const XMLBufferWriter& mw);

    //! Queries whether the binary file is open
    /*!
      \return true if the binary file is open; false otherwise.
    */
    bool is_open();

    //! Closes the last file opened
    void close();
    
    //! Set a replacement of a primitive
//    template<typename T>
//    void set(const std::string& xpath, const T& to_set) 
//      {
//	if (Layout::primaryNode())
//      {  
//	  BasicXPathReader::set<T>(xpath, to_set);
//	}
//      }

    //! Return the entire contents of the Reader as a TreeRep
    void treeRep(TreeRep& output);
        
    //! Return the current context as a TreeRep
    void treeRepCurrentContext(TreeRep& output);
        
    //! Does the result of this Xpath query exist?
    bool exist(const std::string& xpath);

    //! Count the number of occurances from the Xpath query
    int count(const std::string& xpath);

    //! Count the array element entries
    int countArrayElem();

    //! Return tag for array element n
    std::string arrayElem(int n) const;

    //! Return the entire contents of the Reader as a stream
    void print(ostream& is);
        
    //! Print the current context
    void printCurrentContext(ostream& is);
        
    //! Register a namespace. [Why is this needed?]
    void registerNamespace(const std::string& prefix, const std::string& uri);

  protected:
    //! Internal cloning function
    TreeReaderImp* clone(const std::string& xpath);

    //! Return the implementation
    XMLReaderImp& getXMLReader() const;

  private:
    //! Hide the = operator
    void operator=(const XMLReader&) {}
  
    //! Hide the copy constructor
    XMLReader(const XMLReader&) {}
  
    //! Clone a reader
    void open(XMLReader& old, const std::string& xpath);
  };


  //--------------------------------------------------------------------------------
  //! Metadata output class
  /*!
    Use this to write XML.When closing tags, you do not have to specify which
    tag to close since this class will remember the order in which you opened
    the tags and close them in reverse order to ensure well-formed XML.

    Note that only the primary node writes XML.
  */
  class XMLWriter : public TreeWriter
  {
  public:
    // Virtual destructor
    virtual ~XMLWriter();

    //! Writes an empty tag
    /*!
      \param tagname The name of the tag
    */
    void emptyTag(const std::string& tagname);

    // Write all the XML to std::string
    void writeXML(const std::string& output);
 
  protected:
    //! Hide the default constructor
    XMLWriter();

    //! Return the implementation
    XMLWriterImp& getXMLWriter() const;

  private:
    //! Hide the copy constructor
    XMLWriter(const XMLWriter&) {}

    //! Hide the = operator
    void operator=(const XMLWriter&) {}
  };


  //! Write something from a reader
//  void write(TreeWriter& xml, const std::string& s, const XMLReader& d);
  XMLWriter& operator<<(XMLWriter& xml, const XMLReader& d);
 
  //! Write something from a XMLBufferWriter
//  void write(TreeWriter& xml, const std::string& s, const XMLBufferWriter& d);
  XMLWriter& operator<<(XMLWriter& xml, const XMLBufferWriter& d);


#if 0
  // NEED TO FIX THIS
  //! Write a XML multi2d element
  template<class T> 
  inline
  void write(XMLWriter& xml, const std::string& s, const multi2d<T>& s1)
  {
    for(int j=0; j < s1.size1(); ++j)
      for(int i=0; i < s1.size2(); ++i)
      {
	std::ostringstream ost;
	if (Layout::primaryNode()) 
	  ost << s << "[ " << i << " ][ " << j << " ]";
	write(xml, ost.str(), s1[i][j]);
      }
  }

#endif



  //--------------------------------------------------------------------------------
  //! Writes XML metadata to a buffer
  class XMLBufferWriter : public XMLWriter
  {
  public:
    /*! No prologue written */
    XMLBufferWriter();
  
    //! Construct from a string
    explicit XMLBufferWriter(const std::string& s);

    //! Destroy
    ~XMLBufferWriter();

    //! Construct from a string
    void open(const std::string& s);

    //! Return entire buffer as a string
    std::string str() const;
        
    // Return root element as a string
    std::string printCurrentContext() const;
        
    //! Return true if some failure occurred in previous IO operation
    bool fail() const {return false;}

  private:
    // Hide copy constructors
    XMLBufferWriter(const XMLBufferWriter&) {}

    // Hide copy constructors
    void operator=(const XMLBufferWriter&) {}
  };


  //--------------------------------------------------------------------------------
  //! Writes XML metadata to a file
  /*!
    \ingroup io
  */
  class XMLFileWriter : public XMLWriter
  {
  public:
    //! Default constructor
    XMLFileWriter();

    //! Constructor from a filename
    /*!
      \param filename The name of the file
      \param write_prologue Whether to write the standard opening line of
      XML files. Defaults to true.
    */
    explicit XMLFileWriter(const std::string& filename, bool write_prologue=true);

    //! Destructor
    ~XMLFileWriter();

    //! Queries whether the binary file is open
    /*!
      \return true if the binary file is open; false otherwise.
    */
    bool is_open();

    //!Opens a file
    /*!
      \param filename The name of the file
      \param write_prologue Whether to write the standard opening line of
      XML files. Defaults to true.
    */
    void open(const std::string& filename, bool write_prologue=true);

    //! Return true if some failure occurred in previous IO operation
    bool fail() const;

    //! Closes the last  file  opened.
    void close();
       
  private:
    // Hide copy constructors
    XMLFileWriter(const XMLFileWriter&) {}

    // Hide copy constructors
    void operator=(const XMLFileWriter&) {}
  };


  /*! @} */   // end of group io

} // namespace QDP

#endif
