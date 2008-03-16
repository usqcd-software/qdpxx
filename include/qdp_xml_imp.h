// -*- C++ -*-
// $Id: qdp_xml_imp.h,v 1.1.2.2 2008-03-16 16:07:20 edwards Exp $
/*! @file
 * @brief XML IO support via trees
 */

#ifndef QDP_XML_IMP_H
#define QDP_XML_IMP_H

#include <string>
#include <sstream>

#include "qdp_tree_imp.h"

#include "xml_simplewriter.h"
#include "basic_xpath_reader.h"

namespace QDP 
{

  // Forward declarations
  class XMLBufferWriter;


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
  class XMLReaderImp : public TreeReaderImp, protected XMLXPathReader::BasicXPathReader
  {
  public:
    //! Empty constructor
    XMLReaderImp();

    //! Construct from contents of file
    /*!
      Opens and reads an XML file.
      \param filename The name of the file.
    */
    explicit XMLReaderImp(const std::string& filename);

    //! Construct from contents of stream
    explicit XMLReaderImp(std::istream& is);

    //! Construct from contents of a XMLBufferWriter
    explicit XMLReaderImp(const XMLBufferWriter& mw);

    //! Clone a reader but with a possibly different path
    explicit XMLReaderImp(XMLReaderImp& old, const std::string& xpath);

    //! Destructor
    ~XMLReaderImp();

    //! Clone a reader from a different path
    XMLReaderImp* clone(const std::string& xpath);

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

    //! Clone a reader
    void open(XMLReaderImp& old, const std::string& xpath);

    //! Queries whether the binary file is open
    /*!
      \return true if the binary file is open; false otherwise.
    */
    bool is_open();

    //! Queries whether the XML data has been obtained from another XMLReaderImp
    /*!
      A private method allows this XMLReaderImp to be copy the contents of
      another.
    */
    bool is_derived() const;

    //! Closes the last file opened
    void close();
    
    /* So should these, there is just a lot of overloading */
    //! Xpath query
    void read(const std::string& xpath, std::string& result);
    //! Xpath query
    void read(const std::string& xpath, int& result);
    //! Xpath query
    void read(const std::string& xpath, unsigned int& result);
    //! Xpath query
    void read(const std::string& xpath, short int& result);
    //! Xpath query
    void read(const std::string& xpath, unsigned short int& result);
    //! Xpath query
    void read(const std::string& xpath, long int& result);
    //! Xpath query
    void read(const std::string& xpath, unsigned long int& result);
    //! Xpath query
    void read(const std::string& xpath, float& result);
    //! Xpath query
    void read(const std::string& xpath, double& result);
    //! Xpath query
    void read(const std::string& xpath, bool& result);

    /* Overloadings of primitive(elemental) array objects */
    //! Xpath query
    void read(const std::string& xpath, multi1d<int>& result);
    //! Xpath query
    void read(const std::string& xpath, multi1d<unsigned int>& result);
    //! Xpath query
    void read(const std::string& xpath, multi1d<short int>& result);
    //! Xpath query
    void read(const std::string& xpath, multi1d<unsigned short int>& result);
    //! Xpath query
    void read(const std::string& xpath, multi1d<long int>& result);
    //! Xpath query
    void read(const std::string& xpath, multi1d<unsigned long int>& result);
    //! Xpath query
    void read(const std::string& xpath, multi1d<float>& result);
    //! Xpath query
    void read(const std::string& xpath, multi1d<double>& result);
    //! Xpath query
    void read(const std::string& xpath, multi1d<bool>& result);

    //! Set a replacement of a primitive
    template<typename T>
    void set(const std::string& xpath, const T& to_set) 
      {
	if (Layout::primaryNode())
	{  
	  BasicXPathReader::set<T>(xpath, to_set);
	}
      }


    //! Return the entire contents of the Reader as a TreeRep
    void treeRep(TreeRep& output);
        
    //! Return the current context as a TreeRep
    void treeRepCurrentContext(TreeRep& output);
        
    //! Count the number of occurances from the Xpath query
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

  private:
    //! Hide the = operator
    void operator=(const XMLReaderImp&) {}
  
    //! Hide the copy constructor
    XMLReaderImp(const XMLReaderImp&) {}
  
  protected:
    // The universal data-reader. All the read functions call this
    template<typename T>
    void
    readPrimitive(const std::string& xpath, T& output);
  
    //! The needed tree reader
//    virtual TreeReaderImp& getTreeReader() const = 0;

  private:
    bool  iop;  //file open or closed?
    bool  derived; // is this reader derived from another reader?
  };


  //--------------------------------------------------------------------------------
  //! Metadata output class
  /*!
    Use this to write XML.When closing tags, you do not have to specify which
    tag to close since this class will remember the order in which you opened
    the tags and close them in reverse order to ensure well-formed XML.

    Note that only the primary node writes XML.
  */
  class XMLWriterImp : public TreeWriterImp, protected XMLWriterAPI::XMLSimpleWriter
  {
  public:
    //! Default constructor
    XMLWriterImp();

    // Virtual destructor
    virtual ~XMLWriterImp();

    //! Writes an opening XML tag
    /*!
      \param tagname The name of the tag
    */
    virtual void openSimple(const std::string& tagname);
    virtual void closeSimple();

    //! Writes an opening XML tag    
    /*!
      \param tagname The name of the tag
    */
    virtual void openStruct(const std::string& tagname);
    virtual void closeStruct();

    //! Writes an opening XML tag    
    /*!
      \param tagname The name of the tag
    */
    void openTag(const std::string& tagname);

    //! Writes an opening XML tag    
    /*!
      \param nsprefix A namespace prefix for the tag 
      \param tagname The name of the tag
    */
    void openTag(const std::string& nsprefix, const std::string& tagname);

    //! Writes an opening XML tag    
    /*!
      \param tagname The name of the tag
      \param al A list of attributes for this tag
    */
    void openTag(const std::string& tagname, XMLWriterAPI::AttributeList& al);

    //! Writes an opening XML tag    
    /*!
      \param nsprefix A namespace prefix for the tag 
      \param tagname The name of the tag
      \param al A list of attributes for this tag      
    */
    void openTag(const std::string& nsprefix,
		 const std::string& tagname, 
		 XMLWriterAPI::AttributeList& al);

    //! Closes a tag
    void closeTag();

    //! Writes an empty tag
    /*!
      \param tagname The name of the tag
    */
    void emptyTag(const std::string& tagname);

    //! Writes an empty tag
    /*!
      \param nsprefix A namespace prefix for the tag 
      \param tagname The name of the tag
    */
    void emptyTag(const std::string& nsprefix, const std::string& tagname);

    //! Writes an empty tag
    /*!
      \param tagname The name of the tag
      \param al A list of attributes for this tag            
    */
    void emptyTag(const std::string& tagname, XMLWriterAPI::AttributeList& al);

    //! Writes an empty tag
    /*!
      \param nsprefix A namespace prefix for the tag 
      \param tagname The name of the tag
      \param al A list of attributes for this tag            
    */
    void emptyTag(const std::string& nsprefix,
		  const std::string& tagname, 
		  XMLWriterAPI::AttributeList& al);
    

    //! Return tag for array element n
    std::string arrayElem(int n) const;

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

    // Write all the XML to std::string
    void writeXML(const std::string& output);

  protected:
    // The universal data-writer. All the read functions call this
    template<typename T>
    void writePrimitive(const std::string& tagname, const T& output);

    // Write an array of basic types
    template<typename T>
    void writeArrayPrimitive(const std::string& tagname, const multi1d<T>& output);

  protected:
    //! Get the internal ostream
    virtual std::ostream& getOstream() = 0;
  };


  //--------------------------------------------------------------------------------
  //! Writes XML metadata to a buffer
  class XMLBufferWriterImp : public XMLWriterImp
  {
  public:
    /*! No prologue written */
    XMLBufferWriterImp();
  
    //! Construct from a string
    explicit XMLBufferWriterImp(const std::string& s);

    //! Destroy
    ~XMLBufferWriterImp();

    //! Construct from a string
    void open(const std::string& s);

    //! Return entire buffer as a string
    std::string str() const;
        
    // Return root element as a string
    std::string printCurrentContext() const;
        
    //! Flush the buffer
    void flush() {}

    //! Return true if some failure occurred in previous IO operation
    bool fail() const {return false;}

  private:
    // The output stream...
    ostringstream output_stream;

    // The function that supplies the stream to the parent...
    ostream& getOstream(void) {return output_stream;}
  };


  //--------------------------------------------------------------------------------
  //! Writes XML metadata to a file
  /*!
    \ingroup io
  */

  class XMLFileWriterImp : public XMLWriterImp
  {
  public:
    //! Default constructor
    XMLFileWriterImp();

    //! Constructor from a filename
    /*!
      \param filename The name of the file
      \param write_prologue Whether to write the standard opening line of
      XML files. Defaults to true.
    */
    explicit XMLFileWriterImp(const std::string& filename, bool write_prologue=true);

    //! Destructor
    ~XMLFileWriterImp();

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

    //! Flush the buffer
    void flush();

    //! Return true if some failure occurred in previous IO operation
    bool fail() const;

    //! Closes the last  file  opened.
    void close();
        
  private:
    ofstream output_stream;
    ostream& getOstream(void) {return output_stream;}
  };


  /*! @} */   // end of group io

} // namespace QDP

#endif
