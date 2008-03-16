// $Id: qdp_xml_io.cc,v 1.1.2.2 2008-03-16 16:07:21 edwards Exp $
//
/*! @file
 * @brief XML IO support
 */

#include "qdp.h"
#include "qdp_xml_io.h"
#include "qdp_xml_imp.h"

namespace QDP 
{
  using std::string;

  //--------------------------------------------------------------------------------
  // Just for a test to make sure a final object is complete.
  // This is not to run, but just compile
  // Anonymous namespace
  namespace
  {
    void test_xmlreader()
    {
      XMLReader xml("fred");

      int a;
      read(xml, "/a", a);

      XMLReader xml_sub(xml, "/a");
      read(xml, "b", a);

      multi1d<Real> bar;
      read(xml, "/bar", bar);
    }

    void test_xmlbufferwriter()
    {
      XMLBufferWriter xml;

      int a = 0;
      multi1d<Real> bar(5);
      bar = zero;

      push(xml, "root");
      write(xml, "a", a);
      write(xml, "bar", bar);
      pop(xml);
    }

    void test_xmlfilewriter()
    {
      XMLFileWriter xml("foo");

      int a = 0;
      multi1d<Real> bar(5);
      bar = zero;

      push(xml, "root");
      write(xml, "a", a);
      write(xml, "bar", bar);
      pop(xml);
    }
 
    void test_sub_reader()
    {
      XMLReader xml("fred");

      int a;

      TreeReader tree(xml, "/a");
      read(tree, "a", a);

      multi1d<Real> bar;
      read(tree, "/bar", bar);
    }

    void test_treearraywriter()
    {
      XMLFileWriter xml("foo");
      TreeArrayWriter tree_array(xml);

      int a = 0;
      multi1d<Real> bar(5);
      bar = zero;

      push(tree_array, "root");
      for(int i=0; i < bar.size(); ++i)
      {
	pushElem(tree_array);
	write(tree_array, "bar",bar[0]);
	popElem(tree_array);
      }
      pop(tree_array);
    }
 
  }
  


  //--------------------------------------------------------------------------------
  // XML classes
  // XML reader class
  XMLReader::XMLReader()
  {
    registerObject(new XMLReaderImp());
  }

  XMLReader::XMLReader(const std::string& filename)
  {
    registerObject(new XMLReaderImp(filename));
  }

  XMLReader::XMLReader(std::istream& is)
  {
    registerObject(new XMLReaderImp(is));
  }

  XMLReader::XMLReader(const XMLBufferWriter& mw)
  {
    registerObject(new XMLReaderImp(mw));
  }

  XMLReader::XMLReader(XMLReader& old, const string& xpath)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    registerObject(old.clone(xpath));
  }

  XMLReader::~XMLReader() {}

  // Return the implementation
  XMLReaderImp& XMLReader::getXMLReader() const
  {
    return dynamic_cast<XMLReaderImp&>(getTreeReader());
  }


  void XMLReader::open(const string& filename)
  {
    getXMLReader().open(filename);
  }

  void XMLReader::open(std::istream& is)
  {
    getXMLReader().open(is);
  }

  void XMLReader::open(const XMLBufferWriter& mw)
  {
    getXMLReader().open(mw);
  }

  // Clone a reader
  void XMLReader::open(XMLReader& old, const string& xpath)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    getXMLReader().open(old.getXMLReader(), xpath);
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
  }

  void XMLReader::close()
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    getXMLReader().close();
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
  }

  // Internal cloning function
  TreeReaderImp* XMLReader::clone(const std::string& xpath)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    return new XMLReaderImp(getXMLReader(), xpath);
  }


  bool XMLReader::is_open()
  {
    return getXMLReader().is_open();
  }

  void XMLReader::print(ostream& os)
  {
    getXMLReader().print(os);
  }
   
  void XMLReader::printCurrentContext(ostream& os)
  {
    getXMLReader().print(os);
  }
   
  // Return the entire contents of the Reader as a TreeRep
  void XMLReader::treeRep(TreeRep& output)
  {
    QDP_error_exit("XMLReader::treeRep function not implemented in a derived class");
  }
        
  //! Return the current context as a TreeRep
  void XMLReader::treeRepCurrentContext(TreeRep& output)
  {
    QDP_error_exit("XMLReader::treeRepCurrentContext function not implemented in a derived class");
  }
        
  //! Count the number of occurances from the Xpath query
  bool XMLReader::exist(const std::string& xpath)
  {
    return getXMLReader().exist(xpath);
  }

  // Count the number of occurances from the Xpath query
  int XMLReader::count(const std::string& xpath)
  {
    getXMLReader().count(xpath);
  }

  //! Count the number of occurances from the Xpath query
  int XMLReader::countArrayElem()
  {
    return getXMLReader().countArrayElem();
  }

  // Return tag for array element n
  std::string XMLReader::arrayElem(int n) const
  {
    return getXMLReader().arrayElem(n);
  }

  // Namespace Registration?
  void XMLReader::registerNamespace(const std::string& prefix, const string& uri)
  {
    getXMLReader().registerNamespace(prefix, uri);
  }



  //--------------------------------------------------------------------------------
  // XML writer base class
  XMLWriter::XMLWriter()
  {
  }

  XMLWriter::~XMLWriter()
  {
  }

  void XMLWriter::openSimple(const string& tagname)
  {
    getXMLWriter().openSimple(tagname);
  }

  void XMLWriter::closeSimple()
  {
    getXMLWriter().closeSimple();
  }

  void XMLWriter::openStruct(const string& tagname)
  {
    getXMLWriter().openStruct(tagname);
  }

  void XMLWriter::closeStruct()
  {
    getXMLWriter().closeStruct();
  }

  void XMLWriter::openTag(const string& tagname)
  {
    getXMLWriter().openTag(tagname);
  }

  void XMLWriter::openTag(const string& nsprefix, const string& tagname)
  {
    getXMLWriter().openTag(nsprefix,tagname);
  }

//  void XMLWriter::openTag(const string& tagname, XMLWriterAPI::AttributeList& al)
//  {
//    getXMLWriter().openTag(tagname,al);
//  }

//  void XMLWriter::openTag(const string& nsprefix, 
//			  const string& tagname, 
//			  XMLWriterAPI::AttributeList& al)
//  {
//    getXMLWriter().openTag(nsprefix,tagname,al);
//  }

  void XMLWriter::closeTag()
  {
    getXMLWriter().closeTag();
  }

  void XMLWriter::emptyTag(const string& tagname)
  {
    getXMLWriter().emptyTag(tagname);
  }
//  void XMLWriter::emptyTag(const string& tagname,  XMLWriterAPI::AttributeList& al)
//  { 
//    getXMLWriter().emptyTag(tagname,al);
//  }

//  void XMLWriter::emptyTag(const string& nsprefix, 
//			   const string& tagname, 
//			   XMLWriterAPI::AttributeList& al)
// {
//    getXMLWriter().emptyTag(nsprefix,tagname,al);
//  }

  // Return tag for array element n
  std::string XMLWriter::arrayElem(int n) const
  {
    return getXMLWriter().arrayElem(n);
  }

  // Flush the buffer
  void XMLWriter::flush()
  {
    getXMLWriter().flush();
  }

  // Time to build a telephone book of basic primitives
  // Overloaded Writer Functions
  void XMLWriter::write(const std::string& tagname, const string& output)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    getXMLWriter().write(tagname,output);
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
  }
  void XMLWriter::write(const string& tagname, const int& output)
  {
    getXMLWriter().write(tagname,output);
  }
  void XMLWriter::write(const string& tagname, const unsigned int& output)
  {
    getXMLWriter().write(tagname,output);
  }
  void XMLWriter::write(const string& tagname, const short int& output)
  {
    getXMLWriter().write(tagname,output);
  }
  void XMLWriter::write(const string& tagname, const unsigned short int& output)
  {
    getXMLWriter().write(tagname,output);
  }
  void XMLWriter::write(const string& tagname, const long int& output)
  {
    getXMLWriter().write(tagname,output);
  }
  void XMLWriter::write(const string& tagname, const unsigned long int& output)
  {
    getXMLWriter().write(tagname,output);
  }
  void XMLWriter::write(const string& tagname, const float& output)
  {
    getXMLWriter().write(tagname,output);
  }
  void XMLWriter::write(const string& tagname, const double& output)
  {
    getXMLWriter().write(tagname,output);
  }
  void XMLWriter::write(const string& tagname, const bool& output)
  {
    getXMLWriter().write(tagname,output);
  }
   
  // Write XML string
  void XMLWriter::writeXML(const string& output)
  {
    getXMLWriter().writeXML(output);
  }

  void XMLWriter::write(const std::string& tagname, const multi1d<int>& output)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    getXMLWriter().write(tagname,output);
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
  }
  void XMLWriter::write(const std::string& tagname, const multi1d<unsigned int>& output)
  {
    getXMLWriter().write(tagname,output);
  }
  void XMLWriter::write(const std::string& tagname, const multi1d<short int>& output)
  {
    getXMLWriter().write(tagname,output);
  }
  void XMLWriter::write(const std::string& tagname, const multi1d<unsigned short int>& output)
  {
    getXMLWriter().write(tagname,output);
  }
  void XMLWriter::write(const std::string& tagname, const multi1d<long int>& output)
  {
    getXMLWriter().write(tagname,output);
  }
  void XMLWriter::write(const std::string& tagname, const multi1d<unsigned long int>& output)
  {
    getXMLWriter().write(tagname,output);
  }
  void XMLWriter::write(const std::string& tagname, const multi1d<float>& output)
  {
    getXMLWriter().write(tagname,output);
  }
  void XMLWriter::write(const std::string& tagname, const multi1d<double>& output)
  {
    getXMLWriter().write(tagname,output);
  }
  void XMLWriter::write(const std::string& tagname, const multi1d<bool>& output)
  {
    getXMLWriter().write(tagname,output);
  }


  // Write something from a reader
//  void write(TreeWriter& xml, const std::string& tagname, const XMLReader& d)
//  {
//    xml.openTag(tagname);
//    xml << d;
//    xml.closeTag();
//  }

  XMLWriter& operator<<(XMLWriter& xml, const XMLReader& d)
  {
    ostringstream os;
    const_cast<XMLReader&>(d).printCurrentContext(os);
    xml.writeXML(os.str());
    return xml;
  }

  // Write something from a XMLBufferWriter
//  void write(XMLWriter& xml, const std::string& tagname, const XMLBufferWriter& d)
//  {
//    xml.openTag(tagname);
//    xml << d;
//    xml.closeTag();
//  }

  XMLWriter& operator<<(XMLWriter& xml, const XMLBufferWriter& d)
  {
    xml.writeXML(const_cast<XMLBufferWriter&>(d).printCurrentContext());
    return xml;
  }


  //--------------------------------------------------------------------------------
  // XML writer to a buffer
  XMLBufferWriter::XMLBufferWriter() 
  {
    buffer_writer = new XMLBufferWriterImp();
    initP = true;
  }

  XMLBufferWriter::XMLBufferWriter(const std::string& s) 
  {
    buffer_writer = new XMLBufferWriterImp(s);
    initP = true;
  }

  XMLBufferWriter::~XMLBufferWriter() 
  {
    if (initP)
      delete buffer_writer;
  }

  // The actual implementation
  XMLWriterImp& XMLBufferWriter::getXMLWriter() const
  {
    if (! initP)
    {
      QDPIO::cerr << "XMLBufferWriter is not initialized" << endl;
      QDP_abort(1);
    }

    return *buffer_writer;
  }

  void XMLBufferWriter::open(const std::string& s) 
  {
    dynamic_cast<XMLBufferWriterImp&>(getXMLWriter()).open(s);
  }

  string XMLBufferWriter::str() const
  {
    return dynamic_cast<XMLBufferWriterImp&>(getXMLWriter()).str();
  }

  string XMLBufferWriter::printCurrentContext() const 
  {
    return dynamic_cast<XMLBufferWriterImp&>(getXMLWriter()).printCurrentContext();
  }


  //--------------------------------------------------------------------------------
  // XML Writer to a file
  XMLFileWriter::XMLFileWriter()
  {
    file_writer = new XMLFileWriterImp();
    initP = true;
  }

  // Constructor from a filename
  XMLFileWriter::XMLFileWriter(const std::string& filename, bool write_prologue)
  {
    file_writer = new XMLFileWriterImp(filename, write_prologue);
    initP = true;
  }

  XMLFileWriter::~XMLFileWriter() 
  {
    if (initP)
      delete file_writer;
  }

  // The actual implementation
  XMLWriterImp& XMLFileWriter::getXMLWriter() const
  {
    if (! initP)
    {
      QDPIO::cerr << "XMLFileWriter is not initialized" << endl;
      QDP_abort(1);
    }

    return *file_writer;
  }

  void XMLFileWriter::open(const std::string& filename, bool write_prologue)
  {
    dynamic_cast<XMLFileWriterImp&>(getXMLWriter()).open(filename, write_prologue);
  }


  void XMLFileWriter::close()
  {
    dynamic_cast<XMLFileWriterImp&>(getXMLWriter()).close();
  }

  // Propagate status to all nodes
  bool XMLFileWriter::is_open()
  {
    return dynamic_cast<XMLFileWriterImp&>(getXMLWriter()).is_open();
  }


  // Flush the buffer
  void XMLFileWriter::flush()
  {
    getXMLWriter().flush();
  }

  // Propagate status to all nodes
  bool XMLFileWriter::fail() const
  {
    return dynamic_cast<XMLFileWriterImp&>(getXMLWriter()).fail();
  }

} // namespace QDP;
