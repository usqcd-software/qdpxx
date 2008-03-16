// $Id: qdp_xml_imp.cc,v 1.1.2.2 2008-03-16 16:07:21 edwards Exp $
//
/*! @file
 * @brief XML IO support implementation
 */

#include "qdp.h"
#include "qdp_xml_imp.h"

namespace QDP 
{
  using std::string;

  //--------------------------------------------------------------------------------
  // XML classes
  // XML reader class
  XMLReaderImp::XMLReaderImp() {iop=derived=false;}

  XMLReaderImp::XMLReaderImp(const std::string& filename)
  {
    iop = derived = false;
    open(filename);
  }

  XMLReaderImp::XMLReaderImp(std::istream& is)
  {
    iop = derived = false;
    open(is);
  }

  XMLReaderImp::XMLReaderImp(const XMLBufferWriter& mw)
  {
    iop = derived = false;
    open(mw);
  }

  XMLReaderImp::XMLReaderImp(XMLReaderImp& old, const string& xpath) : 
    TreeReaderImp(), BasicXPathReader() 
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    iop = false;
    derived = true;
    open(old, xpath);
  }


  // Clone a reader from a different path
  XMLReaderImp* XMLReaderImp::clone(const std::string& xpath)
  {
    return new XMLReaderImp(*this, xpath);
  }


  void XMLReaderImp::open(const string& filename)
  {
    if (Layout::primaryNode())
      BasicXPathReader::open(filename);

    iop = true;
    derived = false;
  }

  void XMLReaderImp::open(std::istream& is)
  {
    if (Layout::primaryNode())
      BasicXPathReader::open(is);

    iop = true;
    derived = false;
  }

  void XMLReaderImp::open(const XMLBufferWriter& mw)
  {
    if (Layout::primaryNode())
    {  
      std::istringstream is(const_cast<XMLBufferWriter&>(mw).str()+"\n");
      BasicXPathReader::open(is);
    }

    iop = true;
    derived = false;
  }

  // Clone a reader
  void XMLReaderImp::open(XMLReaderImp& old, const string& xpath)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    if( Layout::primaryNode()) 
    {
      BasicXPathReader::open((BasicXPathReader&)old, xpath);
    }

    iop = true;
    derived = true;
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
  }

  void XMLReaderImp::close()
  {
    if (is_open()) 
    {
      if (Layout::primaryNode()) 
	BasicXPathReader::close();

      iop = false;
      derived = false;
    }
  }

  bool XMLReaderImp::is_open() {return iop;}

  bool XMLReaderImp::is_derived() const {return derived;}

  XMLReaderImp::~XMLReaderImp() {close();}


  // Driver for telephone reads
  template<typename T>
  void XMLReaderImp::readPrimitive(const std::string& xpath, T& result)
  {
    if (Layout::primaryNode()) {
      BasicXPathReader::get(xpath, result);
    }

    // Now broadcast back out to all nodes
    Internal::broadcast(result);
  }

  // Overloaded Reader Functions
  void XMLReaderImp::read(const std::string& xpath, string& result)
  {
    // Only primary node can grab string
    if (Layout::primaryNode()) 
      BasicXPathReader::get(xpath, result);

    // broadcast string
    Internal::broadcast_str(result);
  }

  void XMLReaderImp::read(const std::string& xpath, int& result)
  {
    readPrimitive<int>(xpath, result);
  }
  void XMLReaderImp::read(const std::string& xpath, unsigned int& result)
  {
    readPrimitive<unsigned int>(xpath, result);
  }
  void XMLReaderImp::read(const std::string& xpath, short int& result)
  {
    readPrimitive<short int>(xpath, result);
  }
  void XMLReaderImp::read(const std::string& xpath, unsigned short int& result)
  {
    readPrimitive<unsigned short int>(xpath, result);
  }
  void XMLReaderImp::read(const std::string& xpath, long int& result)
  {
    readPrimitive<long int>(xpath, result);
  }
  void XMLReaderImp::read(const std::string& xpath, unsigned long int& result)
  {
    readPrimitive<unsigned long int>(xpath, result);
  }
  void XMLReaderImp::read(const std::string& xpath, float& result)
  {
    readPrimitive<float>(xpath, result);
  }
  void XMLReaderImp::read(const std::string& xpath, double& result)
  {
    readPrimitive<double>(xpath, result);
  }
  void XMLReaderImp::read(const std::string& xpath, bool& result)
  {
    readPrimitive<bool>(xpath, result);
  }
   
  void XMLReaderImp::print(ostream& os)
  {
    ostringstream newos;
    std::string s;

    if (Layout::primaryNode())
    {
      BasicXPathReader::print(newos);
      s = newos.str();
    }

    // Now broadcast back out to all nodes
    Internal::broadcast_str(s);
    os << s;
  }
   
  void XMLReaderImp::printCurrentContext(ostream& os)
  {
    ostringstream newos;
    std::string s;

    if (Layout::primaryNode())
    {
      if (is_derived())
	BasicXPathReader::printChildren(newos);
      else
	BasicXPathReader::printRoot(newos);

      s = newos.str();
    }

    // Now broadcast back out to all nodes
    Internal::broadcast_str(s);
    os << s;
  }
   
  // Return the entire contents of the Reader as a TreeRep
  void XMLReaderImp::treeRep(TreeRep& output)
  {
    QDP_error_exit("XMLReaderImp::treeRep function not implemented in a derived class");
  }
        
  //! Return the current context as a TreeRep
  void XMLReaderImp::treeRepCurrentContext(TreeRep& output)
  {
    QDP_error_exit("XMLReaderImp::treeRepCurrentContext function not implemented in a derived class");
  }
        
  //! Count the number of occurances from the Xpath query
  bool XMLReaderImp::exist(const std::string& xpath)
  {
    int num = count(xpath);
    return (num > 0) ? true : false;
  }

  // Count the number of occurances from the Xpath query
  int XMLReaderImp::count(const std::string& xpath)
  {
    int n;
    if (Layout::primaryNode())
      n = BasicXPathReader::count(xpath);

    // Now broadcast back out to all nodes
    Internal::broadcast(n);
    return n;
  }

  //! Count the number of occurances from the Xpath query
  int XMLReaderImp::countArrayElem()
  {
    return count("elem");
  }

  // Return tag for array element n
  std::string XMLReaderImp::arrayElem(int n) const
  {
    std::ostringstream os;
    os << "elem[" << n+1 << "]";
    return os.str();
  }

  // Namespace Registration?
  void XMLReaderImp::registerNamespace(const std::string& prefix, const string& uri)
  {
    if (Layout::primaryNode())
      BasicXPathReader::registerNamespace(prefix, uri);
  }

  //! Read a XML multi1d element
  template<typename T>
  void readArrayPrimitive(XMLReaderImp& xml, const std::string& s, multi1d<T>& result)
  {
    std::ostringstream error_message;
  
    // Try reading the list as a string
    string list_string;
    xml.read(s, list_string);

    // Count the number of elements
    std::istringstream list_stream(list_string);
	
    int array_size = 0;
    T dummy;
    while(list_stream >> dummy)
      ++array_size;

    if ((! list_stream.eof()) && list_stream.fail())
    {
      error_message << "Error in reading array " << s << endl;
      throw error_message.str();
    }

    // It is not an error to have a zero-length array
    //  if (array_size == 0)
    //  {
    //    error_message << "Something wrong with reading array " << list_string << endl;
    //    throw error_message.str();
    //  }
      
    // Now resize the array to hold the no of elements.
    result.resize(array_size);

    // Get the elements one by one
    // I do not understand why, but use a new stringstream
    //  list_stream.str(list_string);
    std::istringstream list_stream2(list_string);

    for(int i=0; i < result.size(); i++) 
    {
      // read the element.
      list_stream2 >> result[i];
    }
  }

  void XMLReaderImp::read(const std::string& xpath, multi1d<int>& result)
  {
    readArrayPrimitive<int>(*this, xpath, result);
  }
  void XMLReaderImp::read(const std::string& xpath, multi1d<unsigned int>& result)
  {
    readArrayPrimitive<unsigned int>(*this, xpath, result);
  }
  void XMLReaderImp::read(const std::string& xpath, multi1d<short int>& result)
  {
    readArrayPrimitive<short int>(*this, xpath, result);
  }
  void XMLReaderImp::read(const std::string& xpath, multi1d<unsigned short int>& result)
  {
    readArrayPrimitive<unsigned short int>(*this, xpath, result);
  }
  void XMLReaderImp::read(const std::string& xpath, multi1d<long int>& result)
  {
    readArrayPrimitive<long int>(*this, xpath, result);
  }
  void XMLReaderImp::read(const std::string& xpath, multi1d<unsigned long int>& result)
  {
    readArrayPrimitive<unsigned long int>(*this, xpath, result);
  }
  void XMLReaderImp::read(const std::string& xpath, multi1d<float>& result)
  {
    readArrayPrimitive<float>(*this, xpath, result);
  }
  void XMLReaderImp::read(const std::string& xpath, multi1d<double>& result)
  {
    readArrayPrimitive<double>(*this, xpath, result);
  }
  void XMLReaderImp::read(const std::string& xpath, multi1d<bool>& result)
  {
    readArrayPrimitive<bool>(*this, xpath, result);
  }



  //--------------------------------------------------------------------------------
  // XML writer base class
  XMLWriterImp::XMLWriterImp()
  {
  }

  XMLWriterImp::~XMLWriterImp()
  {
  }

  void XMLWriterImp::openSimple(const string& tagname)
  {
    openTag(tagname);
  }

  void XMLWriterImp::closeSimple()
  {
    closeTag();
  }

  void XMLWriterImp::openStruct(const string& tagname)
  {
    openTag(tagname);
  }

  void XMLWriterImp::closeStruct()
  {
    closeTag();
  }

  void XMLWriterImp::openTag(const string& tagname)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    if (Layout::primaryNode())
      XMLSimpleWriter::openTag(tagname);
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
  }

  void XMLWriterImp::openTag(const string& nsprefix, const string& tagname)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::openTag(nsprefix,tagname);
  }

  void XMLWriterImp::openTag(const string& tagname, XMLWriterAPI::AttributeList& al)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::openTag(tagname,al);
  }

  void XMLWriterImp::openTag(const string& nsprefix, 
			  const string& tagname, 
			  XMLWriterAPI::AttributeList& al)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::openTag(nsprefix,tagname,al);
  }

  void XMLWriterImp::closeTag()
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    if (Layout::primaryNode())
      XMLSimpleWriter::closeTag();
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
  }

  void XMLWriterImp::emptyTag(const string& tagname)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::emptyTag(tagname);
  }
  void XMLWriterImp::emptyTag(const string& tagname,  XMLWriterAPI::AttributeList& al)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::emptyTag(tagname,al);
  }

  void XMLWriterImp::emptyTag(const string& nsprefix, 
			   const string& tagname, 
			   XMLWriterAPI::AttributeList& al)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::emptyTag(nsprefix,tagname,al);
  }

  // Return tag for array element n
  std::string XMLWriterImp::arrayElem(int n) const
  {
    return std::string("elem");
  }


  // Driver for telephone reads
  template<typename T>
  void XMLWriterImp::writePrimitive(const std::string& tagname, const T& output)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    if (Layout::primaryNode())
    {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
      openTag(tagname);
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
      XMLSimpleWriter::write(output);
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
      closeTag();
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    }
  }

  // Time to build a telephone book of basic primitives
  // Overloaded Writer Functions
  void XMLWriterImp::write(const std::string& tagname, const string& output)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    writePrimitive(tagname, output);
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
  }
  void XMLWriterImp::write(const string& tagname, const int& output)
  {
    writePrimitive(tagname, output);
  }
  void XMLWriterImp::write(const string& tagname, const unsigned int& output)
  {
    writePrimitive(tagname, output);
  }
  void XMLWriterImp::write(const string& tagname, const short int& output)
  {
    writePrimitive(tagname, output);
  }
  void XMLWriterImp::write(const string& tagname, const unsigned short int& output)
  {
    writePrimitive(tagname, output);
  }
  void XMLWriterImp::write(const string& tagname, const long int& output)
  {
    writePrimitive(tagname, output);
  }
  void XMLWriterImp::write(const string& tagname, const unsigned long int& output)
  {
    writePrimitive(tagname, output);
  }
  void XMLWriterImp::write(const string& tagname, const float& output)
  {
    writePrimitive(tagname, output);
  }
  void XMLWriterImp::write(const string& tagname, const double& output)
  {
    writePrimitive(tagname, output);
  }
  void XMLWriterImp::write(const string& tagname, const bool& output)
  {
    writePrimitive(tagname, output);
  }
   
  // Write XML string
  void XMLWriterImp::writeXML(const string& output)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::writeXML(output);
  }


  // Write an array of basic types
  template<typename T>
  void XMLWriterImp::writeArrayPrimitive(const std::string& s, const multi1d<T>& s1)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    std::ostringstream output;

    if (s1.size() > 0)
    {
      output << s1[0];
      for(int index=1; index < s1.size(); index++) 
	output << " " << s1[index];
    }
    
    // Write the array - do not use a normal string write
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    write(s, output.str());
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
  }


  void XMLWriterImp::write(const std::string& tagname, const multi1d<int>& output)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    writeArrayPrimitive(tagname, output);
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
  }
  void XMLWriterImp::write(const std::string& tagname, const multi1d<unsigned int>& output)
  {
    writeArrayPrimitive(tagname, output);
  }
  void XMLWriterImp::write(const std::string& tagname, const multi1d<short int>& output)
  {
    writeArrayPrimitive(tagname, output);
  }
  void XMLWriterImp::write(const std::string& tagname, const multi1d<unsigned short int>& output)
  {
    writeArrayPrimitive(tagname, output);
  }
  void XMLWriterImp::write(const std::string& tagname, const multi1d<long int>& output)
  {
    writeArrayPrimitive(tagname, output);
  }
  void XMLWriterImp::write(const std::string& tagname, const multi1d<unsigned long int>& output)
  {
    writeArrayPrimitive(tagname, output);
  }
  void XMLWriterImp::write(const std::string& tagname, const multi1d<float>& s1)
  {
    std::ostringstream output;
    output.precision(7);

    if (s1.size() > 0)
    {
      output << s1[0];
      for(int index=1; index < s1.size(); index++) 
	output << " " << s1[index];
    }
    
    // Write the array - do not use a normal string write
    write(tagname, output.str());
  }
  void XMLWriterImp::write(const std::string& tagname, const multi1d<double>& s1)
  {
    std::ostringstream output;
    output.precision(15);

    if (s1.size() > 0)
    {
      output << s1[0];
      for(int index=1; index < s1.size(); index++) 
	output << " " << s1[index];
    }
    
    // Write the array - do not use a normal string write
    write(tagname, output.str());
  }
  void XMLWriterImp::write(const std::string& tagname, const multi1d<bool>& output)
  {
    writeArrayPrimitive(tagname, output);
  }


  //--------------------------------------------------------------------------------
  // XML writer to a buffer
  XMLBufferWriterImp::XMLBufferWriterImp() {indent_level=0;}

  XMLBufferWriterImp::XMLBufferWriterImp(const std::string& s) {open(s);}

  void XMLBufferWriterImp::open(const std::string& s) 
  {
    if (Layout::primaryNode())
      output_stream.str(s);
  }

  string XMLBufferWriterImp::str() const
  {
    ostringstream s;
  
    if (Layout::primaryNode()) 
    {
      writePrologue(s);
      s << output_stream.str() << "\n";
    }
    
    return s.str();
  }

  string XMLBufferWriterImp::printCurrentContext() const {return output_stream.str();}

  XMLBufferWriterImp::~XMLBufferWriterImp() {}


  //--------------------------------------------------------------------------------
  // XML Writer to a file
  XMLFileWriterImp::XMLFileWriterImp() {indent_level=0;}

  // Constructor from a filename
  XMLFileWriterImp::XMLFileWriterImp(const std::string& filename, bool write_prologue)
  {
    open(filename, write_prologue);
  }

  void XMLFileWriterImp::open(const std::string& filename, bool write_prologue)
  {
    if (Layout::primaryNode())
    {
      output_stream.open(filename.c_str(), ofstream::out);
      if (output_stream.fail())
      {
	QDPIO::cerr << "Error opening write file = " << filename << endl;
	QDP_abort(1);
      }
      if (write_prologue)
	writePrologue(output_stream);
    }

    indent_level=0;
  }


  void XMLFileWriterImp::close()
  {
    if (is_open()) 
    {
      if (Layout::primaryNode()) 
	output_stream.close();
    }
  }

  // Propagate status to all nodes
  bool XMLFileWriterImp::is_open()
  {
    bool s = QDP_isInitialized();

    if (s)
    {
      if (Layout::primaryNode()) 
	s = output_stream.is_open();

      Internal::broadcast(s);
    }

    return s;
  }


  // Flush the buffer
  void XMLFileWriterImp::flush()
  {
    if (is_open()) 
    {
      if (Layout::primaryNode()) 
	output_stream.flush();
    }
  }

  // Propagate status to all nodes
  bool XMLFileWriterImp::fail() const
  {
    bool s = QDP_isInitialized();

    if (s)
    {
      if (Layout::primaryNode()) 
	s = output_stream.fail();

      Internal::broadcast(s);
    }

    return s;
  }

  XMLFileWriterImp::~XMLFileWriterImp() {close();}


} // namespace QDP;
