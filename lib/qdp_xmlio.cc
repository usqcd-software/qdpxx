// $Id: qdp_xmlio.cc,v 1.42 2008-06-27 13:31:23 bjoo Exp $
//
/*! @file
 * @brief XML IO support
 */

#include "qdp.h"
#include <list>

namespace QDP 
{

  using std::string;

  //--------------------------------------------------------------------------------
  // XML classes
  // XML reader class
  XMLReader::XMLReader() {iop=derived=false;}

  XMLReader::XMLReader(const std::string& filename)
  {
    iop = derived = false;
    open(filename);
  }

  XMLReader::XMLReader(std::istream& is)
  {
    iop = derived = false;
    open(is);
  }

  XMLReader::XMLReader(const XMLBufferWriter& mw)
  {
    iop = derived = false;
    open(mw);
  }

  XMLReader::XMLReader(XMLReader& old, const string& xpath) : BasicXPathReader() 
  {
    iop = false;
    derived = true;
    open(old, xpath);
  }


  void XMLReader::open(const string& filename)
  {
    if (Layout::primaryNode())
    {
#if 0
      BasicXPathReader::open(filename);
#else

#if defined(USE_REMOTE_QIO)
      QDPUtil::RemoteInputFileStream f;
      f.open(filename.c_str(),std::ifstream::in);
      BasicXPathReader::open(f);
#else
      ifstream f;
      f.open(filename.c_str(), ios::binary);
      if (f.fail())
      {
	QDPIO::cerr << "Error opening read file = " << filename << endl;
	QDP_abort(1);
      }
      BasicXPathReader::open(f);
#endif

#endif
    }

    iop = true;
    derived = false;
  }

  void XMLReader::open(std::istream& is)
  {
    if (Layout::primaryNode())
      BasicXPathReader::open(is);

    iop = true;
    derived = false;
  }

  void XMLReader::open(const XMLBufferWriter& mw)
  {
    if (Layout::primaryNode())
    {  
      std::istringstream is(const_cast<XMLBufferWriter&>(mw).str()+"\n");
      BasicXPathReader::open(is);
    }

    iop = true;
    derived = false;
  }

  void XMLReader::open(XMLReader& old, const string& xpath)
  {
    if( Layout::primaryNode()) 
    {
      BasicXPathReader::open((BasicXPathReader&)old, xpath);
    }

    iop = true;
    derived = true;
  }

  void XMLReader::close()
  {
    if (is_open()) 
    {
      if (Layout::primaryNode()) 
	BasicXPathReader::close();

      iop = false;
      derived = false;
    }
  }



  bool XMLReader::is_open() {return iop;}

  bool XMLReader::is_derived() const {return derived;}

  XMLReader::~XMLReader() {close();}


  // Overloaded Reader Functions
  void XMLReader::get(const std::string& xpath, string& result)
  {
    // Only primary node can grab string
    if (Layout::primaryNode()) 
      BasicXPathReader::get(xpath, result);

    // broadcast string
    Internal::broadcast_str(result);
  }

  void XMLReader::get(const std::string& xpath, int& result)
  {
    readPrimitive<int>(xpath, result);
  }
  void XMLReader::get(const std::string& xpath, unsigned int& result)
  {
    readPrimitive<unsigned int>(xpath, result);
  }
  void XMLReader::get(const std::string& xpath, short int& result)
  {
    readPrimitive<short int>(xpath, result);
  }
  void XMLReader::get(const std::string& xpath, unsigned short int& result)
  {
    readPrimitive<unsigned short int>(xpath, result);
  }
  void XMLReader::get(const std::string& xpath, long int& result)
  {
    readPrimitive<long int>(xpath, result);
  }
  void XMLReader::get(const std::string& xpath, unsigned long int& result)
  {
    readPrimitive<unsigned long int>(xpath, result);
  }
  void XMLReader::get(const std::string& xpath, float& result)
  {
    readPrimitive<float>(xpath, result);
  }
  void XMLReader::get(const std::string& xpath, double& result)
  {
    readPrimitive<double>(xpath, result);
  }
  void XMLReader::get(const std::string& xpath, bool& result)
  {
    readPrimitive<bool>(xpath, result);
  }
   
  template<typename T>
  void XMLReader::readPrimitive(const std::string& xpath, T& result)
  {
    if (Layout::primaryNode()) {
      BasicXPathReader::get(xpath, result);
    }

    // Now broadcast back out to all nodes
    Internal::broadcast(result);
  }

  void XMLReader::print(ostream& os)
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
   
  void XMLReader::printCurrentContext(ostream& os)
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
   
  int XMLReader::count(const string& xpath)
  {
    int n;
    if (Layout::primaryNode())
      n = BasicXPathReader::count(xpath);

    // Now broadcast back out to all nodes
    Internal::broadcast(n);
    return n;
  }
   
  // Namespace Registration?
  void XMLReader::registerNamespace(const std::string& prefix, const string& uri)
  {
    if (Layout::primaryNode())
      BasicXPathReader::registerNamespace(prefix, uri);
  }

  // Overloaded Reader Functions
  void read(XMLReader& xml, const std::string& xpath, string& result)
  {
    xml.get(xpath, result);
  }
  void read(XMLReader& xml, const std::string& xpath, int& result)
  {
    xml.get(xpath, result);
  }
  void read(XMLReader& xml, const std::string& xpath, unsigned int& result)
  {
    xml.get(xpath, result);
  }
  void read(XMLReader& xml, const std::string& xpath, short int& result)
  {
    xml.get(xpath, result);
  }
  void read(XMLReader& xml, const std::string& xpath, unsigned short int& result)
  {
    xml.get(xpath, result);
  }
  void read(XMLReader& xml, const std::string& xpath, long int& result)
  {
    xml.get(xpath, result);
  }
  void read(XMLReader& xml, const std::string& xpath, unsigned long int& result)
  {
    xml.get(xpath, result);
  }
  void read(XMLReader& xml, const std::string& xpath, float& result)
  {
    xml.get(xpath, result);
  }
  void read(XMLReader& xml, const std::string& xpath, double& result)
  {
    xml.get(xpath, result);
  }
  void read(XMLReader& xml, const std::string& xpath, bool& result)
  {
    xml.get(xpath, result);
  }
   

  //! Read a XML multi1d element
  template<typename T>
  void readArrayPrimitive(XMLReader& xml, const std::string& s, multi1d<T>& result)
  {
    std::ostringstream error_message;
  
    // Try reading the list as a string
    string list_string;
    read(xml, s, list_string);

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

  template<>
  void read(XMLReader& xml, const std::string& xpath, multi1d<int>& result)
  {
    readArrayPrimitive<int>(xml, xpath, result);
  }
  template<>
  void read(XMLReader& xml, const std::string& xpath, multi1d<unsigned int>& result)
  {
    readArrayPrimitive<unsigned int>(xml, xpath, result);
  }
  template<>
  void read(XMLReader& xml, const std::string& xpath, multi1d<short int>& result)
  {
    readArrayPrimitive<short int>(xml, xpath, result);
  }
  template<>
  void read(XMLReader& xml, const std::string& xpath, multi1d<unsigned short int>& result)
  {
    readArrayPrimitive<unsigned short int>(xml, xpath, result);
  }
  template<>
  void read(XMLReader& xml, const std::string& xpath, multi1d<long int>& result)
  {
    readArrayPrimitive<long int>(xml, xpath, result);
  }
  template<>
  void read(XMLReader& xml, const std::string& xpath, multi1d<unsigned long int>& result)
  {
    readArrayPrimitive<unsigned long int>(xml, xpath, result);
  }
  template<>
  void read(XMLReader& xml, const std::string& xpath, multi1d<float>& result)
  {
    readArrayPrimitive<float>(xml, xpath, result);
  }
  template<>
  void read(XMLReader& xml, const std::string& xpath, multi1d<double>& result)
  {
    readArrayPrimitive<double>(xml, xpath, result);
  }
  template<>
  void read(XMLReader& xml, const std::string& xpath, multi1d<bool>& result)
  {
    readArrayPrimitive<bool>(xml, xpath, result);
  }
  template<>
  void read(XMLReader& xml, const std::string& xpath, multi1d<Integer>& result)
  {
    readArrayPrimitive<Integer>(xml, xpath, result);
  }
  template<>
  void read(XMLReader& xml, const std::string& xpath, multi1d<Real32>& result)
  {
    readArrayPrimitive<Real32>(xml, xpath, result);
  }
  template<>
  void read(XMLReader& xml, const std::string& xpath, multi1d<Real64>& result)
  {
    readArrayPrimitive<Real64>(xml, xpath, result);
  }
  template<>
  void read(XMLReader& xml, const std::string& xpath, multi1d<Boolean>& result)
  {
    readArrayPrimitive<Boolean>(xml, xpath, result);
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
    openTag(tagname);
  }

  void XMLWriter::closeSimple()
  {
    closeTag();
  }

  void XMLWriter::openStruct(const string& tagname)
  {
    openTag(tagname);
  }

  void XMLWriter::closeStruct()
  {
    closeTag();
  }

  void XMLWriter::openTag(const string& tagname)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::openTag(tagname);
  }

  void XMLWriter::openTag(const string& nsprefix, const string& tagname)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::openTag(nsprefix,tagname);
  }

  void XMLWriter::openTag(const string& tagname, XMLWriterAPI::AttributeList& al)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::openTag(tagname,al);
  }

  void XMLWriter::openTag(const string& nsprefix, 
			  const string& tagname, 
			  XMLWriterAPI::AttributeList& al)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::openTag(nsprefix,tagname,al);
  }

  void XMLWriter::closeTag()
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::closeTag();
  }

  void XMLWriter::emptyTag(const string& tagname)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::emptyTag(tagname);
  }
  void XMLWriter::emptyTag(const string& tagname,  XMLWriterAPI::AttributeList& al)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::emptyTag(tagname,al);
  }

  void XMLWriter::emptyTag(const string& nsprefix, 
			   const string& tagname, 
			   XMLWriterAPI::AttributeList& al)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::emptyTag(nsprefix,tagname,al);
  }


  // Overloaded Writer Functions
  void XMLWriter::write(const string& output)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::write(output);
  }
  void XMLWriter::write(const int& output)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::write(output);
  }
  void XMLWriter::write(const unsigned int& output)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::write(output);
  }
  void XMLWriter::write(const short int& output)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::write(output);
  }
  void XMLWriter::write(const unsigned short int& output)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::write(output);
  }
  void XMLWriter::write(const long int& output)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::write(output);
  }
  void XMLWriter::write(const unsigned long int& output)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::write(output);
  }
  void XMLWriter::write(const float& output)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::write(output);
  }
  void XMLWriter::write(const double& output)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::write(output);
  }
  void XMLWriter::write(const bool& output)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::write(output);
  }
   
  // Write XML string
  void XMLWriter::writeXML(const string& output)
  {
    if (Layout::primaryNode())
      XMLSimpleWriter::writeXML(output);
  }


  // Push a group name
  void push(XMLWriter& xml, const string& s) {xml.openStruct(s);}

  // Pop a group name
  void pop(XMLWriter& xml) {xml.closeStruct();}

  // Write something from a reader
  void write(XMLWriter& xml, const std::string& s, const XMLReader& d)
  {
    xml.openTag(s);
    xml << d;
    xml.closeTag();
  }

  XMLWriter& operator<<(XMLWriter& xml, const XMLReader& d)
  {
    ostringstream os;
    const_cast<XMLReader&>(d).printCurrentContext(os);
    xml.writeXML(os.str());
    return xml;
  }

  // Write something from a XMLBufferWriter
  void write(XMLWriter& xml, const std::string& s, const XMLBufferWriter& d)
  {
    xml.openTag(s);
    xml << d;
    xml.closeTag();
  }

  XMLWriter& operator<<(XMLWriter& xml, const XMLBufferWriter& d)
  {
    xml.writeXML(const_cast<XMLBufferWriter&>(d).printCurrentContext());
    return xml;
  }

  // Time to build a telephone book of basic primitives
  template<typename T>
  void writePrimitive(XMLWriter& xml, const string& s, const T& d)
  {
    xml.openTag(s);
    xml.write(d);
    xml.closeTag();
  }

  void write(XMLWriter& xml, const string& s, const string& d)
  {
    writePrimitive<string>(xml, s, d);
  }

  void write(XMLWriter& xml, const string& s, const char* d)
  {
    writePrimitive<string>(xml, s, string(d));
  }

  void write(XMLWriter& xml, const string& s, const char& d)
  {
    writePrimitive<char>(xml, s, d);
  }

  void write(XMLWriter& xml, const string& s, const int& d)
  {
    writePrimitive<int>(xml, s, d);
  }

  void write(XMLWriter& xml, const string& s, const unsigned int& d)
  {
    writePrimitive<unsigned int>(xml, s, d);
  }

  void write(XMLWriter& xml, const string& s, const short int& d)
  {
    writePrimitive<short int>(xml, s, d);
  }

  void write(XMLWriter& xml, const string& s, const unsigned short int& d)
  {
    writePrimitive<unsigned short int>(xml, s, d);
  }

  void write(XMLWriter& xml, const string& s, const long int& d)
  {
    writePrimitive<long int>(xml, s, d);
  }

  void write(XMLWriter& xml, const string& s, const unsigned long int& d)
  {
    writePrimitive<unsigned long int>(xml, s, d);
  }

  void write(XMLWriter& xml, const string& s, const float& d)
  {
    writePrimitive<float>(xml, s, d);
  }

  void write(XMLWriter& xml, const string& s, const double& d)
  {
    writePrimitive<double>(xml, s, d);
  }

  void write(XMLWriter& xml, const string& s, const bool& d)
  {
    writePrimitive<bool>(xml, s, d);
  }

  // Versions that do not print a name
  XMLWriter& operator<<(XMLWriter& xml, const string& d) {xml.write(d);return xml;}
  XMLWriter& operator<<(XMLWriter& xml, const char* d) {xml.write(string(d));return xml;}
  XMLWriter& operator<<(XMLWriter& xml, const char& d) {xml.write(d);return xml;}
  XMLWriter& operator<<(XMLWriter& xml, const int& d) {xml.write(d);return xml;}
  XMLWriter& operator<<(XMLWriter& xml, const unsigned int& d) {xml.write(d);return xml;}
  XMLWriter& operator<<(XMLWriter& xml, const short int& d) {xml.write(d);return xml;}
  XMLWriter& operator<<(XMLWriter& xml, const unsigned short int& d) {xml.write(d);return xml;}
  XMLWriter& operator<<(XMLWriter& xml, const long int& d) {xml.write(d);return xml;}
  XMLWriter& operator<<(XMLWriter& xml, const unsigned long int& d) {xml.write(d);return xml;}
  XMLWriter& operator<<(XMLWriter& xml, const float& d) {xml.write(d);return xml;}
  XMLWriter& operator<<(XMLWriter& xml, const double& d) {xml.write(d);return xml;}
  XMLWriter& operator<<(XMLWriter& xml, const bool& d) {xml.write(d);return xml;}


  // Write an array of basic types
  template<typename T>
  void writeArrayPrimitive(XMLWriter& xml, const std::string& s, const multi1d<T>& s1)
  {
    std::ostringstream output;

    if (s1.size() > 0)
    {
      output << s1[0];
      for(int index=1; index < s1.size(); index++) 
	output << " " << s1[index];
    }
    
    // Write the array - do not use a normal string write
    xml.openTag(s);
    xml << output.str();
    xml.closeTag();
  }


  template<>
  void write(XMLWriter& xml, const std::string& xpath, const multi1d<int>& output)
  {
    writeArrayPrimitive<int>(xml, xpath, output);
  }
  template<>
  void write(XMLWriter& xml, const std::string& xpath, const multi1d<unsigned int>& output)
  {
    writeArrayPrimitive<unsigned int>(xml, xpath, output);
  }
  template<>
  void write(XMLWriter& xml, const std::string& xpath, const multi1d<short int>& output)
  {
    writeArrayPrimitive<short int>(xml, xpath, output);
  }
  template<>
  void write(XMLWriter& xml, const std::string& xpath, const multi1d<unsigned short int>& output)
  {
    writeArrayPrimitive<unsigned short int>(xml, xpath, output);
  }
  template<>
  void write(XMLWriter& xml, const std::string& xpath, const multi1d<long int>& output)
  {
    writeArrayPrimitive<long int>(xml, xpath, output);
  }
  template<>
  void write(XMLWriter& xml, const std::string& xpath, const multi1d<unsigned long int>& output)
  {
    writeArrayPrimitive<unsigned long int>(xml, xpath, output);
  }
  template<>
  void write(XMLWriter& xml, const std::string& s, const multi1d<float>& s1)
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
    xml.openTag(s);
    xml << output.str();
    xml.closeTag();
  }
  template<>
  void write(XMLWriter& xml, const std::string& s, const multi1d<double>& s1)
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
    xml.openTag(s);
    xml << output.str();
    xml.closeTag();
  }
  template<>
  void write(XMLWriter& xml, const std::string& xpath, const multi1d<bool>& output)
  {
    writeArrayPrimitive<bool>(xml, xpath, output);
  }
  template<>
  void write(XMLWriter& xml, const std::string& xpath, const multi1d<Integer>& output)
  {
    writeArrayPrimitive<Integer>(xml, xpath, output);
  }
  template<>
  void write(XMLWriter& xml, const std::string& s, const multi1d<Real32>& s1)
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
    xml.openTag(s);
    xml << output.str();
    xml.closeTag();
  }
  template<>
  void write(XMLWriter& xml, const std::string& s, const multi1d<Real64>& s1)
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
    xml.openTag(s);
    xml << output.str();
    xml.closeTag();
  }
  template<>
  void write(XMLWriter& xml, const std::string& xpath, const multi1d<Boolean>& output)
  {
    writeArrayPrimitive<Boolean>(xml, xpath, output);
  }



  //--------------------------------------------------------------------------------
  // XML writer to a buffer
  XMLBufferWriter::XMLBufferWriter() {indent_level=0;}

  XMLBufferWriter::XMLBufferWriter(const std::string& s) {open(s);}

  void XMLBufferWriter::open(const std::string& s) 
  {
    if (Layout::primaryNode())
      output_stream.str(s);
  }

  string XMLBufferWriter::str() const
  {
    ostringstream s;
  
    if (Layout::primaryNode()) 
    {
      writePrologue(s);
      s << output_stream.str() << "\n";
    }
    
    return s.str();
  }

  string XMLBufferWriter::printCurrentContext() const {return output_stream.str();}

  XMLBufferWriter::~XMLBufferWriter() {}


  //--------------------------------------------------------------------------------
  // XML Writer to a file
  XMLFileWriter::XMLFileWriter() {indent_level=0;}

  void XMLFileWriter::open(const std::string& filename, bool write_prologue)
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


  void XMLFileWriter::close()
  {
    if (is_open()) 
    {
      if (Layout::primaryNode()) 
	output_stream.close();
    }
  }

  // Propagate status to all nodes
  bool XMLFileWriter::is_open()
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
  void XMLFileWriter::flush()
  {
    if (is_open()) 
    {
      if (Layout::primaryNode()) 
	output_stream.flush();
    }
  }

  // Propagate status to all nodes
  bool XMLFileWriter::fail() const
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

  XMLFileWriter::~XMLFileWriter() {close();}


  //--------------------------------------------------------------------------------
  // XML handle class for arrays
  XMLArrayWriter::~XMLArrayWriter()
  {
    if (initP)
      closeArray();
  }

  void XMLArrayWriter::openArray(const string& tagname)
  {
    //  QDP_info("openArray: stack_empty = %d  tagname=%s",
    //	   (contextStack.empty()) ? 1 : 0,
    //	   tagname.c_str());

    if (initP)
      QDP_error_exit("XMLArrayWriter: calling openArray twice");

    if (arrayTag)
      QDP_error_exit("XMLArrayWriter: internal error - array tag already written");

    if (! contextStack.empty())
      QDP_error_exit("XMLArrayWriter: context stack not empty");

    qname = tagname;
    elem_qname = "elem";    // for now fix the name - maintains internal consistency

    openTag(qname);   // array tagname

    initP = false;          // not fully initialized yet
    arrayTag = true;
  }

  void XMLArrayWriter::closeArray()
  {
    //  QDP_info("closeArray");

    if (! initP)
      QDP_error_exit("XMLArrayWriter: calling closeArray but not initialized");

    if (! contextStack.empty())
      QDP_error_exit("XMLArrayWriter: context stack not empty");

    closeTag();   // array tagname

    if (array_size > 0 && elements_written != array_size)
      QDP_error_exit("XMLArrayWriter: failed to write all the %d required elements: instead = %d",
		     array_size,elements_written);

    initP = arrayTag = false;
    elements_written = 0;
    indent_level = 0;
    simpleElements = false; // do not know this yet
  }

  void XMLArrayWriter::openStruct(const string& tagname)
  {
    //  QDP_info("openStruct: stack_empty = %d  tagname=%s",
    //	   (contextStack.empty()) ? 1 : 0,
    //	   tagname.c_str());

    if (! arrayTag)
    {
      openArray(tagname);
      return;
    }

    if (! initP)
    {
      if (elements_written == 0)
      {
	// This is the first time this is called
	// From now on, all elements must be STRUCT
	simpleElements = false;
      }
      else
	QDP_error_exit("XMLArrayWriter: internal error - data written but state not initialized");

      initP = true;
    }

    if (simpleElements)
      QDP_error_exit("XMLArrayWriter: suppose to write simple types but asked to write a struct");


    if (contextStack.empty())
      openTag(elem_qname);   // ignore user provided name and use default name
    else
      openTag(tagname);  // use user provided name

    ElementType el = STRUCT;
    contextStack.push(el);
  }

  void XMLArrayWriter::closeStruct()
  {
    //  QDP_info("closeStruct: stack_empty = %d",
    //	   (contextStack.empty()) ? 1 : 0);

    if (! initP)
      QDP_error_exit("XMLArrayWriter: calling closeStruct but not initialized");

    if (contextStack.empty())
    {
      //    QDP_error_exit("XMLArrayWriter: context stack empty - probably no openStruct");
      closeArray();
      return;
    }

    ElementType topval = contextStack.top();
    if (topval != STRUCT)
      QDP_error_exit("XMLArrayWriter: found closeStruct without corresponding openStruct");

    contextStack.pop();

    closeTag();   // struct (or elem_qname)  tagname

    if (contextStack.empty())
    {
      elements_written++;
      //    QDP_info("finished writing element %d",elements_written);
    }
  }

  // Push a group name
  void push(XMLArrayWriter& xml) {xml.openStruct("");}

  // Pop a group name
  void pop(XMLArrayWriter& xml) {xml.closeStruct();}


} // namespace QDP;
