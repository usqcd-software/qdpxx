// -*- C++ -*-
// $Id: qdp_xmlio.h,v 1.6 2003-05-23 05:20:27 edwards Exp $

/*! @file
 * @brief XML IO support
 */

#include <string>
#include <fstream>
#include <sstream>

#include "xml_simplewriter.h"
#include "basic_xpath_reader.h"

QDP_BEGIN_NAMESPACE(QDP);

// Forward declarations
class XMLReader;
class XMLWriter;
class XMLBufferWriter;
class XMLFileWriter;


/*! @addtogroup io
 *
 * XML File input and output operations on QDP types
 *
 * @{
 */

//--------------------------------------------------------------------------------
//! XML reader class
class XMLReader : protected XMLXPathReader::BasicXPathReader
{
public:
  //! Empty constructor
  XMLReader();

  //! Construct from contents of file
  XMLReader(const std::string& filename);

  //! Construct from contents of stream
  XMLReader(std::istream& is);

  //! Construct from contents of a XMLBufferWriter
  XMLReader(const XMLBufferWriter& mw);

  ~XMLReader();

  /* The meaning of these should be clear to you */
  void open(const std::string& filename);
  void open(std::istream& is);
  void open(const XMLBufferWriter& mw);
  bool is_open();
  void close();
    
  /* So should these, there is just a lot of overloading */
  void get(const std::string& xpath, std::string& result);
  void get(const std::string& xpath, int& result);
  void get(const std::string& xpath, unsigned int& result);
  void get(const std::string& xpath, short int& result);
  void get(const std::string& xpath, unsigned short int& result);
  void get(const std::string& xpath, long int& result);
  void get(const std::string& xpath, unsigned long int& result);
  void get(const std::string& xpath, float& result);
  void get(const std::string& xpath, double& result);
  void get(const std::string& xpath, bool& result);

  //! Return the entire contents of the Reader as a stream
  void print(ostream& is);
        
  //! Return the root element of the Reader as a stream
  void printRoot(ostream& is);
        
private:
  bool  iop;  //file open or closed?
};


// Time to build a telephone book of basic primitives
void read(XMLReader& xml, const std::string& s, std::string& d);
void read(XMLReader& xml, const std::string& s, int& d);
void read(XMLReader& xml, const std::string& s, float& d);
void read(XMLReader& xml, const std::string& s, double& d);
void read(XMLReader& xml, const std::string& s, bool& d);




//--------------------------------------------------------------------------------
//! Metadata output class
class XMLWriter : protected XMLWriterAPI::XMLSimpleWriter
{
public:
  XMLWriter();
  ~XMLWriter();

  void openTag(const std::string& tagname);
  void openTag(const std::string& nsprefix, const std::string& tagname);
  void openTag(const std::string& tagname, XMLWriterAPI::AttributeList& al);

  void openTag(const std::string& nsprefix,
	       const std::string& tagname, 
	       XMLWriterAPI::AttributeList& al);

  void closeTag();

  void emptyTag(const std::string& tagname);
  void emptyTag(const std::string& nsprefix, const std::string& tagname);
  void emptyTag(const std::string& tagname, XMLWriterAPI::AttributeList& al);

  void emptyTag(const std::string& nsprefix,
		const std::string& tagname, 
		XMLWriterAPI::AttributeList& al);
    

  // Overloaded Writer Functions
  void write(const std::string& output);
  void write(const int& output);
  void write(const unsigned int& output);
  void write(const short int& output);
  void write(const unsigned short int& output);
  void write(const long int& output);
  void write(const unsigned long int& output);
  void write(const float& output);
  void write(const double& output);
  void write(const bool& output);

  // Write XML string
  void writeXML(const std::string& output);
};


//! Push a group name
void push(XMLWriter& xml, const std::string& s);

//! Pop a group name
void pop(XMLWriter& xml);

//! Write something from a reader
void write(XMLWriter& xml, const std::string& s, const XMLReader& d);
XMLWriter& operator<<(XMLWriter& xml, const XMLReader& d);

//! Write something from a XMLBufferWriter
void write(XMLWriter& xml, const std::string& s, const XMLBufferWriter& d);
XMLWriter& operator<<(XMLWriter& xml, const XMLBufferWriter& d);

// Time to build a telephone book of basic primitives
void write(XMLWriter& xml, const std::string& s, const std::string& d);
void write(XMLWriter& xml, const std::string& s, const int& d);
void write(XMLWriter& xml, const std::string& s, const float& d);
void write(XMLWriter& xml, const std::string& s, const double& d);
void write(XMLWriter& xml, const std::string& s, const bool& d);

// Versions that do not print a name
XMLWriter& operator<<(XMLWriter& xml, const std::string& d);
XMLWriter& operator<<(XMLWriter& xml, const int& d);
XMLWriter& operator<<(XMLWriter& xml, const float& d);
XMLWriter& operator<<(XMLWriter& xml, const double& d);
XMLWriter& operator<<(XMLWriter& xml, const bool& d);


//! Write a XML multi1d element
template<class T>
inline
void write(XMLWriter& xml, const std::string& s, const multi1d<T>& s1)
{
  xml.openTag(s);
  xml << s1;
  xml.closeTag(s);
}

//! Write a XML multi1d element
/*! This is the verbose method */
template<class T>
inline
void writeArray(XMLWriter& xml, 
		const std::string& sizeName, 
		const std::string& elemName, 
		const std::string& indexName,
		const unsigned int& indexStart,
		const multi1d<T>& s1)
{
  XMLWriterAPI::AttributeList alist;
  alist.push_back(XMLWriterAPI::Attribute("sizeName",  sizeName));
  alist.push_back(XMLWriterAPI::Attribute("elemName",  elemName));
  alist.push_back(XMLWriterAPI::Attribute("indexName", indexName));
  alist.push_back(XMLWriterAPI::Attribute("indexStart", indexStart));
      
  // Write the array - tag
  xml.openTag("array", alist);

  xml.openTag(sizeName);
  xml.write(s1.size());
  xml.closeTag();

  unsigned int index;
  for(index=0; index < s1.size(); index++) {
    alist.clear();
    alist.push_back(XMLWriterAPI::Attribute(indexName, index + indexStart));
    xml.openTag(elemName, alist);
    xml << s1[index];   // NOTE, can possibly grab user defined write's here
    xml.closeTag();
  }

  xml.closeTag(); // Array
}


//! Write a XML multi1d element
template<class T>
inline
XMLWriter& operator<<(XMLWriter& xml, const multi1d<T>& s1)
{
  // These attributes are the defaults
  std::string sizeName = "size";
  std::string elemName = "element";
  std::string indexName = "index";

  writeArray(xml, sizeName, elemName, indexName, 0, s1);
  return xml;
}


//! XML OScalar output
template<class T>
inline
void write(XMLWriter& xml, const std::string& s, const OScalar<T>& d)
{
  xml.openTag(s);
  xml << d;
  xml.closeTag();
}

//! XML OLattice output
template<class T>
inline
void write(XMLWriter& xml, const std::string& s, const OLattice<T>& d)
{
  xml.openTag(s);
  xml << d;
  xml.closeTag();
}


#if 0
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
//! Write metadata to a buffer
class XMLBufferWriter : public XMLWriter
{
public:
  //! Constructor
  /*! No prologue written */
  XMLBufferWriter();
  
  //! Destructor
  ~XMLBufferWriter();

  // Return entire stream as a string
  std::string str();
        
  // Return root element as a string
  std::string printRoot();
        
private:
  // The output stream...
  ostringstream output_stream;

  // The function that supplies the stream to the parent...
  ostream& getOstream(void) {return output_stream;}
};


//--------------------------------------------------------------------------------
//! Write data to a file
class XMLFileWriter : public XMLWriter
{
public:
  //! Empty constructor
  XMLFileWriter();

  //! Constructor from a filename
  explicit XMLFileWriter(const std::string& filename, bool write_prologue=true)
    {
      open(filename, write_prologue);
    }

  //! Destructor
  ~XMLFileWriter();

  bool is_open();
  void open(const std::string& filename, bool write_prologue=true)
    {
      if (Layout::primaryNode())
      {
	output_stream.open(filename.c_str(), ofstream::out);
	if (write_prologue)
	  writePrologue(output_stream);
      }

      indent_level=0;
      iop=true;
    }

  void close();
        
private:
  bool iop;
  ofstream output_stream;
  ostream& getOstream(void) {return output_stream;}
};


/*! @} */   // end of group io
QDP_END_NAMESPACE();
