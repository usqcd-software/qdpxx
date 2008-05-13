// $Id: qdp_binx.cc,v 1.9 2008-05-13 20:00:17 bjoo Exp $
//
// QDP data parallel interface to binx writers
//
// I assume that the primary node business is dealt
// with by the lower obects.
//

#include "qdp.h"

namespace QDP {




//-----------------------------------------
//! Binx Binary writer support
BinxWriter::BinxWriter() {}

BinxWriter::BinxWriter(const std::string& p) {open(p);}

void BinxWriter::open(const std::string& p) 
{
  std::string p_xml = p + "_binx.xml" ; 
  tobinary = new(nothrow) BinaryFileWriter(p); 
  if( tobinary == 0x0 ) {
    QDP_error_exit("UNable to new tobinary in qdp_binx.cc\n");
  }
  
  toxml =  new(nothrow) XMLFileWriter(p_xml) ;
  if( toxml == 0x0 ) { 
    QDP_error_exit("Unable to new toxml in qdp_binx.cc\n");
  }

  write_xml = true ;

  // write some binx stuff
  XMLWriterAPI::AttributeList alist;
  alist.clear();
  alist.push_back(XMLWriterAPI::Attribute("byteOrder", string("bigEndian")));
  toxml->openTag("binx", alist);

  alist.clear();
  alist.push_back(XMLWriterAPI::Attribute("src", p));
  toxml->openTag("dataset", alist);


  //xml << d.elem(i);
  //toxml.closeTag();



  if (! is_open())
    QDP_error_exit("BinxWriter: error opening file %s",p.c_str());
}

void BinxWriter::close()
{
  if (is_open())
  {
    toxml->closeTag();
    toxml->closeTag();

    tobinary->close(); 
    toxml->close(); 
  }
}


// Propagate status to all nodes
bool BinxWriter::is_open()
{
  bool s;

  // more thought
    s = tobinary->is_open(); 

  Internal::broadcast(s);
  return s;
}

void BinxWriter::flush()
{
  if (is_open()) 
  {
    tobinary->flush();
    toxml->flush();
  }
}

// Propagate status to all nodes
bool BinxWriter::fail()
{
  bool s;

  //  if (Layout::primaryNode()) 
  //  s = f.fail();

  s = tobinary->fail(); 

  // probably not needed
  Internal::broadcast(s);
  return s;
}

BinxWriter::~BinxWriter() {
  close();

  delete toxml ;
  delete tobinary ; 
}


// Wrappers for write functions
void write(BinxWriter& bin, const std::string& output)
{
  bin.write(output);
}

void write(BinxWriter& bin, const char* output)
{
  bin.write(std::string(output));
}

void write(BinxWriter& bin, char output)
{
  bin.write(output);
}

void write(BinxWriter& bin, int output)
{
  bin.write(output);
}

void write(BinxWriter& bin, unsigned int output)
{
  bin.write(output);
}

void write(BinxWriter& bin, short int output)
{
  bin.write(output);
}

void write(BinxWriter& bin, unsigned short int output)
{
  bin.write(output);
}

void write(BinxWriter& bin, long int output)
{
  bin.write(output);
}

void write(BinxWriter& bin, unsigned long int output)
{
  bin.write(output);
}

void write(BinxWriter& bin, float output)
{
  bin.write(output);
}

void write(BinxWriter& bin, double output)
{
  bin.write(output);
}

void write(BinxWriter& bin, bool output)
{
  bin.write(output);
}

// Different bindings for write functions
BinxWriter& operator<<(BinxWriter& bin, const std::string& output)
{
  write(bin, output);
  return bin;
}

BinxWriter& operator<<(BinxWriter& bin, const char* output)
{
  write(bin, output);
  return bin;
}

BinxWriter& operator<<(BinxWriter& bin, int output)
{
  write(bin, output);
  return bin;
}

BinxWriter& operator<<(BinxWriter& bin, unsigned int output)
{
  write(bin, output);
  return bin;
}

BinxWriter& operator<<(BinxWriter& bin, short int output)
{
  write(bin, output);
  return bin;
}

BinxWriter& operator<<(BinxWriter& bin, unsigned short int output)
{
  write(bin, output);
  return bin;
}

BinxWriter& operator<<(BinxWriter& bin, long int output)
{
  write(bin, output);
  return bin;
}

BinxWriter& operator<<(BinxWriter& bin, unsigned long int output)
{
  write(bin, output);
  return bin;
}

BinxWriter& operator<<(BinxWriter& bin, float output)
{
  write(bin, output);
  return bin;
}

BinxWriter& operator<<(BinxWriter& bin, double output)
{
  write(bin, output);
  return bin;
}

BinxWriter& operator<<(BinxWriter& bin, bool output)
{
  write(bin, output);
  return bin;
}

// ========================================
//
//  write based methods for binx
//
// ========================================

void BinxWriter::write(const string& output)
{
  if( write_xml )
    {
      string nn = "character-8" ;
      toxml->emptyTag(nn) ; 
    }
  tobinary->write(output);

}

void BinxWriter::write(const char* output)
{

  if( write_xml )
    {
      string nn = "character-8" ;
      toxml->emptyTag(nn) ; 
    }
  tobinary->write(output);
}

void BinxWriter::write(const char& output) 
{
  tobinary->write(output);
  if( write_xml )
    {
      string nn = "character-8" ;
      toxml->emptyTag(nn) ; 
    }

}

void BinxWriter::write(const int& output) 
{
  if( write_xml )
    {
      string nn = "integer-32";
      toxml->emptyTag(nn) ; 
    }
  tobinary->write(output);
}

void BinxWriter::write(const unsigned int& output)
{
  tobinary->write(output);
}

void BinxWriter::write(const short int& output)
{
  tobinary->write(output);
}

void BinxWriter::write(const unsigned short int& output)
{
  tobinary->write(output);
}

void BinxWriter::write(const long int& output)
{
  tobinary->write(output);
}

void BinxWriter::write(const unsigned long int& output)
{
  tobinary->write(output);
}

void BinxWriter::write(const float& output)
{
  if( write_xml )
    {
      string nn = "float-32";
      toxml->emptyTag(nn) ; 
    }

  tobinary->write(output);
}

void BinxWriter::write(const double& output)
{
  if( write_xml )
    {
      string nn = "double-64";
      toxml->emptyTag(nn) ; 
    }
  tobinary->write(output);
}

void BinxWriter::write(const bool& output)
{
  tobinary->write(output);
}

void BinxWriter::writeArray(const char* output, size_t size, size_t nmemb)
{
  tobinary->writeArray(output, size, nmemb) ;
}

//
//  header information 
//
void BinxWriter::write_1D_header(const int& output,const int& dim)
{

  push(*toxml,"arrayFixed"); 
  XMLWriterAPI::AttributeList alist;
  string varName = "QDP_array" ;
  alist.clear();
  alist.push_back(XMLWriterAPI::Attribute("varName",varName ));

  string nn = "integer-32";
  toxml->emptyTag(nn,alist) ; 

  alist.clear();
  alist.push_back(XMLWriterAPI::Attribute("indexTo",dim ));
  string name = "QCD_array" ;
  alist.push_back(XMLWriterAPI::Attribute("name",name ));
  string nnn = "dim";
  toxml->emptyTag(nnn,alist) ; 



  pop(*toxml);  // push(toxml,"dataset");  
  write_xml  = false ;
}



} // namespace QDP;
