// $Id: qdp_binx.cc,v 1.3 2004-03-26 12:24:39 mcneile Exp $
//
// QDP data parallel interface to binx writers
//
// I assume that the primary node business is dealt
// with by the lower obects.
//

#include "qdp.h"

QDP_BEGIN_NAMESPACE(QDP);




//-----------------------------------------
//! Binx Binary writer support
BinxWriter::BinxWriter() {}

BinxWriter::BinxWriter(const std::string& p) {open(p);}

void BinxWriter::open(const std::string& p) 
{
  std::string p_xml = p + "_binx.xml" ; 
  tobinary = new BinaryWriter(p); 
  toxml =  new XMLFileWriter(p_xml) ;

  // write some binx stuff
  XMLWriterAPI::AttributeList alist;
  string xmlns = "http://www.edikt.org/binx/2003/06/databinx" ;
  alist.clear();
  alist.push_back(XMLWriterAPI::Attribute("xmlns", xmlns));
  toxml->openTag("databinx", alist);

  alist.clear();
  alist.push_back(XMLWriterAPI::Attribute("src", p));
  toxml->openTag("binx", alist);


  //xml << d.elem(i);
  //toxml.closeTag();



  string dataset = "dataset" ; 
  push(*toxml,dataset);

  if (! is_open())
    QDP_error_exit("BinxWriter: error opening file %s",p.c_str());
}

void BinxWriter::close()
{
  if (is_open())
  {
    pop(*toxml);  // push(toxml,"dataset");
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

//
//  write based methods for binx
//


void BinxWriter::write(const string& output)
{
  string nn = "<character-8 />" ;
  push(*toxml,nn);
  pop(*toxml); 

  tobinary->write(output);
}

void BinxWriter::write(const char* output)
{
  string nn = "<character-8 />" ;
  push(*toxml,nn);
  pop(*toxml); 

  write(string(output));
}

void BinxWriter::write(const char& output) 
{
  tobinary->write(output);
  string nn = "<character-8 />" ;
  push(*toxml,nn);
  pop(*toxml); 

}

void BinxWriter::write(const int& output) 
{
  string nn = "integer-32";
  push(*toxml,nn);
  pop(*toxml); 

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
  string nn = "float-32";
  push(*toxml,nn);
  pop(*toxml); 

  tobinary->write(output);
}

void BinxWriter::write(const double& output)
{
  string nn = "float-64";
  push(*toxml,nn);
  pop(*toxml); 

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


QDP_END_NAMESPACE();
