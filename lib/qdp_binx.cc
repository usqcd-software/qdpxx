// $Id: qdp_binx.cc,v 1.1 2004-03-25 13:58:57 mcneile Exp $
//
// QDP data parallel interface to binx writers
//

#include "qdp.h"

QDP_BEGIN_NAMESPACE(QDP);




//-----------------------------------------
//! Binx Binary writer support
BinxWriter::BinxWriter() {}

BinxWriter::BinxWriter(const std::string& p) {open(p);}

void BinxWriter::open(const std::string& p) 
{
  if (Layout::primaryNode()) 
    f.open(p.c_str(),std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);

  if (! is_open())
    QDP_error_exit("BinxWriter: error opening file %s",p.c_str());
}

void BinxWriter::close()
{
  if (is_open())
  {
    if (Layout::primaryNode()) 
      f.close();
  }
}


// Propagate status to all nodes
bool BinxWriter::is_open()
{
  bool s;

  if (Layout::primaryNode()) 
    s = f.is_open();

  Internal::broadcast(s);
  return s;
}

void BinxWriter::flush()
{
  if (is_open()) 
  {
    if (Layout::primaryNode()) 
      f.flush();
  }
}

// Propagate status to all nodes
bool BinxWriter::fail()
{
  bool s;

  if (Layout::primaryNode()) 
    s = f.fail();

  Internal::broadcast(s);
  return s;
}

BinxWriter::~BinxWriter() {close();}

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

void BinxWriter::write(const string& output)
{
  size_t n = output.length();
  writeArray(output.c_str(), sizeof(char), n);
  write('\n');   // Convention is to write a line terminator
}

void BinxWriter::write(const char* output)
{
  write(string(output));
}

void BinxWriter::write(const char& output) 
{
  writePrimitive<char>(output);
}

void BinxWriter::write(const int& output) 
{
  writePrimitive<int>(output);
}

void BinxWriter::write(const unsigned int& output)
{
  writePrimitive<unsigned int>(output);
}

void BinxWriter::write(const short int& output)
{
  writePrimitive<short int>(output);
}

void BinxWriter::write(const unsigned short int& output)
{
  writePrimitive<unsigned short int>(output);
}

void BinxWriter::write(const long int& output)
{
  writePrimitive<long int>(output);
}

void BinxWriter::write(const unsigned long int& output)
{
  writePrimitive<unsigned long int>(output);
}

void BinxWriter::write(const float& output)
{
  writePrimitive<float>(output);
}

void BinxWriter::write(const double& output)
{
  writePrimitive<double>(output);
}

void BinxWriter::write(const bool& output)
{
  writePrimitive<bool>(output);
}

template< typename T>
void BinxWriter::writePrimitive(const T& output)
{
  writeArray((const char*)&output, sizeof(T), 1);
}

void BinxWriter::writeArray(const char* output, size_t size, size_t nmemb)
{
  if (Layout::primaryNode())
  {
    //    if (QDPUtil::big_endian())

    if (true)
    {
      /* big-endian */
      /* Write */
      getOstream().write(output, size*nmemb);
    }
    else
    {
      /* little-endian */
      /* Swap and write and swap */
      //      QDPUtil::byte_swap(const_cast<char *>(output), size, nmemb);
      getOstream().write(output, size*nmemb);
      //QDPUtil::byte_swap(const_cast<char *>(output), size, nmemb);
    }
  }
}


QDP_END_NAMESPACE();
