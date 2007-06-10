// -*- C++ -*-
// $Id: qdp_io.cc,v 1.24 2007-06-10 15:57:11 edwards Exp $
/*! @file
 * @brief IO support
 */

#include "qdp.h"
#include "qdp_byteorder.h"

namespace QDP
{

  //--------------------------------------------------------------------------------
  // Text reader object support
  TextReader::TextReader() {}

  // Propagate status to all nodes
  bool TextReader::fail()
  {
    bool s;

    if (Layout::primaryNode()) 
      s = getIstream().fail();

    Internal::broadcast(s);
    return s;
  }

  // String reader
  void TextReader::read(std::string& input)
  {
    char *dd_tmp;
    int lleng;

    // Only primary node can grab string
    if (Layout::primaryNode()) 
    {
      getIstream() >> input;
      lleng = input.length() + 1;
    }

    // First must broadcast size of string
    Internal::broadcast(lleng);

    // Now every node can alloc space for string
    dd_tmp = new(nothrow) char[lleng];
    if( dd_tmp == 0x0 ) { 
      QDP_error_exit("Unable to allocate dd_tmp in qdp_io.cc\n");
    }

    if (Layout::primaryNode())
      memcpy(dd_tmp, input.c_str(), lleng);
  
    // Now broadcast char array out to all nodes
    Internal::broadcast((void *)dd_tmp, lleng);

    // All nodes can now grab char array and make a string
    input = dd_tmp;

    // Clean-up and boogie
    delete[] dd_tmp;
  }

  // Readers
  void TextReader::read(char& input) 
  {
    readPrimitive<char>(input);
  }
  void TextReader::read(int& input) 
  {
    readPrimitive<int>(input);
  }
  void TextReader::read(unsigned int& input)
  {
    readPrimitive<unsigned int>(input);
  }
  void TextReader::read(short int& input)
  {
    readPrimitive<short int>(input);
  }
  void TextReader::read(unsigned short int& input)
  {
    readPrimitive<unsigned short int>(input);
  }
  void TextReader::read(long int& input)
  {
    readPrimitive<long int>(input);
  }
  void TextReader::read(unsigned long int& input)
  {
    readPrimitive<unsigned long int>(input);
  }
  void TextReader::read(float& input)
  {
    readPrimitive<float>(input);
  }
  void TextReader::read(double& input)
  {
    readPrimitive<double>(input);
  }
  void TextReader::read(bool& input)
  {
    readPrimitive<bool>(input);
  }

  template< typename T>
  void TextReader::readPrimitive(T& input)
  {
    if (Layout::primaryNode())
      getIstream() >> input;

    // Now broadcast back out to all nodes
    Internal::broadcast(input);
  }

  // Different bindings for read functions
  TextReader& operator>>(TextReader& txt, std::string& input)
  {
    txt.read(input);
    return txt;
  }
  TextReader& operator>>(TextReader& txt, char& input)
  {
    txt.read(input);
    return txt;
  }
  TextReader& operator>>(TextReader& txt, int& input)
  {
    txt.read(input);
    return txt;
  }
  TextReader& operator>>(TextReader& txt, unsigned int& input)
  {
    txt.read(input);
    return txt;
  }
  TextReader& operator>>(TextReader& txt, short int& input)
  {
    txt.read(input);
    return txt;
  }
  TextReader& operator>>(TextReader& txt, unsigned short int& input)
  {
    txt.read(input);
    return txt;
  }
  TextReader& operator>>(TextReader& txt, long int& input)
  {
    txt.read(input);
    return txt;
  }
  TextReader& operator>>(TextReader& txt, unsigned long int& input)
  {
    txt.read(input);
    return txt;
  }
  TextReader& operator>>(TextReader& txt, float& input)
  {
    txt.read(input);
    return txt;
  }
  TextReader& operator>>(TextReader& txt, double& input)
  {
    txt.read(input);
    return txt;
  }
  TextReader& operator>>(TextReader& txt, bool& input)
  {
    txt.read(input);
    return txt;
  }


  //--------------------------------------------------------------------------------
  // Text buffer reader support
  TextBufferReader::TextBufferReader() {}

  TextBufferReader::TextBufferReader(const std::string& s) {open(s);}

  void TextBufferReader::open(const std::string& s) 
  {
    if (Layout::primaryNode())
      f.str() = s;
  }

  // Output the stream
  std::string TextBufferReader::str() const
  {
    std::string s;
    
    if (Layout::primaryNode()) 
      s = f.str();
    
    return s;
  }

  // Close
  TextBufferReader::~TextBufferReader() {}


  //--------------------------------------------------------------------------------
  // Text file reader support
  TextFileReader::TextFileReader() {}

  TextFileReader::TextFileReader(const std::string& p) {open(p);}

  void TextFileReader::open(const std::string& p) 
  {
    if (Layout::primaryNode())
      f.open(p.c_str(),std::ifstream::in);

    if (! is_open())
      QDP_error_exit("failed to open file %s",p.c_str());
  }

  void TextFileReader::close()
  {
    if (is_open()) 
    {
      if (Layout::primaryNode()) 
	f.close();
    }
  }

  // Propagate status to all nodes
  bool TextFileReader::is_open()
  {
    bool s = QDP_isInitialized();

    if (s)
    {
      if (Layout::primaryNode())
	s = f.is_open();

      Internal::broadcast(s);
    }

    return s;
  }

  // Close
  TextFileReader::~TextFileReader() {close();}


  //--------------------------------------------------------------------------------
  // Text object writer support
  TextWriter::TextWriter() {}

  // Propagate status to all nodes
  bool TextWriter::fail()
  {
    bool s;

    if (Layout::primaryNode()) 
      s = getOstream().fail();

    Internal::broadcast(s);
    return s;
  }

  // Different bindings for write functions
  TextWriter& operator<<(TextWriter& txt, const std::string& output)
  {
    txt.write(output);
    return txt;
  }

  TextWriter& operator<<(TextWriter& txt, const char* output)
  {
    txt.write(output);
    return txt;
  }

  TextWriter& operator<<(TextWriter& txt, int output)
  {
    txt.write(output);
    return txt;
  }

  TextWriter& operator<<(TextWriter& txt, unsigned int output)
  {
    txt.write(output);
    return txt;
  }

  TextWriter& operator<<(TextWriter& txt, short int output)
  {
    txt.write(output);
    return txt;
  }

  TextWriter& operator<<(TextWriter& txt, unsigned short int output)
  {
    txt.write(output);
    return txt;
  }

  TextWriter& operator<<(TextWriter& txt, long int output)
  {
    txt.write(output);
    return txt;
  }

  TextWriter& operator<<(TextWriter& txt, unsigned long int output)
  {
    txt.write(output);
    return txt;
  }

  TextWriter& operator<<(TextWriter& txt, float output)
  {
    txt.write(output);
    return txt;
  }

  TextWriter& operator<<(TextWriter& txt, double output)
  {
    txt.write(output);
    return txt;
  }

  TextWriter& operator<<(TextWriter& txt, bool output)
  {
    txt.write(output);
    return txt;
  }


  void TextWriter::write(const string& output)
  {
    writePrimitive<string>(output);
  }

  void TextWriter::write(const char* output)
  {
    write(string(output));
  }

  void TextWriter::write(const char& output) 
  {
    writePrimitive<char>(output);
  }

  void TextWriter::write(const int& output) 
  {
    writePrimitive<int>(output);
  }

  void TextWriter::write(const unsigned int& output)
  {
    writePrimitive<unsigned int>(output);
  }

  void TextWriter::write(const short int& output)
  {
    writePrimitive<short int>(output);
  }

  void TextWriter::write(const unsigned short int& output)
  {
    writePrimitive<unsigned short int>(output);
  }

  void TextWriter::write(const long int& output)
  {
    writePrimitive<long int>(output);
  }

  void TextWriter::write(const unsigned long int& output)
  {
    writePrimitive<unsigned long int>(output);
  }

  void TextWriter::write(const float& output)
  {
    writePrimitive<float>(output);
  }

  void TextWriter::write(const double& output)
  {
    writePrimitive<double>(output);
  }

  void TextWriter::write(const bool& output)
  {
    writePrimitive<bool>(output);
  }

  template< typename T>
  void TextWriter::writePrimitive(const T& output)
  {
    if (Layout::primaryNode())
      getOstream() << output;
  }


  //--------------------------------------------------------------------------------
  // Text buffer writer support
  TextBufferWriter::TextBufferWriter() {}

  TextBufferWriter::TextBufferWriter(const std::string& p) {open(p);}

  void TextBufferWriter::open(const std::string& s)
  {
    if (Layout::primaryNode())
      f.str() = s;
  }

  // Output the stream
  std::string TextBufferWriter::str() const
  {
    std::string s;
    
    if (Layout::primaryNode()) 
      s = f.str();
    
    return s;
  }

  // Close
  TextBufferWriter::~TextBufferWriter() {}


  //--------------------------------------------------------------------------------
  // Text file writer support
  TextFileWriter::TextFileWriter() {}

  TextFileWriter::TextFileWriter(const std::string& p) {open(p);}

  void TextFileWriter::open(const std::string& p)
  {
    if (Layout::primaryNode())
      f.open(p.c_str(),std::ofstream::out | std::ofstream::trunc);

    if (! is_open())
      QDP_error_exit("failed to open file %s",p.c_str());
  }

  void TextFileWriter::close()
  {
    if (is_open()) 
    {
      if (Layout::primaryNode()) 
	f.close();
    }
  }

  // Propagate status to all nodes
  bool TextFileWriter::is_open()
  {
    bool s = QDP_isInitialized();

    if (s)
    {
      if (Layout::primaryNode())
	s = f.is_open();

      Internal::broadcast(s);
    }

    return s;
  }

  void TextFileWriter::flush()
  {
    if (is_open()) 
    {
      if (Layout::primaryNode()) 
	f.flush();
    }
  }

  // Close
  TextFileWriter::~TextFileWriter() {close();}


  //--------------------------------------------------------------------------------
  //! Binary reader support
  BinaryReader::BinaryReader() {}

  // Propagate status to all nodes
  bool BinaryReader::fail()
  {
    bool s;

    if (Layout::primaryNode()) 
      s = getIstream().fail();

    Internal::broadcast(s);
    return s;
  }

  // Wrappers for read functions
  void read(BinaryReader& bin, std::string& input, size_t maxBytes)
  {
    bin.read(input, maxBytes);
  }

  void read(BinaryReader& bin, char& input)
  {
    bin.read(input);
  }

  void read(BinaryReader& bin, int& input)
  {
    bin.read(input);
  }

  void read(BinaryReader& bin, unsigned int& input)
  {
    bin.read(input);
  }

  void read(BinaryReader& bin, short int& input)
  {
    bin.read(input);
  }

  void read(BinaryReader& bin, unsigned short int& input)
  {
    bin.read(input);
  }

  void read(BinaryReader& bin, long int& input)
  {
    bin.read(input);
  }

  void read(BinaryReader& bin, unsigned long int& input)
  {
    bin.read(input);
  }

  void read(BinaryReader& bin, float& input)
  {
    bin.read(input);
  }

  void read(BinaryReader& bin, double& input)
  {
    bin.read(input);
  }

  void read(BinaryReader& bin, bool& input)
  {
    bin.read(input);
  }

  // Different bindings for read functions
  BinaryReader& operator>>(BinaryReader& bin, char& input)
  {
    read(bin, input);
    return bin;
  }

  BinaryReader& operator>>(BinaryReader& bin, int& input)
  {
    read(bin, input);
    return bin;
  }

  BinaryReader& operator>>(BinaryReader& bin, unsigned int& input)
  {
    read(bin, input);
    return bin;
  }

  BinaryReader& operator>>(BinaryReader& bin, short int& input)
  {
    read(bin, input);
    return bin;
  }

  BinaryReader& operator>>(BinaryReader& bin, unsigned short int& input)
  {
    read(bin, input);
    return bin;
  }

  BinaryReader& operator>>(BinaryReader& bin, long int& input)
  {
    read(bin, input);
    return bin;
  }

  BinaryReader& operator>>(BinaryReader& bin, unsigned long int& input)
  {
    read(bin, input);
    return bin;
  }

  BinaryReader& operator>>(BinaryReader& bin, float& input)
  {
    read(bin, input);
    return bin;
  }

  BinaryReader& operator>>(BinaryReader& bin, double& input)
  {
    read(bin, input);
    return bin;
  }

  BinaryReader& operator>>(BinaryReader& bin, bool& input)
  {
    read(bin, input);
    return bin;
  }

  void BinaryReader::read(string& input, size_t maxBytes)
  {
    char *str = new(nothrow) char[maxBytes];
    if( str == 0x0 ) { 
      QDP_error_exit("Couldnt new str in qdp_io.cc\n");
    }

    size_t n;

    if (Layout::primaryNode())
    {
      getIstream().getline(str, maxBytes);
      n = strlen(str);
      setChecksum() = QDPUtil::crc32(getChecksum(), str, n);   // no string terminator
      ++n;
      setChecksum() = QDPUtil::crc32(getChecksum(), "\n", 1);   // account for newline written
    }

    Internal::broadcast(n);
    Internal::broadcast((void *)str, n);

    input = str;
    delete[] str;
  }

  void BinaryReader::read(char& input) 
  {
    readPrimitive<char>(input);
  }

  void BinaryReader::read(int& input) 
  {
    readPrimitive<int>(input);
  }

  void BinaryReader::read(unsigned int& input)
  {
    readPrimitive<unsigned int>(input);
  }

  void BinaryReader::read(short int& input)
  {
    readPrimitive<short int>(input);
  }

  void BinaryReader::read(unsigned short int& input)
  {
    readPrimitive<unsigned short int>(input);
  }

  void BinaryReader::read(long int& input)
  {
    readPrimitive<long int>(input);
  }

  void BinaryReader::read(unsigned long int& input)
  {
    readPrimitive<unsigned long int>(input);
  }

  void BinaryReader::read(float& input)
  {
    readPrimitive<float>(input);
  }

  void BinaryReader::read(double& input)
  {
    readPrimitive<double>(input);
  }

  void BinaryReader::read(bool& input)
  {
    readPrimitive<bool>(input);
  }

  template< typename T>
  void BinaryReader::readPrimitive(T& input)
  {
    readArray((char*)&input, sizeof(T), 1);
  }

  void BinaryReader::readArray(char* input, size_t size, size_t nmemb)
  {
    readArrayPrimaryNode(input, size, nmemb);

    // Now broadcast back out to all nodes
    Internal::broadcast((void*)input, size*nmemb);
  }

  // Read array from the primary node
  void BinaryReader::readArrayPrimaryNode(char* input, size_t size, size_t nmemb)
  {
    if (Layout::primaryNode())
    {
      // Read
      // By default, we expect all data to be in big-endian
      getIstream().read(input, size*nmemb);
      setChecksum() = QDPUtil::crc32(getChecksum(), input, size*nmemb);

      if (! QDPUtil::big_endian())
      {
	// little-endian
	// Swap
	QDPUtil::byte_swap(input, size, nmemb);
      }
    }
  }

  //--------------------------------------------------------------------------------
  //! Binary buffer reader support
  BinaryBufferReader::BinaryBufferReader() {checksum=0;}

  // Construct from a string
  BinaryBufferReader::BinaryBufferReader(const std::string& s) {open(s);}

  BinaryBufferReader::~BinaryBufferReader() {}

  void BinaryBufferReader::open(const std::string& s)
  {
    if (Layout::primaryNode())
      f.str() = s;
  }

  // Output the stream
  std::string BinaryBufferReader::str() const
  {
    std::string s;
    
    if (Layout::primaryNode()) 
      s = f.str();
    
    return s;
  }

  // Get the current checksum
  QDPUtil::n_uint32_t BinaryBufferReader::getChecksum() const
  {
    QDPUtil::n_uint32_t c = checksum;
    Internal::broadcast(c);   // picks up only node 0
    return c;
  }



  //--------------------------------------------------------------------------------
  //! Binary reader support
  BinaryFileReader::BinaryFileReader() {checksum=0;}

  BinaryFileReader::BinaryFileReader(const std::string& p) {checksum=0;open(p);}

  void BinaryFileReader::open(const std::string& p) 
  {
    checksum = 0;
    if (Layout::primaryNode()) 
      f.open(p.c_str(),std::ifstream::in | std::ifstream::binary);

    if (! is_open())
      QDP_error_exit("BinaryFileReader: error opening file %s",p.c_str());
  }

  // Close
  void BinaryFileReader::close()
  {
    if (is_open())
    {
      if (Layout::primaryNode()) 
	f.close();
    }
  }

  // Propagate status to all nodes
  bool BinaryFileReader::is_open()
  {
    bool s = QDP_isInitialized();

    if (s)
    {
      if (Layout::primaryNode())
	s = f.is_open();

      Internal::broadcast(s);
    }

    return s;
  }

  // Shutdown
  BinaryFileReader::~BinaryFileReader() {close();}

  // Get the current checksum
  QDPUtil::n_uint32_t BinaryFileReader::getChecksum() const
  {
    QDPUtil::n_uint32_t c = checksum;
    Internal::broadcast(c);   // picks up only node 0
    return c;
  }


  //--------------------------------------------------------------------------------
  //! Binary writer support
  BinaryWriter::BinaryWriter() {}

  // Propagate status to all nodes
  bool BinaryWriter::fail()
  {
    bool s;

    if (Layout::primaryNode()) 
      s = getOstream().fail();

    Internal::broadcast(s);
    return s;
  }

  // Wrappers for write functions
  void write(BinaryWriter& bin, const std::string& output)
  {
    bin.write(output);
  }

  void write(BinaryWriter& bin, const char* output)
  {
    bin.write(std::string(output));
  }

  void write(BinaryWriter& bin, char output)
  {
    bin.write(output);
  }

  void write(BinaryWriter& bin, int output)
  {
    bin.write(output);
  }

  void write(BinaryWriter& bin, unsigned int output)
  {
    bin.write(output);
  }

  void write(BinaryWriter& bin, short int output)
  {
    bin.write(output);
  }

  void write(BinaryWriter& bin, unsigned short int output)
  {
    bin.write(output);
  }

  void write(BinaryWriter& bin, long int output)
  {
    bin.write(output);
  }

  void write(BinaryWriter& bin, unsigned long int output)
  {
    bin.write(output);
  }

  void write(BinaryWriter& bin, float output)
  {
    bin.write(output);
  }

  void write(BinaryWriter& bin, double output)
  {
    bin.write(output);
  }

  void write(BinaryWriter& bin, bool output)
  {
    bin.write(output);
  }

  // Different bindings for write functions
  BinaryWriter& operator<<(BinaryWriter& bin, const std::string& output)
  {
    write(bin, output);
    return bin;
  }

  BinaryWriter& operator<<(BinaryWriter& bin, const char* output)
  {
    write(bin, output);
    return bin;
  }

  BinaryWriter& operator<<(BinaryWriter& bin, int output)
  {
    write(bin, output);
    return bin;
  }

  BinaryWriter& operator<<(BinaryWriter& bin, unsigned int output)
  {
    write(bin, output);
    return bin;
  }

  BinaryWriter& operator<<(BinaryWriter& bin, short int output)
  {
    write(bin, output);
    return bin;
  }

  BinaryWriter& operator<<(BinaryWriter& bin, unsigned short int output)
  {
    write(bin, output);
    return bin;
  }

  BinaryWriter& operator<<(BinaryWriter& bin, long int output)
  {
    write(bin, output);
    return bin;
  }

  BinaryWriter& operator<<(BinaryWriter& bin, unsigned long int output)
  {
    write(bin, output);
    return bin;
  }

  BinaryWriter& operator<<(BinaryWriter& bin, float output)
  {
    write(bin, output);
    return bin;
  }

  BinaryWriter& operator<<(BinaryWriter& bin, double output)
  {
    write(bin, output);
    return bin;
  }

  BinaryWriter& operator<<(BinaryWriter& bin, bool output)
  {
    write(bin, output);
    return bin;
  }

  void BinaryWriter::write(const string& output)
  {
    // WARNING: CHECK ON NEWLINE IN CHECKSUM
    size_t n = output.length();
    writeArray(output.c_str(), sizeof(char), n);
    write('\n');   // Convention is to write a line terminator
  }

  void BinaryWriter::write(const char* output)
  {
    write(string(output));
  }

  void BinaryWriter::write(const char& output) 
  {
    writePrimitive<char>(output);
  }

  void BinaryWriter::write(const int& output) 
  {
    writePrimitive<int>(output);
  }

  void BinaryWriter::write(const unsigned int& output)
  {
    writePrimitive<unsigned int>(output);
  }

  void BinaryWriter::write(const short int& output)
  {
    writePrimitive<short int>(output);
  }

  void BinaryWriter::write(const unsigned short int& output)
  {
    writePrimitive<unsigned short int>(output);
  }

  void BinaryWriter::write(const long int& output)
  {
    writePrimitive<long int>(output);
  }

  void BinaryWriter::write(const unsigned long int& output)
  {
    writePrimitive<unsigned long int>(output);
  }

  void BinaryWriter::write(const float& output)
  {
    writePrimitive<float>(output);
  }

  void BinaryWriter::write(const double& output)
  {
    writePrimitive<double>(output);
  }

  void BinaryWriter::write(const bool& output)
  {
    writePrimitive<bool>(output);
  }

  template< typename T>
  void BinaryWriter::writePrimitive(const T& output)
  {
    writeArray((const char*)&output, sizeof(T), 1);
  }

  void BinaryWriter::writeArray(const char* output, size_t size, size_t nmemb)
  {
    if (Layout::primaryNode())
    {
      if (QDPUtil::big_endian())
      {
	/* big-endian */
	/* Write */
	setChecksum() = QDPUtil::crc32(getChecksum(), output, size*nmemb);
	getOstream().write(output, size*nmemb);
      }
      else
      {
	/* little-endian */
	/* Swap and write and swap */
	QDPUtil::byte_swap(const_cast<char *>(output), size, nmemb);
	setChecksum() = QDPUtil::crc32(getChecksum(), output, size*nmemb);
	getOstream().write(output, size*nmemb);
	QDPUtil::byte_swap(const_cast<char *>(output), size, nmemb);
      }
    }
  }


  //--------------------------------------------------------------------------------
  //! Binary writer support
  BinaryBufferWriter::BinaryBufferWriter() {checksum=0;}

  // Construct from a string
  BinaryBufferWriter::BinaryBufferWriter(const std::string& s) {open(s);}

  BinaryBufferWriter::~BinaryBufferWriter() {}

  void BinaryBufferWriter::open(const std::string& s)
  {
    if (Layout::primaryNode())
      f.str() = s;
  }

  // Output the stream
  std::string BinaryBufferWriter::str() const
  {
    std::string s;
    
    if (Layout::primaryNode()) 
      s = f.str();
    
    return s;
  }

  //! Get the current checksum
  QDPUtil::n_uint32_t BinaryBufferWriter::getChecksum() const
  {
    QDPUtil::n_uint32_t c = checksum;
    Internal::broadcast(c);   // picks up only node 0
    return c;
  }


  //--------------------------------------------------------------------------------
  //! Binary writer support
  BinaryFileWriter::BinaryFileWriter() {checksum = 0;}

  BinaryFileWriter::BinaryFileWriter(const std::string& p) {checksum = 0; open(p);}

  void BinaryFileWriter::open(const std::string& p) 
  {
    checksum = 0;
    if (Layout::primaryNode()) 
      f.open(p.c_str(),std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);

    if (! is_open())
      QDP_error_exit("BinaryFileWriter: error opening file %s",p.c_str());
  }

  void BinaryFileWriter::close()
  {
    if (is_open())
    {
      if (Layout::primaryNode()) 
	f.close();
    }
  }


  // Propagate status to all nodes
  bool BinaryFileWriter::is_open()
  {
    bool s = QDP_isInitialized();

    if (s)
    {
      if (Layout::primaryNode())
	s = f.is_open();

      Internal::broadcast(s);
    }

    return s;
  }

  void BinaryFileWriter::flush()
  {
    if (is_open()) 
    {
      if (Layout::primaryNode()) 
	f.flush();
    }
  }

  // Close the file
  BinaryFileWriter::~BinaryFileWriter() {close();}

  //! Get the current checksum
  QDPUtil::n_uint32_t BinaryFileWriter::getChecksum() const
  {
    QDPUtil::n_uint32_t c = checksum;
    Internal::broadcast(c);   // picks up only node 0
    return c;
  }

} // namespace QDP
