// -*- C++ -*-
// $Id: qdp_stdio.cc,v 1.2 2003-10-03 02:56:17 edwards Exp $

/*! @file
 * @brief Parallel version of stdio
 *
 * These are the QDP parallel versions of the STL cout,cerr,cin.
 * The user can control which nodes are involved in output/input.
 */


#include "qdp.h"

QDP_BEGIN_NAMESPACE(QDP);


//-----------------------------------------
//! stdin support
StandardInputStream::StandardInputStream() {}

StandardInputStream::StandardInputStream(std::streambuf* b) : is(b) {}

void StandardInputStream::rdbuf(std::streambuf* b)
{
  getIstream().rdbuf(b);
}

// Propagate status to all nodes
bool StandardInputStream::fail()
{
  bool s;

  if (Layout::primaryNode()) 
    s = getIstream().fail();

  Internal::broadcast(s);
  return s;
}

StandardInputStream::~StandardInputStream() {}


// String reader
StandardInputStream& StandardInputStream::operator>>(std::string& input)
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
  dd_tmp = new char[lleng];
  if (Layout::primaryNode())
    memcpy(dd_tmp, input.c_str(), lleng);
  
  // Now broadcast char array out to all nodes
  Internal::broadcast((void *)dd_tmp, lleng);

  // All nodes can now grab char array and make a string
  input = dd_tmp;

  // Clean-up and boogie
  delete[] dd_tmp;

  return *this;
}

// Readers
StandardInputStream& StandardInputStream::read(char& input) 
{
  return readPrimitive<char>(input);
}
StandardInputStream& StandardInputStream::read(int& input) 
{
  return readPrimitive<int>(input);
}
StandardInputStream& StandardInputStream::read(unsigned int& input)
{
  return readPrimitive<unsigned int>(input);
}
StandardInputStream& StandardInputStream::read(short int& input)
{
  return readPrimitive<short int>(input);
}
StandardInputStream& StandardInputStream::read(unsigned short int& input)
{
  return readPrimitive<unsigned short int>(input);
}
StandardInputStream& StandardInputStream::read(long int& input)
{
  return readPrimitive<long int>(input);
}
StandardInputStream& StandardInputStream::read(unsigned long int& input)
{
  return readPrimitive<unsigned long int>(input);
}
StandardInputStream& StandardInputStream::read(float& input)
{
  return readPrimitive<float>(input);
}
StandardInputStream& StandardInputStream::read(double& input)
{
  return readPrimitive<double>(input);
}
StandardInputStream& StandardInputStream::read(bool& input)
{
  return readPrimitive<bool>(input);
}

template< typename T>
StandardInputStream& StandardInputStream::readPrimitive(T& input)
{
  if (Layout::primaryNode())
    getIstream() >> input;

  // Now broadcast back out to all nodes
  Internal::broadcast(input);

  return *this;
}


//-----------------------------------------
//! stdout support
StandardOutputStream::StandardOutputStream() {}

StandardOutputStream::StandardOutputStream(std::streambuf* b)
{
  getOstream().rdbuf(b);
}

void StandardOutputStream::rdbuf(std::streambuf* b)
{
  getOstream().rdbuf(b);
}

void StandardOutputStream::flush()
{
  if (is_open()) 
  {
    if (Layout::primaryNode()) 
      getOstream().flush();
  }
}

// Propagate status to all nodes
bool StandardOutputStream::fail()
{
  bool s;

  if (Layout::primaryNode()) 
    s = getOstream().fail();

  Internal::broadcast(s);
  return s;
}

StandardOutputStream::~StandardOutputStream() {close();}


StandardOutputStream& StandardOutputStream::operator<<(const string& output)
{
  if (Layout::primaryNode())
    getOstream() << output;

  return *this;
}

StandardOutputStream& StandardOutputStream::operator<<(char* output)
{
  if (Layout::primaryNode())
    getOstream() << output;

  return *this;
}

StandardOutputStream& StandardOutputStream::operator<<(char output) 
{
  return writePrimitive<char>(output);
}

StandardOutputStream& StandardOutputStream::operator<<(int output) 
{
  return writePrimitive<int>(output);
}

StandardOutputStream& StandardOutputStream::operator<<(unsigned int output)
{
  return writePrimitive<unsigned int>(output);
}

StandardOutputStream& StandardOutputStream::operator<<(short int output)
{
  return writePrimitive<short int>(output);
}

StandardOutputStream& StandardOutputStream::operator<<(unsigned short int output)
{
  return writePrimitive<unsigned short int>(output);
}

StandardOutputStream& StandardOutputStream::operator<<(long int output)
{
  return writePrimitive<long int>(output);
}

StandardOutputStream& StandardOutputStream::operator<<(unsigned long int output)
{
  return writePrimitive<unsigned long int>(output);
}

StandardOutputStream& StandardOutputStream::operator<<(float output)
{
  return writePrimitive<float>(output);
}

StandardOutputStream& StandardOutputStream::operator<<(double output)
{
  return writePrimitive<double>(output);
}

StandardOutputStream& StandardOutputStream::operator<<(bool output)
{
  return writePrimitive<bool>(output);
}

template<typename T>
StandardOutputStream& StandardOutputStream::writePrimitive(T output)
{
  if (Layout::primaryNode())
    getOstream() << output;

  return *this;
}


QDP_END_NAMESPACE();
