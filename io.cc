// $Id: io.cc,v 1.1 2002-10-01 16:24:41 edwards Exp $
//
// QDP data parallel interface
//

#include "qdp.h"

QDP_BEGIN_NAMESPACE(QDP);

//-----------------------------------------
//! text reader support
TextReader::TextReader() {}

TextReader::TextReader(const char* p) {open(p);}

void TextReader::open(const char* p) {f.open(p,std::ios_base::in);}

void TextReader::close()
{
  if (is_open()) 
    f.close();
}

bool TextReader::is_open() {return f.is_open();}

TextReader::~TextReader() {close();}


//-----------------------------------------
//! text writer support
TextWriter::TextWriter() {}

TextWriter::TextWriter(const char* p) {open(p);}

void TextWriter::open(const char* p) {return f.open(p,std::ios_base::out);}

void TextWriter::close()
{
  if (is_open()) 
    f.close();
}

bool TextWriter::is_open() {return f.is_open();}

TextWriter::~TextWriter() {close();}

//-----------------------------------------
//! Binary writer support
BinaryWriter::BinaryWriter() {}

BinaryWriter::BinaryWriter(const char* p) {open(p);}

void BinaryWriter::open(const char* p) 
{
  f.open(p,std::ios_base::out|std::ios_base::binary);
}

void BinaryWriter::close()
{
  if (is_open()) 
    f.close();
}

bool BinaryWriter::is_open() {return f.is_open();}

BinaryWriter::~BinaryWriter() {close();}



QDP_END_NAMESPACE();
