// $Id: io.cc,v 1.4 2002-10-26 02:25:27 edwards Exp $
//
// QDP data parallel interface
//

#include "qdp.h"

QDP_BEGIN_NAMESPACE(QDP);

//-----------------------------------------
//! text reader support
TextReader::TextReader() {iop=false;}

TextReader::TextReader(const char* p) {open(p);}

void TextReader::open(const char* p)
{
  f.open(p,std::ios_base::in);
  iop=true;
}

void TextReader::close()
{
  if (iop)
  {
    f.close();
    iop = false;
  }
}

#if 0
template<class T>
TextReader& TextReader::operator>>(T& d)
{
  f >> d; 
  return *this;
}
#endif


bool TextReader::is_open() {return iop;}

TextReader::~TextReader() {close();}


//-----------------------------------------
//! text writer support
TextWriter::TextWriter() {iop=false;}

TextWriter::TextWriter(const char* p) {open(p);}

void TextWriter::open(const char* p)
{
  f.open(p,std::ios_base::out);
  iop=true;
}

void TextWriter::close()
{
  if (iop) 
  {
    f.close();
    iop = false;
  }
}

#if 0
template<class T>
TextWriter& TextWriter::operator<<(const T& d)
{
  f << d; 
  return *this;
}
#endif



bool TextWriter::is_open() {return iop;}

TextWriter::~TextWriter() {close();}


//-----------------------------------------
//! text reader support
NmlReader::NmlReader() {iop=false;}

NmlReader::NmlReader(const char* p) {open(p);}

void NmlReader::open(const char* p)
{
  f.open(p,std::ios_base::in);
  iop = true;
}

void NmlReader::close()
{
  if (iop)
  {
    f.close();
    iop = false;
  }
}

bool NmlReader::is_open() {return iop;}

NmlReader::~NmlReader() {close();}


//-----------------------------------------
//! text writer support
NmlWriter::NmlWriter() {iop=false;}

NmlWriter::NmlWriter(const char* p) {open(p);}

void NmlWriter::open(const char* p)
{
  f.open(p,std::ios_base::out);
  iop=true;

  push(*this,"FILE");  // Always start a file with this group
}

void NmlWriter::close()
{
  if (iop) 
  {
    pop(*this);  // Write final end of file group

    f.close();
    iop = false;
  }
}

bool NmlWriter::is_open() {return iop;}

NmlWriter::~NmlWriter() {close();}

//! Push a namelist group 
NmlWriter& push(NmlWriter& nml, const string& s)
{
  nml.get() << "&" << s << endl; return nml;
}

//! Pop a namelist group
NmlWriter& pop(NmlWriter& nml) {nml.get() << "&END\n"; return nml;}

//! Write a comment
NmlWriter& operator<<(NmlWriter& nml, const char* s)
{
  nml.get() << "! " << s << endl; 
  return nml;
}



//-----------------------------------------
//! Binary reader support
BinaryReader::BinaryReader() {f = NULL;}

BinaryReader::BinaryReader(const char* p) {open(p);}

void BinaryReader::open(const char* p) 
{
  if ((f = fopen(p,"rb")) == NULL)
  {
    cerr << "BinaryReader: error opening file: " << p << endl;
    SZ_ERROR("BinaryReader: error opening file");
  }

  iop = true;
}

void BinaryReader::close()
{
  if (iop)
  {
    fclose(f);
    iop = false;
  }
}


// Read End-Of-Record mark
BinaryReader& BinaryReader::eor()
{
  fgetc(f); 
  return *this;
}


bool BinaryReader::is_open() {return iop;}

BinaryReader::~BinaryReader() {close();}



//-----------------------------------------
//! Binary writer support
BinaryWriter::BinaryWriter() {f = NULL;}

BinaryWriter::BinaryWriter(const char* p) {open(p);}

void BinaryWriter::open(const char* p) 
{
  if ((f = fopen(p,"wb")) == NULL)
  {
    cerr << "BinaryWriter: error opening file: " << p << endl;
    SZ_ERROR("BinaryWriter: error opening file");
  }

  iop = true;
}

void BinaryWriter::close()
{
  if (iop)
  {
    fclose(f);
    iop = false;
  }
}


// Write End-Of-Record mark
BinaryWriter& BinaryWriter::eor()
{
  fprintf(f,"\005"); 
  return *this;
}


bool BinaryWriter::is_open() {return iop;}

BinaryWriter::~BinaryWriter() {close();}



QDP_END_NAMESPACE();
