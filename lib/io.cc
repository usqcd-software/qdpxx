// $Id: io.cc,v 1.7 2003-04-10 18:35:43 edwards Exp $
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
  if (Layout::primaryNode())
  {
    f.open(p,std::ifstream::in);

    cerr << "test open again" << endl;
    if (! f.is_open())
      QDP_error_exit("failed to open file %s",p);
  }

  iop=true;
}

void TextReader::close()
{
  if (is_open()) 
  {
    if (Layout::primaryNode()) 
      f.close();
    iop = false;
  }
}

bool TextReader::is_open() {return iop;}

TextReader::~TextReader() {close();}


//-----------------------------------------
//! text writer support
TextWriter::TextWriter() {iop=false;}

TextWriter::TextWriter(const char* p) {open(p);}

void TextWriter::open(const char* p)
{
  if (Layout::primaryNode()) 
    f.open(p,std::ofstream::out);

  iop=true;
}

void TextWriter::close()
{
  if (is_open()) 
  {
    if (Layout::primaryNode()) 
      f.close();
    iop = false;
  }
}

bool TextWriter::is_open() {return iop;}

TextWriter::~TextWriter() {close();}


//-----------------------------------------
//! text reader support
NmlReader::NmlReader() {iop=false;}

NmlReader::NmlReader(const char* p) {open(p);}

void NmlReader::open(const char* p)
{
  if (Layout::primaryNode()) 
    f.open(p,std::ios_base::in);

  iop=true;
}

void NmlReader::close()
{
  if (is_open()) 
  {
    if (Layout::primaryNode()) 
      f.close();

    iop=true;
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
  if (Layout::primaryNode()) 
    f.open(p,std::ios_base::out);
  iop=true;

  push(*this,"FILE");  // Always start a file with this group
}

void NmlWriter::close()
{
  if (iop) 
  {
    pop(*this);  // Write final end of file group

    if (Layout::primaryNode()) 
      f.close();
    iop = false;
  }
}

bool NmlWriter::is_open() {return iop;}

NmlWriter::~NmlWriter() {close();}

//! Push a namelist group 
NmlWriter& push(NmlWriter& nml, const string& s)
{
  if (Layout::primaryNode()) 
    nml.get() << "&" << s << endl; 

  return nml;
}

//! Pop a namelist group
NmlWriter& pop(NmlWriter& nml)
{
  if (Layout::primaryNode()) 
    nml.get() << "&END\n"; 

  return nml;
}

//! Write a comment
NmlWriter& operator<<(NmlWriter& nml, const char* s)
{
  if (Layout::primaryNode()) 
    nml.get() << "! " << s << endl; 

  return nml;
}



//-----------------------------------------
//! Binary reader support
BinaryReader::BinaryReader() {iop=false; f = NULL;}

BinaryReader::BinaryReader(const char* p) {open(p);}

void BinaryReader::open(const char* p) 
{
  if (Layout::primaryNode()) 
  {
    if ((f = fopen(p,"rb")) == NULL)
    {
      cerr << "BinaryReader: error opening file: " << p << endl;
      QDP_error_exit("BinaryReader: error opening file %s",p);
    }
  }

  iop = true;
}

void BinaryReader::close()
{
  if (iop)
  {
    if (Layout::primaryNode()) 
      fclose(f);

    iop = false;
  }
}


bool BinaryReader::is_open() {return iop;}

BinaryReader::~BinaryReader() {close();}



//-----------------------------------------
//! Binary writer support
BinaryWriter::BinaryWriter() {iop=false; f = NULL;}

BinaryWriter::BinaryWriter(const char* p) {open(p);}

void BinaryWriter::open(const char* p) 
{
  if (Layout::primaryNode()) 
  {
    if ((f = fopen(p,"wb")) == NULL)
    {
      cerr << "BinaryWriter: error opening file: " << p << endl;
      QDP_error_exit("BinaryWriter: error opening file %s",p);
    }
  }

  iop = true;
}

void BinaryWriter::close()
{
  if (iop)
  {
    if (Layout::primaryNode()) 
      fclose(f);

    iop = false;
  }
}


bool BinaryWriter::is_open() {return iop;}

BinaryWriter::~BinaryWriter() {close();}



QDP_END_NAMESPACE();
