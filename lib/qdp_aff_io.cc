// $Id: qdp_aff_io.cc,v 1.1.2.2 2008-03-16 02:40:04 edwards Exp $
/*! @file
 * @brief AFF IO support
 */

#include "qdp.h"
#include "qdp_aff_io.h"
#include "qdp_aff_imp.h"

namespace QDP 
{
  using std::string;

  //--------------------------------------------------------------------------------
  // Just for a test to make sure a final object is complete.
  // This is not to run, but just compile
  // Anonymous namespace
  namespace
  {
    void test_affreader()
    {
      AFFReader aff("fred");

      int a;
      read(aff, "/a", a);

      AFFReader aff_sub(aff, "/a");
      read(aff, "b", a);

      multi1d<Real> bar;
      read(aff, "/bar", bar);
    }

    void test_afffilewriter()
    {
      AFFFileWriter aff("foo");

      int a = 0;
      multi1d<Real> bar(5);
      bar = zero;

      push(aff, "root");
      write(aff, "a", a);
      write(aff, "bar", bar);
      pop(aff);
    }
 
    void test_sub_reader()
    {
      AFFReader aff("fred");

      int a;

      TreeReader tree(aff, "/a");
      read(tree, "a", a);

      multi1d<Real> bar;
      read(tree, "/bar", bar);
    }

    void test_treearraywriter()
    {
      AFFFileWriter aff("foo");
      TreeArrayWriter tree_array(aff);

      int a = 0;
      multi1d<Real> bar(5);
      bar = zero;

      push(tree_array, "root");
      for(int i=0; i < bar.size(); ++i)
      {
	pushElem(tree_array);
	write(tree_array, "bar",bar[0]);
	popElem(tree_array);
      }
      pop(tree_array);
    }
 
  }
  


  //--------------------------------------------------------------------------------
  // AFF classes
  // AFF reader class
  AFFReader::AFFReader()
  {
    registerObject(new LAFFReaderImp());
  }

  AFFReader::AFFReader(const std::string& filename)
  {
    registerObject(new LAFFReaderImp(filename));
  }

  AFFReader::AFFReader(AFFReader& old, const string& xpath)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    registerObject(old.clone(xpath));
  }

  AFFReader::~AFFReader() {}

  // Return the implementation
  LAFFReaderImp& AFFReader::getAFFReader() const
  {
    return dynamic_cast<LAFFReaderImp&>(getTreeReader());
  }

  void AFFReader::open(const string& filename)
  {
    getAFFReader().open(filename);
  }

  // Clone a reader
  void AFFReader::open(AFFReader& old, const string& xpath)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    getAFFReader().open(old.getAFFReader(), xpath);
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
  }

  void AFFReader::close()
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    getAFFReader().close();
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
  }

  // Internal cloning function
  TreeReaderImp* AFFReader::clone(const std::string& xpath)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    return new LAFFReaderImp(getAFFReader(), xpath);
  }


  bool AFFReader::is_open()
  {
    return getAFFReader().is_open();
  }

  // Return the entire contents of the Reader as a TreeRep
  void AFFReader::treeRep(TreeRep& output)
  {
    QDP_error_exit("AFFReader::treeRep function not implemented in a derived class");
  }
        
  //! Return the current context as a TreeRep
  void AFFReader::treeRepCurrentContext(TreeRep& output)
  {
    QDP_error_exit("AFFReader::treeRepCurrentContext function not implemented in a derived class");
  }
        
  //! Count the number of occurances from the Xpath query
  bool AFFReader::exist(const std::string& xpath)
  {
    return getAFFReader().exist(xpath);
  }

  //! Count the number of occurances from the Xpath query
  int AFFReader::countArrayElem()
  {
    return getAFFReader().countArrayElem();
  }

  // Return tag for array element n
  std::string AFFReader::arrayElem(int n) const
  {
    return getAFFReader().arrayElem(n);
  }


  //--------------------------------------------------------------------------------
  // AFF Writer to a file
  AFFFileWriter::AFFFileWriter()
  {
    file_writer = new AFFFileWriterImp();
    initP = true;
  }

  // Constructor from a filename
  AFFFileWriter::AFFFileWriter(const std::string& filename)
  {
    file_writer = new AFFFileWriterImp(filename);
    initP = true;
  }

  AFFFileWriter::~AFFFileWriter() 
  {
    if (initP)
      delete file_writer;
  }

  // The actual implementation
  AFFFileWriterImp& AFFFileWriter::getAFFFileWriter() const
  {
    if (! initP)
    {
      QDPIO::cerr << "AFFFileWriter is not initialized" << endl;
      QDP_abort(1);
    }

    return *file_writer;
  }

  void AFFFileWriter::open(const std::string& filename)
  {
    dynamic_cast<AFFFileWriterImp&>(getAFFFileWriter()).open(filename);
  }

  void AFFFileWriter::close()
  {
    dynamic_cast<AFFFileWriterImp&>(getAFFFileWriter()).close();
  }

  // Propagate status to all nodes
  bool AFFFileWriter::is_open()
  {
    return dynamic_cast<AFFFileWriterImp&>(getAFFFileWriter()).is_open();
  }

  // Flush the buffer
  void AFFFileWriter::flush()
  {
    getAFFFileWriter().flush();
  }

  // Propagate status to all nodes
  bool AFFFileWriter::fail() const
  {
    return dynamic_cast<AFFFileWriterImp&>(getAFFFileWriter()).fail();
  }

  void AFFFileWriter::openStruct(const string& tagname)
  {
    getAFFFileWriter().openStruct(tagname);
  }

  void AFFFileWriter::closeStruct()
  {
    getAFFFileWriter().closeStruct();
  }

  void AFFFileWriter::openTag(const string& tagname)
  {
    getAFFFileWriter().openTag(tagname);
  }

  void AFFFileWriter::closeTag()
  {
    getAFFFileWriter().closeTag();
  }

  // Return tag for array element n
  std::string AFFFileWriter::arrayElem(int n) const
  {
    return getAFFFileWriter().arrayElem(n);
  }

  // Time to build a telephone book of basic primitives
  // Overloaded Writer Functions
  void AFFFileWriter::write(const std::string& tagname, const string& output)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    getAFFFileWriter().write(tagname,output);
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
  }
  void AFFFileWriter::write(const string& tagname, const int& output)
  {
    getAFFFileWriter().write(tagname,output);
  }
  void AFFFileWriter::write(const string& tagname, const unsigned int& output)
  {
    getAFFFileWriter().write(tagname,output);
  }
  void AFFFileWriter::write(const string& tagname, const short int& output)
  {
    getAFFFileWriter().write(tagname,output);
  }
  void AFFFileWriter::write(const string& tagname, const unsigned short int& output)
  {
    getAFFFileWriter().write(tagname,output);
  }
  void AFFFileWriter::write(const string& tagname, const long int& output)
  {
    getAFFFileWriter().write(tagname,output);
  }
  void AFFFileWriter::write(const string& tagname, const unsigned long int& output)
  {
    getAFFFileWriter().write(tagname,output);
  }
  void AFFFileWriter::write(const string& tagname, const float& output)
  {
    getAFFFileWriter().write(tagname,output);
  }
  void AFFFileWriter::write(const string& tagname, const double& output)
  {
    getAFFFileWriter().write(tagname,output);
  }
  void AFFFileWriter::write(const string& tagname, const bool& output)
  {
    getAFFFileWriter().write(tagname,output);
  }
   
  void AFFFileWriter::write(const std::string& tagname, const multi1d<int>& output)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    getAFFFileWriter().write(tagname,output);
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
  }
  void AFFFileWriter::write(const std::string& tagname, const multi1d<unsigned int>& output)
  {
    getAFFFileWriter().write(tagname,output);
  }
  void AFFFileWriter::write(const std::string& tagname, const multi1d<short int>& output)
  {
    getAFFFileWriter().write(tagname,output);
  }
  void AFFFileWriter::write(const std::string& tagname, const multi1d<unsigned short int>& output)
  {
    getAFFFileWriter().write(tagname,output);
  }
  void AFFFileWriter::write(const std::string& tagname, const multi1d<long int>& output)
  {
    getAFFFileWriter().write(tagname,output);
  }
  void AFFFileWriter::write(const std::string& tagname, const multi1d<unsigned long int>& output)
  {
    getAFFFileWriter().write(tagname,output);
  }
  void AFFFileWriter::write(const std::string& tagname, const multi1d<float>& output)
  {
    getAFFFileWriter().write(tagname,output);
  }
  void AFFFileWriter::write(const std::string& tagname, const multi1d<double>& output)
  {
    getAFFFileWriter().write(tagname,output);
  }
  void AFFFileWriter::write(const std::string& tagname, const multi1d<bool>& output)
  {
    getAFFFileWriter().write(tagname,output);
  }
  void AFFFileWriter::write(const std::string& tagname, const multi1d<Integer>& output)
  {
    getAFFFileWriter().write(tagname,output);
  }
  void AFFFileWriter::write(const std::string& tagname, const multi1d<Real32>& output)
  {
    getAFFFileWriter().write(tagname,output);
  }
  void AFFFileWriter::write(const std::string& tagname, const multi1d<Real64>& output)
  {
    getAFFFileWriter().write(tagname,output);
  }
  void AFFFileWriter::write(const std::string& tagname, const multi1d<Boolean>& output)
  {
    getAFFFileWriter().write(tagname,output);
  }


  //--------------------------------------------------------------------------------
} // namespace QDP;
