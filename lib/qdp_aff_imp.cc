// $Id: qdp_aff_imp.cc,v 1.1.2.1 2008-03-15 14:28:56 edwards Exp $
//
/*! @file
 * @brief AFF IO support implementation
 */

#include <sstream>

#include "qdp.h"
#include "qdp_aff_imp.h"

//#include <lhpc-aff.h>
#include "/usr/local/share/lhpc/include/lhpc-aff.h"


namespace QDP 
{
  using std::string;

  //--------------------------------------------------------------------------------
  // Just for a test to make sure a final object is complete.
  // This is not to run, but just compile
  // Anonymous namespace
  namespace
  {
    void test_afffilewriter()
    {
      AFFFileWriterImp aff("foo");

      int a = 0;

      aff.write("/a/b", a);
    }

  }
  

  //--------------------------------------------------------------------------------
  // Some utils
  namespace
  {
    /* Rules for chdir_path & mkdir_path (same, except for treatment of absent keys)
       - starting '/' : ignore [rw]_node, start from root
       - double slash '//': ignore repeated slashes
       - empty/NULL key_path -?
       - NULL r_node -?
    */
    AffNode_s* chdirPath(AffReader_s* r, 
			 AffNode_s* r_node, const string& kpath)
    {
      if (NULL == r || NULL == r_node)
      {
	QDPIO::cout << __func__ << ": invalid reader or node" << endl;
	QDP_abort(1);
      }
      if ("" == kpath)
	return NULL;

      AffTree_s*   tree   = aff_reader_tree(r);
      AffSTable_s* stable = aff_reader_stable(r);

      if( '/' == kpath[0] )
	r_node = aff_reader_root(r);

      size_t beg = 0, end = 0 ;
      string key;
      while (end < kpath.size() && r_node != NULL)
      {
	end = kpath.find('/', beg);
	if (beg < end)
	{
	  key = string(kpath, beg, end - beg);
	  r_node = aff_node_chdir(tree, stable, r_node, 0, key.c_str());
	}
	beg = end + 1;
      }
      return r_node;
    }

    //! Return the node corresponding to the path. If it does not exist, throw an exception.
    AffNode_s* getNodePath(AffReader_s* r, 
			   AffNode_s* r_node, const string& xpath)
    {
      AffNode_s* node = chdirPath(r, r_node, xpath);
      
      if (node == 0)
      {
	std::ostringstream os;
	os << "LAFFReader: invalid xpath= " << xpath << endl;
	throw os.str();
      }

      if (aff_reader_errstr(r) != 0)
      {
	QDPIO::cerr << "LAFFReader: Error: " << aff_reader_errstr(r) << endl;
	QDP_abort(1);
      }

      return node;
    }
    
  } // anonymous namespace
  


  //--------------------------------------------------------------------------------
  // AFF classes
  // AFF reader class
  // Return tag for array element n
  std::string AFFReaderImp::arrayElem(int n) const
  {
    std::ostringstream os;
    os << "elem[" << n+1 << "]";
    return os.str();
  }


  //--------------------------------------------------------------------------------
  // AFF classes
  // AFF reader class
  LAFFReaderImp::LAFFReaderImp() 
  {
    input_stream = 0;
    current_dir  = 0;
    initP = derivedP = false;
  }

  LAFFReaderImp::LAFFReaderImp(const std::string& filename)
  {
    initP = derivedP = false;
    input_stream = 0;
    current_dir  = 0;
    open(filename);
  }

  LAFFReaderImp::LAFFReaderImp(LAFFReaderImp& old, const string& xpath)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    initP = derivedP = false;
    input_stream = 0;
    current_dir  = 0;
    open(old, xpath);
  }

  LAFFReaderImp::~LAFFReaderImp() 
  {
    close();
  }

  // Clone a reader from a different path
  LAFFReaderImp* LAFFReaderImp::clone(const std::string& xpath)
  {
    return new LAFFReaderImp(*this, xpath);
  }

  AffReader_s* LAFFReaderImp::getIstream() const
  {
    if (! initP)
    {
      QDPIO::cerr << "LAFFReaderImp is not initialized" << endl;
      QDP_abort(1);
    }

    return input_stream;
  }


  void LAFFReaderImp::open(const string& filename)
  {
    if (! QDP_isInitialized())
    {
      QDPIO::cerr << "AFFFileReader: QDP is not initialized: error opening write file = " << filename << endl;
      QDP_abort(1);
    }

    if (initP)
    {
      QDPIO::cerr << "AFFFileReader: a file is already opened: error opening write file = " << filename << endl;
      QDP_abort(1);
    }

    if (Layout::primaryNode())
    {
      input_stream = aff_reader(filename.c_str());
      if (aff_reader_errstr(input_stream) != 0)
      {
	QDPIO::cerr << "LAFFReader: Error opening read file = " << filename << endl;
	QDP_abort(1);
      }

      current_dir = aff_reader_root(input_stream);
    }
    else
    {
      input_stream = 0;
      current_dir  = 0;
    }
    initP = true;
    derivedP = false;
  }

  // Clone a reader
  void LAFFReaderImp::open(LAFFReaderImp& old, const std::string& xpath)
  {
    if (! old.initP)
    {
      QDPIO::cerr << "AFFFileReader: original reader is not initialized" << endl;
      QDP_abort(1);
    }

    if (Layout::primaryNode())
    {
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << "  xpath=" << xpath << endl;
      input_stream = old.input_stream;
      current_dir = getNodePath(input_stream, old.current_dir, xpath);
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << "  xpath=" << xpath << endl;
    }
    else
    {
      input_stream = 0;
      current_dir  = 0;
    }
    initP = true;
    derivedP = true;
  }

  void LAFFReaderImp::close()
  {
    if (is_open()) 
    {
      if (! is_derived()) 
      {
	if (Layout::primaryNode()) 
	{
	  aff_reader_close(input_stream);
	}
      }
    }
      
    initP = false;
    input_stream = 0;
    current_dir  = 0;
  }

  bool LAFFReaderImp::is_open()
  {
    bool s = QDP_isInitialized();

    if (s)
      s = initP;

    return s;
  }

  bool LAFFReaderImp::is_derived() const {return derivedP;}

  // Check for errors
  void LAFFReaderImp::checkError() const
  {
    if (Layout::primaryNode()) 
    {
      if (aff_reader_errstr(input_stream) != 0)
      {
	QDPIO::cerr << "LAFFReaderImp: Error: " << aff_reader_errstr(input_stream) << endl;
	QDP_abort(1);
      }
    }
  }

  // Check for errors
  void LAFFReaderImp::notImplemented(const std::string& func) const
  {
    if (Layout::primaryNode()) 
    {
      QDPIO::cerr << "LAFFReaderImp: function not implemented: " << func << endl;
      QDP_abort(1);
    }
  }


  // Overloaded Reader Functions
  void LAFFReaderImp::read(const std::string& xpath, string& result)
  {
    if (Layout::primaryNode())
    {
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << "  xpath=" << xpath << endl;
      AffNode_s* new_node = getNodePath(getIstream(), current_dir, xpath);
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
      uint32_t num = aff_node_size(new_node);
      checkError();
      char *new_str = new char[num+1];
      if (aff_node_get_char(getIstream(), new_node, new_str, num) != 0)
      {
	checkError();
      }
      new_str[num] = '\0';
      result = new_str;
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
      delete[] new_str;
    }

    // broadcast string
    Internal::broadcast_str(result);
  }
  void LAFFReaderImp::read(const std::string& xpath, int& result)
  {
    if (Layout::primaryNode())
    {
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << "  xpath=" << xpath << endl;
      AffNode_s* new_node = getNodePath(getIstream(), current_dir, xpath);
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
      uint32_t num = aff_node_size(new_node);
      checkError();
      if (num != 1)
      {
	QDPIO::cerr << "LAFFReaderImp: expected size 1 for tag=" << xpath << " but found size= "
		    << num << endl;
	QDP_abort(1);
      }
      uint32_t res;
      if (aff_node_get_int(getIstream(), new_node, &res, 1) != 0)
      {
	checkError();
      }
      result = res;
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    }

    // Now broadcast back out to all nodes
    Internal::broadcast(result);
  }
  void LAFFReaderImp::read(const std::string& xpath, unsigned int& result)
  {
    notImplemented(__func__);
  }
  void LAFFReaderImp::read(const std::string& xpath, short int& result)
  {
    notImplemented(__func__);
  }
  void LAFFReaderImp::read(const std::string& xpath, unsigned short int& result)
  {
    notImplemented(__func__);
  }
  void LAFFReaderImp::read(const std::string& xpath, long int& result)
  {
    notImplemented(__func__);
  }
  void LAFFReaderImp::read(const std::string& xpath, unsigned long int& result)
  {
    notImplemented(__func__);
  }
  void LAFFReaderImp::read(const std::string& xpath, float& result)
  {
    notImplemented(__func__);
  }
  void LAFFReaderImp::read(const std::string& xpath, double& result)
  {
    if (Layout::primaryNode())
    {
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << "  xpath=" << xpath << endl;
      AffNode_s* new_node = getNodePath(getIstream(), current_dir, xpath);
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
      uint32_t num = aff_node_size(new_node);
      checkError();
      if (num != 1)
      {
	QDPIO::cerr << "LAFFReaderImp: expected size 1 for tag=" << xpath << " but found size= "
		    << num << endl;
	QDP_abort(1);
      }
      if (aff_node_get_double(getIstream(), new_node, &result, 1) != 0)
      {
	checkError();
      }
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    }

    // Now broadcast back out to all nodes
    Internal::broadcast(result);
  }
  void LAFFReaderImp::read(const std::string& xpath, bool& result)
  {
    notImplemented(__func__);
  }
   
  // Return the entire contents of the Reader as a TreeRep
  void LAFFReaderImp::treeRep(TreeRep& output)
  {
    QDP_error_exit("LAFFReaderImp::treeRep function not implemented in a derived class");
  }
        
  //! Return the current context as a TreeRep
  void LAFFReaderImp::treeRepCurrentContext(TreeRep& output)
  {
    QDP_error_exit("LAFFReaderImp::treeRepCurrentContext function not implemented in a derived class");
  }
        
  //! Count the number of occurances from the Xpath query
  bool LAFFReaderImp::exist(const std::string& xpath)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    AffNode_s* node = chdirPath(getIstream(), current_dir, xpath);
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    bool found = (node == 0) ? false : true;
    QDPIO::cout << __func__ << ": exist(" << xpath << ") = " << found << endl;
    return found;
  }


  // Count the number of occurances from the Xpath query
  int LAFFReaderImp::count(const std::string& xpath)
  {
    notImplemented(__func__);
    int n=0;
    
    // Now broadcast back out to all nodes
    Internal::broadcast(n);
    return n;
  }

  //! Count the number of occurances from the Xpath query
  int LAFFReaderImp::countArrayElem()
  {
    notImplemented(__func__);
  }

  void LAFFReaderImp::read(const std::string& xpath, multi1d<int>& result)
  {
    if (Layout::primaryNode())
    {
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << "  xpath=" << xpath << endl;
      AffNode_s* new_node = getNodePath(getIstream(), current_dir, xpath);
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
      uint32_t num = aff_node_size(new_node);
      checkError();
      uint32_t* res = new uint32_t[num];
      if (aff_node_get_int(getIstream(), new_node, res, num) != 0)
      {
	checkError();
      }
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
      result.resize(num);
      for(int i=0; i < result.size(); ++i)
	result[i] = res[i];
      delete[] res;
    }

    // Now broadcast back out to all nodes
    Internal::broadcast(result);
  }
  void LAFFReaderImp::read(const std::string& xpath, multi1d<unsigned int>& result)
  {
    notImplemented(xpath);
  }
  void LAFFReaderImp::read(const std::string& xpath, multi1d<short int>& result)
  {
    notImplemented(xpath);
  }
  void LAFFReaderImp::read(const std::string& xpath, multi1d<unsigned short int>& result)
  {
    notImplemented(xpath);
  }
  void LAFFReaderImp::read(const std::string& xpath, multi1d<long int>& result)
  {
    notImplemented(xpath);
  }
  void LAFFReaderImp::read(const std::string& xpath, multi1d<unsigned long int>& result)
  {
    notImplemented(xpath);
  }
  void LAFFReaderImp::read(const std::string& xpath, multi1d<float>& result)
  {
    notImplemented(xpath);
  }
  void LAFFReaderImp::read(const std::string& xpath, multi1d<double>& result)
  {
    if (Layout::primaryNode())
    {
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << "  xpath=" << xpath << endl;
      AffNode_s* new_node = getNodePath(getIstream(), current_dir, xpath);
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
      uint32_t num = aff_node_size(new_node);
      checkError();
      double* res = new double[num];
      if (aff_node_get_double(getIstream(), new_node, res, num) != 0)
      {
	checkError();
      }
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
      result.resize(num);
      result = *res;
      delete[] res;
    }

    // Now broadcast back out to all nodes
    Internal::broadcast(result);
  }
  void LAFFReaderImp::read(const std::string& xpath, multi1d<bool>& result)
  {
    notImplemented(xpath);
  }

  void LAFFReaderImp::read(const std::string& xpath, multi1d<Integer>& result)
  {
    notImplemented(xpath);
  }
  void LAFFReaderImp::read(const std::string& xpath, multi1d<Real32>& result)
  {
    notImplemented(xpath);
  }
  void LAFFReaderImp::read(const std::string& xpath, multi1d<Real64>& result)
  {
    notImplemented(xpath);
  }
  void LAFFReaderImp::read(const std::string& xpath, multi1d<Boolean>& result)
  {
    notImplemented(xpath);
  }



  //--------------------------------------------------------------------------------
  // AFF Writer to a file
  AFFFileWriterImp::AFFFileWriterImp() 
  {
    output_stream = 0;
    current_node  = 0;
    initP = false;
  }

  AFFFileWriterImp::AFFFileWriterImp(const std::string& filename)
  {
    initP = false;
    output_stream = 0;
    current_node  = 0;
    open(filename);
  }

  AFFFileWriterImp::~AFFFileWriterImp() 
  {
    close();
  }

  AffWriter_s* AFFFileWriterImp::getOstream() const
  {
    if (! initP)
    {
      QDPIO::cerr << "AFFFileWriter is not initialized" << endl;
      QDP_abort(1);
    }

    return output_stream;
  }

  void AFFFileWriterImp::open(const std::string& filename)
  {
    if (! QDP_isInitialized())
    {
      QDPIO::cerr << "AFFFileWriter: QDP is not initialized: error opening write file = " << filename << endl;
      QDP_abort(1);
    }

    if (initP)
    {
      QDPIO::cerr << "AFFFileWriter: a file is already opened: error opening write file = " << filename << endl;
      QDP_abort(1);
    }

    if (Layout::primaryNode())
    {
      output_stream = aff_writer(filename.c_str());

      if (aff_writer_errstr(output_stream) != 0)
      {
	QDPIO::cerr << "AFFFileWriter: Error opening write file = " << filename << endl;
	QDP_abort(1);
      }

      current_node = aff_writer_root(output_stream);
    }
    else
    {
      output_stream = 0;
      current_node  = 0;
    }
    initP = true;
  }


  void AFFFileWriterImp::close()
  {
    if (is_open()) 
    {
      if (Layout::primaryNode()) 
      {
	const char* succ = aff_writer_close(output_stream);
	if (succ != 0)
	{
	  QDPIO::cerr << "AFFFileWriterImp: Error close file: error=" << succ << endl;
	  QDP_abort(1);
	}
      }
    }

    initP = false;
    output_stream = 0;
    current_node  = 0;
  }


  // Propagate status to all nodes
  bool AFFFileWriterImp::is_open()
  {
    bool s = QDP_isInitialized();

    if (s)
      s = initP;

    return s;
  }


  // Flush the buffer
  void AFFFileWriterImp::flush()
  {
    // NOP
  }

  // Propagate status to all nodes
  bool AFFFileWriterImp::fail() const
  {
    bool s = QDP_isInitialized();

    if (s)
    {
      if (Layout::primaryNode()) 
      {
	if (aff_writer_errstr(output_stream) != 0)
	{
	  s = true;
	}
      }

      Internal::broadcast(s);
    }

    return s;
  }


  // Check for errors
  void AFFFileWriterImp::checkError() const
  {
    if (Layout::primaryNode()) 
    {
      if (aff_writer_errstr(output_stream) != 0)
      {
	QDPIO::cerr << "AFFFileWriterImp: Error: " << aff_writer_errstr(output_stream) << endl;
	QDP_abort(1);
      }
    }
  }


  // Check for errors
  void AFFFileWriterImp::notImplemented(const std::string& func) const
  {
    if (Layout::primaryNode()) 
    {
      QDPIO::cerr << "AFFFileWriterImp: function not implemented: " << func << endl;
      QDP_abort(1);
    }
  }


  void AFFFileWriterImp::openStruct(const string& tagname)
  {
    openTag(tagname);
  }

  void AFFFileWriterImp::closeStruct()
  {
    closeTag();
  }

  void AFFFileWriterImp::openTag(const string& tagname)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    if (Layout::primaryNode())
    {
//      string tag_full_path = createPath(tagname);
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << " tagname=XX" << tagname << "XX" << endl;
      string current_tag = aff_symbol_name(aff_node_name(current_node));
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << " current=XX" << current_tag << "XX" << endl;
      current_node = aff_writer_mkdir(getOstream(), current_node, tagname.c_str());
    }
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
  }

  void AFFFileWriterImp::closeTag()
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    if (Layout::primaryNode())
    {
      current_node = aff_node_parent(current_node);
    }
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
  }

  // Return tag for array element n
  std::string AFFFileWriterImp::arrayElem(int n) const
  {
    std::ostringstream os;
    os << "elem" << "." << n+1;
    return os.str();
  }


  // Time to build a telephone book of basic primitives
  // Overloaded Writer Functions
  void AFFFileWriterImp::write(const std::string& tagname, const string& output)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    if (Layout::primaryNode())
    {
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
      openTag(tagname);
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
      if (aff_node_put_char(getOstream(), current_node, 
			    output.c_str(), output.size()) != 0)
      {
	checkError();
      }
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
      closeTag();
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    }
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
  }
  void AFFFileWriterImp::write(const string& tagname, const int& output)
  {
    if (Layout::primaryNode())
    {
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
      openTag(tagname);
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
      uint32_t out = output;
      if (aff_node_put_int(getOstream(), current_node, &out, 1) != 0)
      {
	checkError();
      }
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
      closeTag();
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    }
  }
  void AFFFileWriterImp::write(const string& tagname, const unsigned int& output)
  {
    notImplemented(__func__);
  }
  void AFFFileWriterImp::write(const string& tagname, const short int& output)
  {
    notImplemented(__func__);
  }
  void AFFFileWriterImp::write(const string& tagname, const unsigned short int& output)
  {
    notImplemented(__func__);
  }
  void AFFFileWriterImp::write(const string& tagname, const long int& output)
  {
    notImplemented(__func__);
  }
  void AFFFileWriterImp::write(const string& tagname, const unsigned long int& output)
  {
    notImplemented(__func__);
  }
  void AFFFileWriterImp::write(const string& tagname, const float& output)
  {
    notImplemented(__func__);
  }
  void AFFFileWriterImp::write(const string& tagname, const double& output)
  {
    if (Layout::primaryNode())
    {
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
      openTag(tagname);
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
      if (aff_node_put_double(getOstream(), current_node, &output, 1) != 0)
      {
	checkError();
      }
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
      closeTag();
      QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    }
  }
  void AFFFileWriterImp::write(const string& tagname, const bool& output)
  {
    notImplemented(__func__);
  }
   

#if 0
  // Write an array of basic types
  template<typename T>
  void AFFFileWriterImp::writeArrayPrimitive(const std::string& s, const multi1d<T>& s1)
  {
    notImplemented(__func__);

    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    std::ostringstream output;

//    if (s1.size() > 0)
//    {
//      output << s1[0];
//      for(int index=1; index < s1.size(); index++) 
//	output << " " << s1[index];
//    }
    
    // Write the array - do not use a normal string write
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
  }
#endif


  void AFFFileWriterImp::write(const std::string& tagname, const multi1d<int>& output)
  {
    notImplemented(__func__);
  }
  void AFFFileWriterImp::write(const std::string& tagname, const multi1d<unsigned int>& output)
  {
    notImplemented(__func__);
  }
  void AFFFileWriterImp::write(const std::string& tagname, const multi1d<short int>& output)
  {
    notImplemented(__func__);
  }
  void AFFFileWriterImp::write(const std::string& tagname, const multi1d<unsigned short int>& output)
  {
    notImplemented(__func__);
  }
  void AFFFileWriterImp::write(const std::string& tagname, const multi1d<long int>& output)
  {
    notImplemented(__func__);
  }
  void AFFFileWriterImp::write(const std::string& tagname, const multi1d<unsigned long int>& output)
  {
    notImplemented(__func__);
  }
  void AFFFileWriterImp::write(const std::string& tagname, const multi1d<float>& s1)
  {
    notImplemented(__func__);
  }
  void AFFFileWriterImp::write(const std::string& tagname, const multi1d<double>& s1)
  {
    notImplemented(__func__);
  }
  void AFFFileWriterImp::write(const std::string& tagname, const multi1d<bool>& output)
  {
    notImplemented(__func__);
  }

  void AFFFileWriterImp::write(const std::string& tagname, const multi1d<Integer>& output)
  {
    notImplemented(__func__);
  }
  void AFFFileWriterImp::write(const std::string& tagname, const multi1d<Real32>& s1)
  {
    notImplemented(__func__);
  }
  void AFFFileWriterImp::write(const std::string& tagname, const multi1d<Real64>& s1)
  {
    notImplemented(__func__);
  }
  void AFFFileWriterImp::write(const std::string& tagname, const multi1d<Boolean>& output)
  {
    notImplemented(__func__);
  }



} // namespace QDP;
