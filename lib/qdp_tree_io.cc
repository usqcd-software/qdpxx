// $Id: qdp_tree_io.cc,v 1.1.2.5 2008-03-17 04:03:47 edwards Exp $
//
/*! @file
 * @brief Tree IO support
 */

#include "qdp.h"
#include "qdp_tree_io.h"
#include "qdp_tree_imp.h"

namespace QDP 
{

  using std::string;

  //--------------------------------------------------------------------------------
  // Tree classes

  // Create a TreeReader according to a certain sub-type
  TreeReader::TreeReader(const std::string& id)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    tree_use = createTreeReader(id);
    initP    = true;
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
  }
  
  // Tree reader class
  // Default constructor
  TreeReader::TreeReader()
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    tree_use = 0;
    initP    = false;
  }
  
  // Clone a reader but with a possibly different path
  TreeReader::TreeReader(TreeReader& old, const std::string& xpath)
  { 
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    tree_use = old.clone(xpath);
    initP    = true;
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
  }
  
  // Destructor
  TreeReader::~TreeReader() 
  {
    if (initP)
      delete tree_use;
  }

  // Internal cloning function
  TreeReaderImp* TreeReader::clone(const std::string& xpath)
  {
    TreeReaderImp* a_new_thingy = 0;
    if (initP)
    {
      a_new_thingy = tree_use->clone(xpath);
    }
    else
    {
      QDPIO::cerr << "TreeReader::Clone function not implemented in a derived class" << endl;
      QDP_abort(1);
    }
    return a_new_thingy;
  }

  // Register object with base class
  void TreeReader::registerObject(TreeReaderImp* obj)
  {
    if (initP)
    {
      QDPIO::cerr << "TreeReader already has an object registered" << endl;
      QDP_abort(1);
    }

    tree_use = obj;
    initP    = true;
  }

  // Return the implementation
  TreeReaderImp& TreeReader::getTreeReader() const
  {
    if (! initP)
    {
      QDPIO::cerr << "TreeReader is not initialized" << endl;
      QDP_abort(1);
    }

    return *tree_use;
  }


  // Return the entire contents of the Reader as a TreeRep
  void TreeReader::treeRep(TreeRep& output)
  {
    getTreeReader().treeRep(output);
  }
        
  //! Return the current context as a TreeRep
  void TreeReader::treeRepCurrentContext(TreeRep& output)
  {
    getTreeReader().treeRepCurrentContext(output);
  }
        
  //! Count the number of occurances from the Xpath query
  bool TreeReader::exist(const std::string& xpath)
  {
    return getTreeReader().exist(xpath);
  }

  //! Count the number of occurances from the Xpath query
//  int TreeReader::count(const std::string& xpath)
//  {
//    return getTreeReader().count(xpath);
//  }

  //! Count the number of occurances from the Xpath query
  int TreeReader::countArrayElem()
  {
    return getTreeReader().countArrayElem();
  }

  // Return tag for array element n
  std::string TreeReader::arrayElem(int n) const
  {
    return getTreeReader().arrayElem(n);
  }

  /* So should these, there is just a lot of overloading */
  void TreeReader::read(const std::string& xpath, std::string& result)
  {
    getTreeReader().read(xpath, result);
  }
  void TreeReader::read(const std::string& xpath, int& result)
  {
    getTreeReader().read(xpath, result);
  }
  void TreeReader::read(const std::string& xpath, unsigned int& result)
  {
    getTreeReader().read(xpath, result);
  }
  void TreeReader::read(const std::string& xpath, short int& result)
  {
    getTreeReader().read(xpath, result);
  }
  void TreeReader::read(const std::string& xpath, unsigned short int& result)
  {
    getTreeReader().read(xpath, result);
  }
  void TreeReader::read(const std::string& xpath, long int& result)
  {
    getTreeReader().read(xpath, result);
  }
  void TreeReader::read(const std::string& xpath, unsigned long int& result)
  {
    getTreeReader().read(xpath, result);
  }
  void TreeReader::read(const std::string& xpath, float& result)
  {
    getTreeReader().read(xpath, result);
  }
  void TreeReader::read(const std::string& xpath, double& result)
  {
    getTreeReader().read(xpath, result);
  }
  void TreeReader::read(const std::string& xpath, bool& result)
  {
    getTreeReader().read(xpath, result);
  }

  /* Overloadings of primitive(elemental) array objects */
  void TreeReader::read(const std::string& xpath, multi1d<int>& result)
  {
    getTreeReader().read(xpath, result);
  }
  void TreeReader::read(const std::string& xpath, multi1d<unsigned int>& result)
  {
    getTreeReader().read(xpath, result);
  }
  void TreeReader::read(const std::string& xpath, multi1d<short int>& result)
  {
    getTreeReader().read(xpath, result);
  }
  void TreeReader::read(const std::string& xpath, multi1d<unsigned short int>& result)
  {
    getTreeReader().read(xpath, result);
  }
  void TreeReader::read(const std::string& xpath, multi1d<long int>& result)
  {
    getTreeReader().read(xpath, result);
  }
  void TreeReader::read(const std::string& xpath, multi1d<unsigned long int>& result)
  {
    getTreeReader().read(xpath, result);
  }
  void TreeReader::read(const std::string& xpath, multi1d<float>& result)
  {
    getTreeReader().read(xpath, result);
  }
  void TreeReader::read(const std::string& xpath, multi1d<double>& result)
  {
    getTreeReader().read(xpath, result);
  }
  void TreeReader::read(const std::string& xpath, multi1d<bool>& result)
  {
    getTreeReader().read(xpath, result);
  }


  // Overloaded Reader Functions
  void read(TreeReader& tree, const std::string& xpath, string& result)
  {
    tree.read(xpath, result);
  }
  void read(TreeReader& tree, const std::string& xpath, int& result)
  {
    tree.read(xpath, result);
  }
  void read(TreeReader& tree, const std::string& xpath, unsigned int& result)
  {
    tree.read(xpath, result);
  }
  void read(TreeReader& tree, const std::string& xpath, short int& result)
  {
    tree.read(xpath, result);
  }
  void read(TreeReader& tree, const std::string& xpath, unsigned short int& result)
  {
    tree.read(xpath, result);
  }
  void read(TreeReader& tree, const std::string& xpath, long int& result)
  {
    tree.read(xpath, result);
  }
  void read(TreeReader& tree, const std::string& xpath, unsigned long int& result)
  {
    tree.read(xpath, result);
  }
  void read(TreeReader& tree, const std::string& xpath, float& result)
  {
    tree.read(xpath, result);
  }
  void read(TreeReader& tree, const std::string& xpath, double& result)
  {
    tree.read(xpath, result);
  }
  void read(TreeReader& tree, const std::string& xpath, bool& result)
  {
    tree.read(xpath, result);
  }
   

  template<>
  void read(TreeReader& tree, const std::string& xpath, multi1d<int>& result)
  {
    tree.read(xpath, result);
  }
  template<>
  void read(TreeReader& tree, const std::string& xpath, multi1d<unsigned int>& result)
  {
    tree.read(xpath, result);
  }
  template<>
  void read(TreeReader& tree, const std::string& xpath, multi1d<short int>& result)
  {
    tree.read(xpath, result);
  }
  template<>
  void read(TreeReader& tree, const std::string& xpath, multi1d<unsigned short int>& result)
  {
    tree.read(xpath, result);
  }
  template<>
  void read(TreeReader& tree, const std::string& xpath, multi1d<long int>& result)
  {
    tree.read(xpath, result);
  }
  template<>
  void read(TreeReader& tree, const std::string& xpath, multi1d<unsigned long int>& result)
  {
    tree.read(xpath, result);
  }
  template<>
  void read(TreeReader& tree, const std::string& xpath, multi1d<float>& result)
  {
    tree.read(xpath, result);
  }
  template<>
  void read(TreeReader& tree, const std::string& xpath, multi1d<double>& result)
  {
    tree.read(xpath, result);
  }
  template<>
  void read(TreeReader& tree, const std::string& xpath, multi1d<bool>& result)
  {
    tree.read(xpath, result);
  }


  // Read a Tree multi1d element
  template<typename T>
  void readArrayPrimitive(TreeReader& tree, const std::string& xpath, multi1d<T>& result)
  {
    multi1d<int> res;
    tree.read(xpath, res);
    result.resize(res.size());
    for(int i=0; i < res.size(); ++i)
      result[i] = res[i];
  }
  template<>
  void read(TreeReader& tree, const std::string& xpath, multi1d<Integer>& result)
  {
    readArrayPrimitive(tree, xpath, result);
  }
  template<>
  void read(TreeReader& tree, const std::string& xpath, multi1d<Real32>& result)
  {
    readArrayPrimitive(tree, xpath, result);
  }
  template<>
  void read(TreeReader& tree, const std::string& xpath, multi1d<Real64>& result)
  {
    readArrayPrimitive(tree, xpath, result);
  }
  template<>
  void read(TreeReader& tree, const std::string& xpath, multi1d<Boolean>& result)
  {
    readArrayPrimitive(tree, xpath, result);
  }


  //--------------------------------------------------------------------------------
  // Tree writer base class
  // Create a TreeWriter according to a certain sub-type
  TreeWriter::TreeWriter(const std::string& id)
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    tree_use = createTreeWriter(id);
    initP    = true;
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
  }
  
  // Tree reader class
  // Default constructor
  TreeWriter::TreeWriter()
  {
    QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
    tree_use = 0;
    initP    = false;
  }
  
  // Destructor
  TreeWriter::~TreeWriter() 
  {
    if (initP)
      delete tree_use;
  }

  // Register object with base class
  void TreeWriter::registerObject(TreeWriterImp* obj)
  {
    if (initP)
    {
      QDPIO::cerr << "TreeWriter already has an object registered" << endl;
      QDP_abort(1);
    }

    tree_use = obj;
    initP    = true;
  }

  // Return the implementation
  TreeWriterImp& TreeWriter::getTreeWriter() const
  {
    if (! initP)
    {
      QDPIO::cerr << "TreeWriter is not initialized" << endl;
      QDP_abort(1);
    }

    return *tree_use;
  }

  void TreeWriter::openStruct(const string& tagname)
  {
    getTreeWriter().openStruct(tagname);
  }

  void TreeWriter::closeStruct()
  {
    getTreeWriter().closeStruct();
  }

  void TreeWriter::openTag(const string& tagname)
  {
    getTreeWriter().openTag(tagname);
  }

  void TreeWriter::closeTag()
  {
    getTreeWriter().closeTag();
  }

  // Return tag for array element n
  std::string TreeWriter::arrayElem(int n) const
  {
    return getTreeWriter().arrayElem(n);
  }

  //! Write the number of array elements written
  void TreeWriter::writeArraySize(int size)
  {
    getTreeWriter().writeArraySize(size);
  }

  // Flush the buffer
  void TreeWriter::flush()
  {
    getTreeWriter().flush();
  }

  // Time to build a telephone book of basic primitives
  // Overloaded Writer Functions
  void TreeWriter::write(const string& tagname, const string& output)
  {
    getTreeWriter().write(tagname,output);
  }
  void TreeWriter::write(const string& tagname, const int& output)
  {
    getTreeWriter().write(tagname,output);
  }
  void TreeWriter::write(const string& tagname, const unsigned int& output)
  {
    getTreeWriter().write(tagname,output);
  }
  void TreeWriter::write(const string& tagname, const short int& output)
  {
    getTreeWriter().write(tagname,output);
  }
  void TreeWriter::write(const string& tagname, const unsigned short int& output)
  {
    getTreeWriter().write(tagname,output);
  }
  void TreeWriter::write(const string& tagname, const long int& output)
  {
    getTreeWriter().write(tagname,output);
  }
  void TreeWriter::write(const string& tagname, const unsigned long int& output)
  {
    getTreeWriter().write(tagname,output);
  }
  void TreeWriter::write(const string& tagname, const float& output)
  {
    getTreeWriter().write(tagname,output);
  }
  void TreeWriter::write(const string& tagname, const double& output)
  {
    getTreeWriter().write(tagname,output);
  }
  void TreeWriter::write(const string& tagname, const bool& output)
  {
    getTreeWriter().write(tagname,output);
  }
   
  void TreeWriter::write(const std::string& tagname, const multi1d<int>& output)
  {
    getTreeWriter().write(tagname,output);
  }
  void TreeWriter::write(const std::string& tagname, const multi1d<unsigned int>& output)
  {
    getTreeWriter().write(tagname,output);
  }
  void TreeWriter::write(const std::string& tagname, const multi1d<short int>& output)
  {
    getTreeWriter().write(tagname,output);
  }
  void TreeWriter::write(const std::string& tagname, const multi1d<unsigned short int>& output)
  {
    getTreeWriter().write(tagname,output);
  }
  void TreeWriter::write(const std::string& tagname, const multi1d<long int>& output)
  {
    getTreeWriter().write(tagname,output);
  }
  void TreeWriter::write(const std::string& tagname, const multi1d<unsigned long int>& output)
  {
    getTreeWriter().write(tagname,output);
  }
  void TreeWriter::write(const std::string& tagname, const multi1d<float>& output)
  {
    getTreeWriter().write(tagname,output);
  }
  void TreeWriter::write(const std::string& tagname, const multi1d<double>& output)
  {
    getTreeWriter().write(tagname,output);
  }
  void TreeWriter::write(const std::string& tagname, const multi1d<bool>& output)
  {
    getTreeWriter().write(tagname,output);
  }


  // Push a group name
  void push(TreeWriter& tree, const string& s) {tree.openStruct(s);}

  // Pop a group name
  void pop(TreeWriter& tree) {tree.closeStruct();}

//THIS NEEDS FIXING AND THOUGHT
  // Write something from a reader
  void write(TreeWriter& tree, const std::string& s, const TreeReader& d)
  {
    QDP_error_exit("write(TreeWriter) from a TreeReader not implemented");
  }
//
//  TreeWriter& operator<<(TreeWriter& tree, const TreeReader& d)
//  {
//    ostringstream os;
//    const_cast<TreeReader&>(d).printCurrentContext(os);
//    tree.writeTree(os.str());
//    return tree;
//  }
//
  // Write something from a TreeBufferWriter
  void write(TreeWriter& tree, const std::string& s, const TreeBufferWriter& d)
  {
    QDP_error_exit("write(TreeWriter) from a TreeBufferWriter not implemented");
  }

//  TreeWriter& operator<<(TreeWriter& tree, const TreeBufferWriter& d)
//  {
//    tree.writeTree(const_cast<TreeBufferWriter&>(d).printCurrentContext());
//    return tree;
//  }

  void write(TreeWriter& tree, const string& s, const string& d)
  {
    tree.write(s, d);
  }
  void write(TreeWriter& tree, const string& s, const char* d)
  {
    tree.write(s, string(d));
  }
  void write(TreeWriter& tree, const string& s, const char& d)
  {
    tree.write(s, d);
  }
  void write(TreeWriter& tree, const string& s, const int& d)
  {
    tree.write(s, d);
  }
  void write(TreeWriter& tree, const string& s, const unsigned int& d)
  {
    tree.write(s, d);
  }
  void write(TreeWriter& tree, const string& s, const short int& d)
  {
    tree.write(s, d);
  }
  void write(TreeWriter& tree, const string& s, const unsigned short int& d)
  {
    tree.write(s, d);
  }
  void write(TreeWriter& tree, const string& s, const long int& d)
  {
    tree.write(s, d);
  }
  void write(TreeWriter& tree, const string& s, const unsigned long int& d)
  {
    tree.write(s, d);
  }
  void write(TreeWriter& tree, const string& s, const float& d)
  {
    tree.write(s, d);
  }
  void write(TreeWriter& tree, const string& s, const double& d)
  {
    tree.write(s, d);
  }
  void write(TreeWriter& tree, const string& s, const bool& d)
  {
    tree.write(s, d);
  }


  template<>
  void write(TreeWriter& tree, const std::string& s, const multi1d<int>& output)
  {
    tree.write(s, output);
  }
  template<>
  void write(TreeWriter& tree, const std::string& s, const multi1d<unsigned int>& output)
  {
    tree.write(s, output);
  }
  template<>
  void write(TreeWriter& tree, const std::string& s, const multi1d<short int>& output)
  {
    tree.write(s, output);
  }
  template<>
  void write(TreeWriter& tree, const std::string& s, const multi1d<unsigned short int>& output)
  {
    tree.write(s, output);
  }
  template<>
  void write(TreeWriter& tree, const std::string& s, const multi1d<long int>& output)
  {
    tree.write(s, output);
  }
  template<>
  void write(TreeWriter& tree, const std::string& s, const multi1d<unsigned long int>& output)
  {
    tree.write(s, output);
  }
  template<>
  void write(TreeWriter& tree, const std::string& s, const multi1d<float>& output)
  {
    tree.write(s, output);
  }
  template<>
  void write(TreeWriter& tree, const std::string& s, const multi1d<double>& output)
  {
    tree.write(s, output);
  }
  template<>
  void write(TreeWriter& tree, const std::string& s, const multi1d<bool>& output)
  {
    tree.write(s, output);
  }

   // Write a Tree multi1d element
   template<>
   void write(TreeWriter& tree, const std::string& s, const multi1d<Integer>& output)
   {
     multi1d<int> out;
     out.resize(output.size());
     for(int i=0; i < out.size(); ++i)
       out[i] = toInt(output[i]);
     tree.write(s, out);
   }
   template<>
   void write(TreeWriter& tree, const std::string& s, const multi1d<Real32>& output)
   {
     multi1d<int> out;
     out.resize(output.size());
     for(int i=0; i < out.size(); ++i)
       out[i] = toFloat(output[i]);
     tree.write(s, out);
   }
   template<>
   void write(TreeWriter& tree, const std::string& s, const multi1d<Real64>& output)
   {
     multi1d<int> out;
     out.resize(output.size());
     for(int i=0; i < out.size(); ++i)
       out[i] = toDouble(output[i]);
     tree.write(s, out);
   }
   template<>
   void write(TreeWriter& tree, const std::string& s, const multi1d<Boolean>& output)
   {
     multi1d<int> out;
     out.resize(output.size());
     for(int i=0; i < out.size(); ++i)
       out[i] = toBool(output[i]);
     tree.write(s, out);
   }



 #if 0
   //--------------------------------------------------------------------------------
   // Tree writer to a buffer
   TreeBufferWriter::TreeBufferWriter() {}

   TreeBufferWriter::TreeBufferWriter(const TreeRep& s) {open(s);}

   void TreeBufferWriter::open(const TreeRep& s) 
   {
     // LOTS OF STUFF NEEDS TO HAPPEN HERE
     QDPIO::cerr << __PRETTY_FUNCTION__ << ": not implemented" << endl;
     QDP_abort(1);
   }

   void TreeBufferWriter::treeRep(TreeRep& s) const
   {
     s = getTreeRep();
   }

   void TreeBufferWriter::treeRepCurrentContext(TreeRep& s) const 
   {
     s = getTreeRep();
   }

   TreeBufferWriter::~TreeBufferWriter() {}


   // These are all forwardings to the underlying TreeWriter

   // Start a structur
   void TreeBufferWriter::openStruct(const string& tagname)
   {
     getTreeWriter().openStruct(tagname);
   }

   void TreeBufferWriter::closeStruct()
   {
     getTreeWriter().closeStruct();
   }

   // Writes an opening Tree tag    
   void TreeBufferWriter::openTag(const std::string& tagname)
   {
     return getTreeWriter().openTag(tagname);
   }

   // Closes a tag
   void TreeBufferWriter::closeTag()
   {
     return getTreeWriter().closeTag();
   }

   // Return tag for array element n
   std::string TreeBufferWriter::arrayElem(int n) const
   {
     return getTreeWriter().arrayElem(n);
   }

   // Flush the output. Maybe a nop
   void TreeBufferWriter::flush()
   {
     getTreeWriter().flush();
   }

   // Time to build a telephone book of basic primitives
   // Overloaded Writer Functions
   void TreeBufferWriter::write(const std::string& tagname, const string& output)
   {
     return getTreeWriter().write(tagname, output);
   }
   void TreeBufferWriter::write(const string& tagname, const int& output)
   {
     return getTreeWriter().write(tagname, output);
   }
   void TreeBufferWriter::write(const string& tagname, const unsigned int& output)
   {
     return getTreeWriter().write(tagname, output);
   }
   void TreeBufferWriter::write(const string& tagname, const short int& output)
   {
     return getTreeWriter().write(tagname, output);
   }
   void TreeBufferWriter::write(const string& tagname, const unsigned short int& output)
   {
     return getTreeWriter().write(tagname, output);
   }
   void TreeBufferWriter::write(const string& tagname, const long int& output)
   {
     return getTreeWriter().write(tagname, output);
   }
   void TreeBufferWriter::write(const string& tagname, const unsigned long int& output)
   {
     return getTreeWriter().write(tagname, output);
   }
   void TreeBufferWriter::write(const string& tagname, const float& output)
   {
     return getTreeWriter().write(tagname, output);
   }
   void TreeBufferWriter::write(const string& tagname, const double& output)
   {
     return getTreeWriter().write(tagname, output);
   }
   void TreeBufferWriter::write(const string& tagname, const bool& output)
   {
     return getTreeWriter().write(tagname, output);
   }

   void TreeBufferWriter::write(const std::string& tagname, const multi1d<int>& output)
   {
     return getTreeWriter().write(tagname, output);
   }
   void TreeBufferWriter::write(const std::string& tagname, const multi1d<unsigned int>& output)
   {
     return getTreeWriter().write(tagname, output);
   }
   void TreeBufferWriter::write(const std::string& tagname, const multi1d<short int>& output)
   {
     return getTreeWriter().write(tagname, output);
   }
   void TreeBufferWriter::write(const std::string& tagname, const multi1d<unsigned short int>& output)
   {
     return getTreeWriter().write(tagname, output);
   }
   void TreeBufferWriter::write(const std::string& tagname, const multi1d<long int>& output)
   {
     return getTreeWriter().write(tagname, output);
   }
   void TreeBufferWriter::write(const std::string& tagname, const multi1d<unsigned long int>& output)
   {
     return getTreeWriter().write(tagname, output);
   }
   void TreeBufferWriter::write(const std::string& tagname, const multi1d<float>& output)
   {
     return getTreeWriter().write(tagname, output);
   }
   void TreeBufferWriter::write(const std::string& tagname, const multi1d<double>& output)
   {
     return getTreeWriter().write(tagname, output);
   }
   void TreeBufferWriter::write(const std::string& tagname, const multi1d<bool>& output)
   {
     return getTreeWriter().write(tagname, output);
   }
 #endif


   //--------------------------------------------------------------------------------
   // Tree handle class for arrays
   TreeArrayWriter::TreeArrayWriter(TreeWriter& tree_out, int size) : 
     output_tree(tree_out), array_size(size)
   {
     usedP = initP = false;
     open_elem = 0;
     elements_written = 0;
   }

   TreeArrayWriter::~TreeArrayWriter()
   {
     if (initP)
       closeArray();
   }

   //! Returns the number of array elements written
   int TreeArrayWriter::size() const {return array_size;}

   // Get next array element name
   std::string TreeArrayWriter::nextElem() 
   {
     return arrayElem(elements_written++);
   }

   // Start a structur
   void TreeArrayWriter::openArray(const string& tagname)
   {
     if (usedP)
     {
       QDPIO::cerr << __func__ << ": TreeArrayWriter: has been opened before. It cannot be used twice" << endl;
       QDP_abort(1);
     }

     if (initP)
     {
       QDPIO::cerr << __func__ << ": TreeArrayWriter: calling openArray twice" << endl;
       QDP_abort(1);
     }

     openTag(tagname);   // array tagname

     usedP = initP = true; // it has now been used and initialized
     open_elem = 0;
   }

   void TreeArrayWriter::closeArray()
   {
     if (! initP)
     {
       QDPIO::cerr << __func__ << ": TreeArrayWriter: calling closeArray but not initialized" << endl;
       QDP_abort(1);
     }

     for(;open_elem > 0; --open_elem)
       closeStruct();

   QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;
     // Record the array size
//   writeArraySize(elements_written);
     write("size",elements_written);
  QDPIO::cout << __PRETTY_FUNCTION__ << ": line= " << __LINE__ << endl;

    closeTag();   // array tagname

    if (array_size > 0 && elements_written != array_size)
    {
      QDPIO::cerr << "TreeArrayWriter: failed to write all the " << array_size 
		  << " required elements: instead = "
		  << elements_written << endl;
      QDP_abort(1);
    }

    // NOTE: should do something important here with array size,
    // but not sure how to handle the info yet

    // Reset
    initP = false;  // it is not initialized anymore, but it has been used
    elements_written = 0;
  }

  // Start a structur
  void TreeArrayWriter::openStruct(const string& tagname)
  {
    openTag(tagname);   // element tagname 
    ++open_elem;
  }

  void TreeArrayWriter::closeStruct()
  {
    if (open_elem <= 0)
    {
      QDPIO::cerr << __func__ << ": TreeArrayWriter: calling pop() with no corresponding push()" << endl;
      QDP_abort(1);
    }

    closeTag();   // element tagname
    --open_elem;
  }


  // Push an array group name
  void push(TreeArrayWriter& tree, const std::string& s)
  {
    tree.openArray(s);
  }

  // Pop an array group name
  void pop(TreeArrayWriter& tree)
  {
    tree.closeArray();
  }

  // Push a group name
  void pushElem(TreeArrayWriter& tree) {tree.openStruct(tree.nextElem());}

  // Pop a group name
  void popElem(TreeArrayWriter& tree) {tree.closeStruct();}



} // namespace QDP;
