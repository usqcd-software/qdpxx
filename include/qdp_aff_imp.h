// -*- C++ -*-
// $Id: qdp_aff_imp.h,v 1.1.2.3 2008-03-16 16:07:20 edwards Exp $
/*! @file
 * @brief AFF IO support via trees
 */

#ifndef QDP_AFF_IMP_H
#define QDP_AFF_IMP_H

#include <string>
#include <list>

#include "qdp_tree_imp.h"

// Forward declarations
struct AffReader_s;
struct AffWriter_s;
struct AffNode_s;


namespace QDP 
{

  /*! @ingroup io
   * @{
   */

  //--------------------------------------------------------------------------------
  //! AFF reader class
  /*!
    This is used to read data from an AFF file using Xpath like queries.

    Note that only the primary node opens and reads AFF files. Results from
    Xpath queries are broadcast to all nodes.
  */
  class AFFReaderImp : public TreeReaderImp
  {
  public:
    //! Destructor
    virtual ~AFFReaderImp() {}

    //! Clone a reader from a different path
    /*! Use covariant return rule */
    AFFReaderImp* clone(const std::string& xpath) = 0;

    //! Return tag for array element n
    std::string arrayElem(int n) const;
  };


  //--------------------------------------------------------------------------------
  //! "Lazy" AFF reader class
  /*!
    This is used to read data from an AFF file using Xpath like queries.
    Here, the file is opened and the symbol table read, but the file
    is not all slurped into memory. Queries go against the file on disk,
    hence a "lazy" reader in the computer science sense.

    Note that only the primary node opens and reads AFF files. Results from
    Xpath queries are broadcast to all nodes.
  */
  class LAFFReaderImp : public AFFReaderImp
  {
  public:
    //! Empty constructor
    LAFFReaderImp();

    //! Construct from contents of file
    /*!
      Opens and reads an AFF file.
      \param filename The name of the file.
    */
    explicit LAFFReaderImp(const std::string& filename);

    //! Clone a reader but with a possibly different path
    explicit LAFFReaderImp(LAFFReaderImp& old, const std::string& xpath);

    //! Destructor
    ~LAFFReaderImp();

    //! Clone a reader from a different path
    LAFFReaderImp* clone(const std::string& xpath);

    /* The meaning of these should be clear to you */

    //! Opens and reads an AFF file.
    /*!
      \param filename The name of the file
      \post Any previously opened file is closed.
    */
    void open(const std::string& filename);

    //! Clone a reader
    void open(LAFFReaderImp& old, const std::string& xpath);

    //! Queries whether the binary file is open
    /*!
      \return true if the binary file is open; false otherwise.
    */
    bool is_open();

    //! Queries whether the AFF data has been obtained from another LAFFReaderImp
    /*!
      A private method allows this LAFFReaderImp to be copy the contents of
      another.
    */
    bool is_derived() const;

    //! Closes the last file opened
    void close();
    
    /* So should these, there is just a lot of overloading */
    //! Xpath query
    void read(const std::string& xpath, std::string& result);
    //! Xpath query
    void read(const std::string& xpath, int& result);
    //! Xpath query
    void read(const std::string& xpath, unsigned int& result);
    //! Xpath query
    void read(const std::string& xpath, short int& result);
    //! Xpath query
    void read(const std::string& xpath, unsigned short int& result);
    //! Xpath query
    void read(const std::string& xpath, long int& result);
    //! Xpath query
    void read(const std::string& xpath, unsigned long int& result);
    //! Xpath query
    void read(const std::string& xpath, float& result);
    //! Xpath query
    void read(const std::string& xpath, double& result);
    //! Xpath query
    void read(const std::string& xpath, bool& result);

    /* Overloadings of primitive(elemental) array objects */
    //! Xpath query
    void read(const std::string& xpath, multi1d<int>& result);
    //! Xpath query
    void read(const std::string& xpath, multi1d<unsigned int>& result);
    //! Xpath query
    void read(const std::string& xpath, multi1d<short int>& result);
    //! Xpath query
    void read(const std::string& xpath, multi1d<unsigned short int>& result);
    //! Xpath query
    void read(const std::string& xpath, multi1d<long int>& result);
    //! Xpath query
    void read(const std::string& xpath, multi1d<unsigned long int>& result);
    //! Xpath query
    void read(const std::string& xpath, multi1d<float>& result);
    //! Xpath query
    void read(const std::string& xpath, multi1d<double>& result);
    //! Xpath query
    void read(const std::string& xpath, multi1d<bool>& result);

    //! Return the entire contents of the Reader as a TreeRep
    void treeRep(TreeRep& output);
        
    //! Return the current context as a TreeRep
    void treeRepCurrentContext(TreeRep& output);
        
    //! Count the number of occurances from the Xpath query
    bool exist(const std::string& xpath);

    //! Count the number of occurances from the Xpath query
    int count(const std::string& xpath);

    //! Count the array element entries
    int countArrayElem();

  private:
    //! Hide the = operator
    void operator=(const LAFFReaderImp&) {}
  
    //! Hide the copy constructor
    LAFFReaderImp(const LAFFReaderImp&) {}
  
  private:
    //! Check for errors
    void checkError() const;

    //! Output not implemented
    void notImplemented(const std::string& func) const;

  private:
    AffReader_s* getIstream() const;

  private:
    bool         derivedP; /*!< is this reader derived from another reader? */
    bool         initP;   /*!< is it initialized */
    AffNode_s*   current_dir;
    AffReader_s* input_stream;
  };


  //--------------------------------------------------------------------------------
  //! Metadata output class
  /*!
    Use this to write AFF.When closing tags, you do not have to specify which
    tag to close since this class will remember the order in which you opened
    the tags and close them in reverse order to ensure well-formed AFF.

    Note that only the primary node writes AFF.
  */
  class AFFFileWriterImp : public TreeWriterImp
  {
  public:
    //! Default constructor
    AFFFileWriterImp();

    //! Constructor from a filename
    /*!
      \param filename The name of the file
    */
    explicit AFFFileWriterImp(const std::string& filename);

    //! Destructor
    ~AFFFileWriterImp();

    //! Queries whether the binary file is open
    /*!
      \return true if the binary file is open; false otherwise.
    */
    bool is_open();

    //!Opens a file
    /*!
      \param filename The name of the file
      \param write_prologue Whether to write the standard opening line of
      AFF files. Defaults to true.
    */
    void open(const std::string& filename);

    //! Flush the buffer
    void flush();

    //! Return true if some failure occurred in previous IO operation
    bool fail() const;

    //! Closes the last file opened.
    void close();
        
    //! Writes an opening AFF tag    
    /*!
      \param tagname The name of the tag
    */
    virtual void openStruct(const std::string& tagname);
    virtual void closeStruct();

    //! Writes an opening AFF tag    
    /*!
      \param tagname The name of the tag
    */
    void openTag(const std::string& tagname);

    //! Closes a tag
    void closeTag();

    //! Return tag for array element n
    std::string arrayElem(int n) const;

    // Overloaded Writer Functions
    //! Write tag and contents
    void write(const std::string& tagname, const std::string& output);
    //! Write tag and contents
    void write(const std::string& tagname, const int& output);
    //! Write tag contents
    void write(const std::string& tagname, const unsigned int& output);
    //! Write tag and contents
    void write(const std::string& tagname, const short int& output);
    //! Write tag and contents
    void write(const std::string& tagname, const unsigned short int& output);
    //! Write tag and contents
    void write(const std::string& tagname, const long int& output);
    //! Write tag and contents
    void write(const std::string& tagname, const unsigned long int& output);
    //! Write tag and contents
    void write(const std::string& tagname, const float& output);
    //! Write tag and contents
    void write(const std::string& tagname, const double& output);
    //! Write tag and contents
    void write(const std::string& tagname, const bool& output);

    // Overloaded array (elemental list) Writer Functions
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<int>& output);
    //! Write tag contents
    void write(const std::string& tagname, const multi1d<unsigned int>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<short int>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<unsigned short int>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<long int>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<unsigned long int>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<float>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<double>& output);
    //! Write tag and contents
    void write(const std::string& tagname, const multi1d<bool>& output);

  protected:
    //! Hide the = operator
    void operator=(const AFFFileWriterImp&) {}
  
    //! Hide the copy constructor
    AFFFileWriterImp(const AFFFileWriterImp&) {}
  
  protected:
    // Write an array of basic types
    template<typename T>
    void writeArrayPrimitive(const std::string& tagname, const multi1d<T>& output);

  private:
    //! Check for errors
    void checkError() const;

    //! Output not implemented
    void notImplemented(const std::string& func) const;

  private:
    AffWriter_s* getOstream() const;

  private:
    bool         initP;
    AffNode_s*   current_node;
    AffWriter_s* output_stream;
  };


  /*! @} */   // end of group io

} // namespace QDP

#endif
