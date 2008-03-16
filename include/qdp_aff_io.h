// -*- C++ -*-
// $Id: qdp_aff_io.h,v 1.1.2.3 2008-03-16 16:07:20 edwards Exp $

/*! @file
 * @brief AFF IO support
 */

#ifndef QDP_AFF_IO_H
#define QDP_AFF_IO_H

#include <string>
#include <sstream>

namespace QDP 
{

  // Forward declarations
  class LAFFReaderImp;
  class AFFFileWriterImp;

  /*! @ingroup io
   * @{
   */

  //--------------------------------------------------------------------------------
  //! AFF reader class
  /*!
    This is used to read data from an AFF files.

    Note that only the primary node opens and reads AFF files. Results from
    queries are broadcast to all nodes.
  */
  class AFFReader : public TreeReader
  {
  public:
    //! Empty constructor
    AFFReader();

    //! Construct from contents of file
    /*!
      Opens and reads an AFF file.
      \param filename The name of the file.
    */
    AFFReader(const std::string& filename);

    //! Clone a reader but with a possibly different path
    AFFReader(AFFReader& old, const std::string& xpath);

    //! Destructor
    ~AFFReader();

    /* The meaning of these should be clear to you */

    //! Opens and reads an AFF file.
    /*!
      \param filename The name of the file
      \post Any previously opened file is closed.
    */
    void open(const std::string& filename);

    //! Queries whether the binary file is open
    /*!
      \return true if the binary file is open; false otherwise.
    */
    bool is_open();

    //! Queries whether the AFF data has been obtained from another AFFReader
    /*!
      A private method allows this AFFReader to be copy the contents of
      another.
    */
    bool is_derived() const;

    //! Closes the last file opened
    void close();
    
    //! Return the entire contents of the Reader as a TreeRep
    void treeRep(TreeRep& output);
        
    //! Return the current context as a TreeRep
    void treeRepCurrentContext(TreeRep& output);
        
    //! Does the result of this Xpath query exist?
    /*! THIS IS NEEDED. PROBABLY WILL NOT SUPPORT GENERIC XPATH */
    bool exist(const std::string& xpath);

    //! Count the number of occurances from the Xpath query
    /*! PROBABLY WILL NOT SUPPORT GENERIC XPATH */
//    int count(const std::string& xpath);

    //! Count the array element entries
    int countArrayElem();

    //! Return tag for array element n
    std::string arrayElem(int n) const;

  private:
    //! Hide the = operator
    void operator=(const AFFReader&) {}
  
    //! Hide the copy constructor
    AFFReader(const AFFReader&) {}
  
    void open(AFFReader& old, const std::string& xpath);

  protected:
    //! Internal cloning function
    TreeReaderImp* clone(const std::string& xpath);

    //! Return the implementation
    LAFFReaderImp& getAFFReader() const;
  };


  //--------------------------------------------------------------------------------
  //! Metadata output class
  /*!
    Use this to write AFF.
    Note that only the primary node writes AFF.
  */
  class AFFFileWriter : public TreeWriter
  {
  public:
    //! Default constructor
    AFFFileWriter();

    //! Constructor from a filename
    /*!
      \param filename The name of the file
      \param write_prologue Whether to write the standard opening line of
      AFF files. Defaults to true.
    */
    explicit AFFFileWriter(const std::string& filename);

    //! Destructor
    ~AFFFileWriter();

    //! Writes an opening Tree tag    
    /*!
      \param tagname The name of the tag
    */
    void openTag(const std::string& tagname);

    //! Closes a tag
    void closeTag();

    //! Writes an opening AFF tag    
    /*!
      \param tagname The name of the tag
    */
    void openStruct(const std::string& tagname);
    void closeStruct();

    //! Return tag for array element n
    std::string arrayElem(int n) const;

    //!Opens a file
    /*!
      \param filename The name of the file
      \param write_prologue Whether to write the standard opening line of
      AFF files. Defaults to true.
    */
    void open(const std::string& filename);

    //! Return true if some failure occurred in previous IO operation
    bool fail() const;

    //! Flush the output. Maybe a nop.
    void flush();

    //! Closes the last  file  opened.
    void close();
       
    //! Queries whether the binary file is open
    /*!
      \return true if the binary file is open; false otherwise.
    */
    bool is_open();

    // Overloaded Writer Functions

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

  private:
    // Hide copy constructors
    AFFFileWriter(const AFFFileWriter&) {}

    // Hide copy constructors
    void operator=(const AFFFileWriter&) {}

  protected:
    //! The actual implementation
    /*! Cannot use covariant return rule since AFFFileWriterImp is not fully declared */
    AFFFileWriterImp& getAFFFileWriter() const;

  private:
    bool              initP;       /*!< Has this buffer been initialized? */
    AFFFileWriterImp* file_writer; /*<! The output writer */
  };

  /*! @} */   // end of group io

} // namespace QDP

#endif
