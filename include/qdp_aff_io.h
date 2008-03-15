// -*- C++ -*-
// $Id: qdp_aff_io.h,v 1.1.2.1 2008-03-15 14:28:54 edwards Exp $

/*! @file
 * @brief XML IO support
 */

#ifndef QDP_AFF_IO_H
#define QDP_AFF_IO_H

#include <string>
#include <sstream>

namespace QDP 
{

  // Forward declarations
  class AFFReader;
  class AFFWriter;
  class AFFBufferWriter;
  class AFFFileWriter;
  class AFFArrayWriter;

  /*! @ingroup io
   * @{
   */

#if 0

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

    //! Opens and reads an AFF file.
    /*!
      \param id The input stream of the file
      \post Any previously opened file is closed      
    */
    void open(std::istream& is);

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
    
    /* So should these, there is just a lot of overloading */
    //! Xpath query
    void get(const std::string& xpath, std::string& result);
    //! Xpath query
    void get(const std::string& xpath, int& result);
    //! Xpath query
    void get(const std::string& xpath, unsigned int& result);
    //! Xpath query
    void get(const std::string& xpath, short int& result);
    //! Xpath query
    void get(const std::string& xpath, unsigned short int& result);
    //! Xpath query
    void get(const std::string& xpath, long int& result);
    //! Xpath query
    void get(const std::string& xpath, unsigned long int& result);
    //! Xpath query
    void get(const std::string& xpath, float& result);
    //! Xpath query
    void get(const std::string& xpath, double& result);
    //! Xpath query
    void get(const std::string& xpath, bool& result);

    //! Set a replacement of a primitive
    template<typename T>
    void set(const std::string& xpath, const T& to_set) 
      {
	if (Layout::primaryNode())
	{  
	  BasicXPathReader::set<T>(xpath, to_set);
	}
      }


    //! Return the entire contents of the Reader as a stream
    void print(ostream& is);
        
    //! Print the current context
    void printCurrentContext(ostream& is);
        
    //! Count the number of occurances from the Xpath query
    int count(const std::string& xpath);

    void registerNamespace(const std::string& prefix, const std::string& uri);

  private:
    //! Hide the = operator
    void operator=(const AFFReader&) {}
  
    //! Hide the copy constructor
    AFFReader(const AFFReader&) {}
  
    void open(AFFReader& old, const std::string& xpath);
  protected:
    // The universal data-reader. All the read functions call this
    template<typename T>
    void
    readPrimitive(const std::string& xpath, T& output);

  private:
    bool  iop;  //file open or closed?
    bool  derived; // is this reader derived from another reader?
  };


  //--------------------------------------------------------------------------------
  //! Metadata output class
  /*!
    Use this to write AFF.When closing tags, you do not have to specify which
    tag to close since this class will remember the order in which you opened
    the tags and close them in reverse order to ensure well-formed AFF.

    Note that only the primary node writes AFF.
  */
  class AFFWriter : public TreeWriter
  {
  public:
    AFFWriter();

    // Virtual destructor
    virtual ~AFFWriter();

    //! Writes an opening AFF tag
    /*!
      \param tagname The name of the tag
    */
    virtual void openSimple(const std::string& tagname);
    virtual void closeSimple();

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

    //! Writes an opening AFF tag    
    /*!
      \param nsprefix A namespace prefix for the tag 
      \param tagname The name of the tag
    */
    void openTag(const std::string& nsprefix, const std::string& tagname);

    //! Writes an opening AFF tag    
    /*!
      \param tagname The name of the tag
      \param al A list of attributes for this tag
    */
    void openTag(const std::string& tagname, AFFWriterAPI::AttributeList& al);

    //! Writes an opening AFF tag    
    /*!
      \param nsprefix A namespace prefix for the tag 
      \param tagname The name of the tag
      \param al A list of attributes for this tag      
    */
    void openTag(const std::string& nsprefix,
		 const std::string& tagname, 
		 AFFWriterAPI::AttributeList& al);

    //! Closes a tag
    void closeTag();

    //! Writes an empty tag
    /*!
      \param tagname The name of the tag
    */
    void emptyTag(const std::string& tagname);

    //! Writes an empty tag
    /*!
      \param nsprefix A namespace prefix for the tag 
      \param tagname The name of the tag
    */
    void emptyTag(const std::string& nsprefix, const std::string& tagname);

    //! Writes an empty tag
    /*!
      \param tagname The name of the tag
      \param al A list of attributes for this tag            
    */
    void emptyTag(const std::string& tagname, AFFWriterAPI::AttributeList& al);

    //! Writes an empty tag
    /*!
      \param nsprefix A namespace prefix for the tag 
      \param tagname The name of the tag
      \param al A list of attributes for this tag            
    */
    void emptyTag(const std::string& nsprefix,
		  const std::string& tagname, 
		  AFFWriterAPI::AttributeList& al);
    

    // Overloaded Writer Functions

    //! Write tag contents
    void write(const std::string& output);
    //! Write tag contents
    void write(const int& output);
    //! Write tag contents
    void write(const unsigned int& output);
    //! Write tag contents
    void write(const short int& output);
    //! Write tag contents
    void write(const unsigned short int& output);
    //! Write tag contents
    void write(const long int& output);
    //! Write tag contents
    void write(const unsigned long int& output);
    //! Write tag contents
    void write(const float& output);
    //! Write tag contents
    void write(const double& output);
    //! Write tag contents
    void write(const bool& output);

    // Write all the AFF to std::string
    void writeAFF(const std::string& output);
  };




#if 0
  // NEED TO FIX THIS
  //! Write a AFF multi2d element
  template<class T> 
  inline
  void write(AFFWriter& xml, const std::string& s, const multi2d<T>& s1)
  {
    for(int j=0; j < s1.size1(); ++j)
      for(int i=0; i < s1.size2(); ++i)
      {
	std::ostringstream ost;
	if (Layout::primaryNode()) 
	  ost << s << "[ " << i << " ][ " << j << " ]";
	write(xml, ost.str(), s1[i][j]);
      }
  }

#endif


  //--------------------------------------------------------------------------------
  //! Writes AFF metadata to a file
  /*!
    \ingroup io
  */

  class AFFFileWriter : public AFFWriter
  {
  public:

    AFFFileWriter();

    //! Constructor from a filename
    /*!
      \param filename The name of the file
      \param write_prologue Whether to write the standard opening line of
      AFF files. Defaults to true.
    */
    explicit AFFFileWriter(const std::string& filename, bool write_prologue=true)
      {
	open(filename, write_prologue);
      }

    ~AFFFileWriter();

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
    void open(const std::string& filename, bool write_prologue=true);

    //! Flush the buffer
    void flush();

    //! Return true if some failure occurred in previous IO operation
    bool fail() const;

    //! Closes the last  file  opened.
    void close();
        
  private:
    ofstream output_stream;
    ostream& getOstream(void) {return output_stream;}
  };

#endif

  /*! @} */   // end of group io

} // namespace QDP

#endif
