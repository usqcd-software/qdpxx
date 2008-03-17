// -*- C++ -*-
// $Id: qdp_tree_types.h,v 1.1.2.1 2008-03-17 03:55:36 edwards Exp $
/*! @file
 * @brief Tree IO support
 *
 * Support for parent class Tree representation
 */

#ifndef QDP_TREE_TYPES_H
#define QDP_TREE_TYPES_H

namespace QDP 
{

  // Forward declarations
  class TreeReaderImp;
  class TreeWriterImp;


  /*! @ingroup io
   * @{
   */

  //--------------------------------------------------------------------------------
  //! Types of TreeReaders
  enum TreeReaderType 
  {
    TREE_READER_TYPE_XML,
    TREE_READER_TYPE_LAZY_AFF,
    TREE_READER_TYPE_DOCUMENT_AFF
  };


  //--------------------------------------------------------------------------------
  //! Types of TreeWriter
  enum TreeWriterType 
  {
    TREE_WRITER_TYPE_XML,
    TREE_WRITER_TYPE_AFF
  };


  //--------------------------------------------------------------------------------
  //! TreeReader implementation factory
  /*!
    Construct a TreeReader implementation given a key
  */
  TreeReaderImp* createTreeReader(const std::string& id);

  //! Tree reader implementation factory
  /*!
    Construct a TreeReader implementation given a key
  */
  TreeReaderImp* createTreeReader(enum TreeReaderType id);


  //--------------------------------------------------------------------------------
  //! TreeWriter implementation factory
  /*!
    Construct a TreeWriter implementation given a key
  */
  TreeWriterImp* createTreeWriter(const std::string& id);

  //! Tree reader implementation factory
  /*!
    Construct a TreeWriter implementation given a key
  */
  TreeWriterImp* createTreeWriter(enum TreeWriterType id);


  /*! @} */   // end of group io

} // namespace QDP

#endif
