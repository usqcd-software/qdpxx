// $Id: qdp_tree_types.cc,v 1.1.2.1 2008-03-17 03:55:36 edwards Exp $
//
/*! @file
 * @brief Tree IO support
 */

#include "qdp.h"
#include "qdp_tree_types.h"
#include "qdp_tree_imp.h"
#include "qdp_aff_imp.h"
#include "qdp_xml_imp.h"

namespace QDP 
{

  using std::string;

  //--------------------------------------------------------------------------------
  // TreeReader implementation factory
  /*
    Construct a TreeReader implementation given a key

    This is a very simple implementation. No maps here.
  */
  TreeReaderImp* createTreeReader(const std::string& id)
  {
    TreeReaderType id_enum;

    if (id == "XMLReader")
      id_enum = TREE_READER_TYPE_XML;
    else if (id == "LazyAFFReader")
      id_enum = TREE_READER_TYPE_LAZY_AFF;
    else if (id == "DocumentAFFReader")
      id_enum = TREE_READER_TYPE_DOCUMENT_AFF;
    else
    {
      QDPIO::cerr << __func__ << ": unknown id of a TreeReader: id= " << id << endl;
      QDP_abort(1);
    }

    return createTreeReader(id_enum);
  }


  // TreeReader implementation factory
  TreeReaderImp* createTreeReader(enum TreeReaderType id)
  {
    TreeReaderImp* obj = 0;

    switch (id)
    {
    case TREE_READER_TYPE_XML:
      obj = new XMLReaderImp();
      break;

    case TREE_READER_TYPE_LAZY_AFF:
      obj = new LAFFReaderImp();
      break;

    case TREE_READER_TYPE_DOCUMENT_AFF:
      QDPIO::cerr << __func__ << ": unsupported TreeReader: id= " << id << endl;
      QDP_abort(1);
      break;

    default:
      QDPIO::cerr << __func__ << ": unknown id of a TreeReader: id= " << id << endl;
      QDP_abort(1);
    }

    return obj;
  }


  //--------------------------------------------------------------------------------
  //! TreeWriter implementation factory
  /*!
    Construct a TreeWriter implementation given a key
  */
  TreeWriterImp* createTreeWriter(const std::string& id)
  {
    TreeWriterType id_enum;

    if (id == "XMLWriter")
      id_enum = TREE_WRITER_TYPE_XML;
    else if (id == "AFFWriter")
      id_enum = TREE_WRITER_TYPE_AFF;
    else
    {
      QDPIO::cerr << __func__ << ": unknown id of a TreeWriter: id= " << id << endl;
      QDP_abort(1);
    }

    return createTreeWriter(id_enum);
  }

  // TreeWriter implementation factory
  /*
    Construct a TreeWriter implementation given a key
  */
  TreeWriterImp* createTreeWriter(enum TreeWriterType id)
  {
    TreeWriterImp* obj = 0;

    switch (id)
    {
    case TREE_WRITER_TYPE_XML:
      obj = new XMLFileWriterImp();
      break;

    case TREE_WRITER_TYPE_AFF:
      obj = new AFFFileWriterImp();
      break;

    default:
      QDPIO::cerr << __func__ << ": unknown id of a TreeWriter: id= " << id << endl;
      QDP_abort(1);
    }

    return obj;
  }


} // namespace QDP;
