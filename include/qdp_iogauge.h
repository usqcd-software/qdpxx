// -*- C++ -*-
// $Id: qdp_iogauge.h,v 1.1 2003-10-15 16:53:24 edwards Exp $

/*! @file
 * @brief NERSC Archive gauge support
 *
 * Functions for reading and writing gauge fields in NERSC Archive format
 */

#ifndef QDP_IOGAUGE_INCLUDE
#define QDP_IOGAUGE_INCLUDE


QDP_BEGIN_NAMESPACE(QDP);


//! Archive gauge field header
struct ArchivGauge_t
{
  multi1d<int> nrow;       // Lattice size
  multi1d<int> boundary;   // Lattice size

  Real  w_plaq;
  Real  link;

  /* assume matrix size is 12 (matrix is compressed) 
     and change if we find out otherwise */
  size_t mat_size;

  int ensemble_id;
  int sequence_number;
  std::string ensemble_label;
  std::string creator;
  std::string creator_hardware;
  std::string creation_date;
  std::string archive_date;
};


//! Initialize header with default values
/*!
 * \ingroup io
 *
 * \param header     structure holding config info ( Modify )
 */    
void archivGaugeInit(ArchivGauge_t& header);

//! Source header read
void read(XMLReader& xml, const string& path, ArchivGauge_t& header);

//! Source header writer
void write(XMLWriter& xml, const string& path, const ArchivGauge_t& header);


//! Read a QCD (NERSC) Archive format gauge field
/*!
 * \ingroup io
 *
 * \param header     structure holding config info ( Modify )
 * \param u          gauge configuration ( Modify )
 * \param file       path ( Read )
 */    
void readArchiv(ArchivGauge_t& header, multi1d<LatticeColorMatrix>& u, const string& file);

//! Read a Archive configuration file
/*!
 * \ingroup io
 *
 * \param xml        xml reader holding config info ( Modify )
 * \param u          gauge configuration ( Modify )
 * \param cfg_file   path ( Read )
 */    
void readArchiv(XMLReader& xml, multi1d<LatticeColorMatrix>& u, const string& cfg_file);

//! Read a QCD (NERSC) Archive format gauge field
/*!
 * \ingroup io
 *
 * \param u          gauge configuration ( Modify )
 * \param cfg_file   path ( Read )
 */    
void readArchiv(multi1d<LatticeColorMatrix>& u, const string& cfg_file);


//! Write a QCD (NERSC) Archive format gauge field
/*!
 * \ingroup io
 *
 * \param xml        xml writer holding config info ( Modify )
 * \param u          gauge configuration ( Modify )
 * \param file       path ( Read )
 */    
void writeArchiv(ArchivGauge_t& header, const multi1d<LatticeColorMatrix>& u, const string& file);

//! Write a Archive configuration file
/*!
 * \ingroup io
 *
 * \param xml        xml writer holding config info ( Read )
 * \param u          gauge configuration ( Read )
 * \param cfg_file   path ( Read )
 */    

void writeArchiv(XMLBufferWriter& xml, const multi1d<LatticeColorMatrix>& u, 
		 const string& cfg_file);

//! Write a Archive configuration file
/*!
 * \ingroup io
 *
 * \param u          gauge configuration ( Read )
 * \param cfg_file   path ( Read )
 */    

void writeArchiv(const multi1d<LatticeColorMatrix>& u, 
		 const string& cfg_file);


QDP_END_NAMESPACE();

#endif
