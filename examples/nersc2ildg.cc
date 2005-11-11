// $Id: nersc2ildg.cc,v 1.1 2005-11-11 21:18:54 bjoo Exp $
/*! \file
 *  \brief Skeleton of a QDP main program
 */

#include "qdp.h"
#include "qdp_iogauge.h"
#include <iostream>
#include <string>

using namespace QDP;

typedef struct { 
  std::string NERSC_file_name;
  std::string ILDG_file_name;
  std::string dataLFN;
} UserInput;

int main(int argc, char *argv[])
{
  // Put the machine into a known state
  QDP_initialize(&argc, &argv);

  // Setup the lattice size
  // NOTE: in general, the user will need/want to set the
  // lattice size at run-time
  multi1d<int> nrow(Nd);
  UserInput p;

  try { 
    XMLReader param("./nersc2ildg.xml");
    

    XMLReader paramtop(param, "/nersc2ildg");
    read(paramtop, "NERSC_file", p.NERSC_file_name);
    read(paramtop, "ILDG_file", p.ILDG_file_name);
    read(paramtop, "dataLFN", p.dataLFN);
    read(paramtop, "nrow", nrow);

  } catch(const std::string& e) { 
    QDPIO::cout << "Caught exception while reading XML: " << e << endl;
    QDP_abort(1);
  }

  // Insert the lattice size into the Layout
  // There can be additional calls here like setting the number of
  // processors to use in an SMP implementation
  Layout::setLattSize(nrow);
  
  // Create the lattice layout
  // Afterwards, QDP is useable
  Layout::create();

  // Try to read the NERSC Archive file
  multi1d<LatticeColorMatrix> u(Nd);
  readArchiv(u, p.NERSC_file_name);

  // Do checks here.
  XMLBufferWriter file_metadata;
  push(file_metadata, "file_metadata");
  write(file_metadata, "annotation", "NERSC Config Converted by QDP++ NERS2ILDG");
  pop(file_metadata);


  QDPFileWriter ildg_out(file_metadata,  
			 p.ILDG_file_name,
			 QDPIO_SINGLEFILE,
			 QDPIO_SERIAL,
  			 p.dataLFN);

  XMLBufferWriter record_metadata;
  push(record_metadata, "record_metadata");
  write(record_metadata, "annotation", "NERSC Config Record Converted by QDP++ NERSC2ILDG");
  pop(record_metadata);

  ildg_out.write(record_metadata, u);
  ildg_out.close();
			 		      

  // Possibly shutdown the machine
  QDP_finalize();

  exit(0);
}
