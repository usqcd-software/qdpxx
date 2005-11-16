#include "qdp.h"

namespace QDP {
  namespace QIOStrings { 
    
    
    // Specialisation
    // Gauge Field Type: multi1d<LatticeColorMatrix> 
    // Need specific type string to output in ILDG format with QIO
    // FOR SOME F***ED UP TEMPLATING REASON THESE CANNOT LIVE IN 
    // QDP_QDPIO_H withouth causing linkage errors. WHY?
    template<>
    void QIOTypeStringFromType(std::string& tname , 
			       const multi1d< LatticeColorMatrixF3 >& t ) 
    {
      tname  = "QDP_F3_ColorMatrix";
    }
  
    template<> 
    void QIOTypeStringFromType(std::string& tname , 
			       const multi1d< LatticeColorMatrixD3 >& t)
    {
      tname  = "QDP_D3_ColorMatrix";
    }
    
    char QIOSizeToStr(size_t size) { 
      char s;
      switch( size ) { 
      case 4: 
	s = 'F';
	break;
      case 8:
	s = 'D';
	break;
      default:
	QDPIO::cerr << "Unsupported QIO precision with size: " << size << endl;
	QDP_abort(1);
      }  
      return s;
    }
    
  
  }; // End Namespace QIOStrings

}
