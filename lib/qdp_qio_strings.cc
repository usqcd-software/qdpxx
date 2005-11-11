#include "qdp.h"

namespace QDP {
  namespace QIOStrings { 
    
    // Catch all base (Hopefully never called)
    template<typename T> 
    void QIOTypeStringFromType(std::string& tname, const T& t) 
    {    
      tname = "QDP_GenericType";
    }
    
    // Backward compatibility
    template<typename T>
    void QIOTypeStringFromType(std::string& tname , const OScalar<T>& t) 
    { 
      tname  = "Scalar";
    }
    
    // Backward compatibility
    template<typename T>
    void QIOTypeStringFromType(std::string& tname , 
			       const multi1d< OScalar<T> >& t) 
    { 
      tname  = "Scalar";
    }
    
    // Backward Compatibility
    template<typename T>
    void QIOTypeStringFromType(std::string& tname , const OLattice<T>& t) 
    {
      tname  = "Lattice";
    }
    
    // Backward Compatibility
    template<typename T>
    void QIOTypeStringFromType(std::string& tname , 
			       const multi1d< OLattice<T> >& t) 
    {
      tname  = "Lattice";
    }
    
    
    // Specialisation
    // Gauge Field Type: multi1d<LatticeColorMatrix> 
  // Need specific type string to output in ILDG format with QIO
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
