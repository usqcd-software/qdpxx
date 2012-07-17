#include "qdp.h"


namespace QDP {


  ParseTag::ParseTag( const QDPJitArgs& jitArgs , const string& idxName): jitArgs(jitArgs), stringIdx(idxName) {}

  const QDPJitArgs& ParseTag::getJitArgs() const { 
    return jitArgs; 
  }


  bool ParseTag::insertObject( string& strIdentifier , const string& strType, const FnPeekColorMatrix & op ) const 
  {
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("ParseTag::insertObject FnPeekColorMatrix");
#endif
    int pos_row = jitArgs.addInt( op.getRow() );
    int pos_col = jitArgs.addInt( op.getCol() );
    ostringstream code;
    code << strType << "(" 
	 << jitArgs.getPtrName() << "[ " << pos_row  << " ].Int, " 
	 << jitArgs.getPtrName() << "[ " << pos_col  << " ].Int " 
	 << ")";
    strIdentifier = code.str();
    return true;
  }

  bool ParseTag::insertObject( string& strIdentifier , const string& strType, const FnPeekColorVector & op ) const 
  {
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("ParseTag::insertObject FnPeekColorVector");
#endif
    int pos_row = jitArgs.addInt( op.getRow() );
    ostringstream code;
    code << strType << "(" 
	 << jitArgs.getPtrName() << "[ " << pos_row  << " ].Int " 
	 << ")";
    strIdentifier = code.str();
    return true;
  }

  bool ParseTag::insertObject( string& strIdentifier , const string& strType, const FnPeekSpinMatrix & op ) const 
  {
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("ParseTag::insertObject FnPeekSpinMatrix ");
#endif
    int pos_row = jitArgs.addInt( op.getRow() );
    int pos_col = jitArgs.addInt( op.getCol() );
    ostringstream code;
    code << strType << "(" 
	 << jitArgs.getPtrName() << "[ " << pos_row  << " ].Int, " 
	 << jitArgs.getPtrName() << "[ " << pos_col  << " ].Int " 
	 << ")";
    strIdentifier = code.str();
    return true;
  }

  bool ParseTag::insertObject( string& strIdentifier , const string& strType, const FnPeekSpinVector & op ) const 
  {
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("ParseTag::insertObject FnPeekSpinVector");
#endif
    int pos_row = jitArgs.addInt( op.getRow() );
    ostringstream code;
    code << strType << "(" 
	 << jitArgs.getPtrName() << "[ " << pos_row  << " ].Int " 
	 << ")";
    strIdentifier = code.str();
    return true;
  }
  bool ParseTag::insertObject( string& strIdentifier , const string& strType, const FnPokeColorMatrix & op ) const 
  {
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("ParseTag::insertObject FnPokeColorMatrix");
#endif
    int pos_row = jitArgs.addInt( op.getRow() );
    int pos_col = jitArgs.addInt( op.getCol() );
    ostringstream code;
    code << strType << "(" 
	 << jitArgs.getPtrName() << "[ " << pos_row  << " ].Int, " 
	 << jitArgs.getPtrName() << "[ " << pos_col  << " ].Int " 
	 << ")";
    strIdentifier = code.str();
    return true;
  }
  bool ParseTag::insertObject( string& strIdentifier , const string& strType, const FnPokeColorVector & op ) const 
  {
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("ParseTag::insertObject FnPokeColorVector");
#endif
    int pos_row = jitArgs.addInt( op.getRow() );
    ostringstream code;
    code << strType << "(" 
	 << jitArgs.getPtrName() << "[ " << pos_row  << " ].Int " 
	 << ")";
    strIdentifier = code.str();
    return true;
  }
  bool ParseTag::insertObject( string& strIdentifier , const string& strType, const FnPokeSpinMatrix & op ) const 
  {
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("ParseTag::insertObject FnPokeSpinMatrix");
#endif
    int pos_row = jitArgs.addInt( op.getRow() );
    int pos_col = jitArgs.addInt( op.getCol() );
    ostringstream code;
    code << strType << "(" 
	 << jitArgs.getPtrName() << "[ " << pos_row  << " ].Int, " 
	 << jitArgs.getPtrName() << "[ " << pos_col  << " ].Int " 
	 << ")";
    strIdentifier = code.str();
    return true;
  }
  bool ParseTag::insertObject( string& strIdentifier , const string& strType, const FnPokeSpinVector & op ) const 
  {
#ifdef GPU_DEBUG_DEEP
    QDP_debug_deep("ParseTag::insertObject FnPokeSpinVector");
#endif
    int pos_row = jitArgs.addInt( op.getRow() );
    ostringstream code;
    code << strType << "(" 
	 << jitArgs.getPtrName() << "[ " << pos_row  << " ].Int " 
	 << ")";
    strIdentifier = code.str();
    return true;
  }

  string ParseTag::getIndex() const { return stringIdx; } 

  void getTypeString( string& typeString , const float& l )
  {
    typeString = "float";
  }

  void getTypeString( string& typeString , const double& l )
  {
    typeString = "double";
  }



}
