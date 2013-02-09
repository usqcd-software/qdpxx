#include "qdp.h"

namespace QDP {

  std::vector< std::string > QDPJitArgs::vecUnionMember;

  QDPJitArgs::QDPJitArgs(): myId(-1) {

    vecArgs.reserve(DeviceParams::Instance().getMaxKernelArg());
    vecType.reserve(DeviceParams::Instance().getMaxKernelArg());

    vecUnionMember.push_back(".ptr");
    vecUnionMember.push_back(".Int");
    vecUnionMember.push_back(".Bool");
    vecUnionMember.push_back(".IntPtr");
    vecUnionMember.push_back(".Size_t");
  }


  QDPJitArgs::~QDPJitArgs() {
    if (myId >=0) 
      QDPCache::Instance().signoff( myId );
  }
  
  string QDPJitArgs::getPtrName() const  { return "args"; }

  string QDPJitArgs::getCode(int i) const {
    assert( i < vecArgs.size() );
    ostringstream code;
    code << getPtrName() << "[ " << i << " ]" << vecUnionMember[ vecType[i] ] << " ";
    return code.str();
  }

  UnionDevPtr* QDPJitArgs::getDevPtr() {
    assert( vecArgs.size() > 0 );
    if (myId < 0)
      myId = QDPCache::Instance().registrateOwnHostMem( sizeof(UnionDevPtr) * vecArgs.size() , (void*)vecArgs.data() );
    return (UnionDevPtr*)QDPCache::Instance().getDevicePtr( myId );
  }

  int QDPJitArgs::addPtr(void * devicePtr) const {
    assert( vecArgs.capacity() > vecArgs.size() );
    vecArgs.resize(vecArgs.size()+1);
    vecArgs.back().ptr = devicePtr;
    vecType.push_back(Ptr);
    return vecArgs.size()-1;
  }

  int QDPJitArgs::addInt(int i) const {
    assert( vecArgs.capacity() > vecArgs.size() );
    vecArgs.resize(vecArgs.size()+1);
    vecArgs.back().Int = i;
    vecType.push_back(Int);
    return vecArgs.size()-1;
  }

  int QDPJitArgs::addBool(bool b) const {
    assert( vecArgs.capacity() > vecArgs.size() );
    vecArgs.resize(vecArgs.size()+1);
    vecArgs.back().Bool = b;
    vecType.push_back(Bool);
    return vecArgs.size()-1;
  }

  int QDPJitArgs::addIntPtr( int * intPtr) const {
    assert( vecArgs.capacity() > vecArgs.size() );
    vecArgs.resize(vecArgs.size()+1);
    vecArgs.back().IntPtr = intPtr;
    vecType.push_back(IntPtr);
    return vecArgs.size()-1;
  }

  int QDPJitArgs::addSize_t(size_t i) const {
    assert( vecArgs.capacity() > vecArgs.size() );
    vecArgs.resize(vecArgs.size()+1);
    vecArgs.back().Size_t = i;
    vecType.push_back(Size_t);
    return vecArgs.size()-1;
  }


  bool operator== (const QDPJitArgs& a, const QDPJitArgs& b)
  {
    if (a.vecArgs.size() != b.vecArgs.size())
      return false;
    assert( a.vecArgs.size() == a.vecType.size() );
    assert( b.vecArgs.size() == b.vecType.size() );

    for ( int i = 0 ; i < a.vecArgs.size() ; i++ ) {
      if (a.vecType.at(i) != b.vecType.at(i))
	return false;
      switch ( a.vecType.at(i) ) {
      case QDPJitArgs::Ptr:    if (a.vecArgs.at(i).ptr    != b.vecArgs.at(i).ptr) return false; break;
      case QDPJitArgs::Int:    if (a.vecArgs.at(i).Int    != b.vecArgs.at(i).Int) return false; break;
      case QDPJitArgs::Bool:   if (a.vecArgs.at(i).Bool   != b.vecArgs.at(i).Bool) return false; break;
      case QDPJitArgs::IntPtr: if (a.vecArgs.at(i).IntPtr != b.vecArgs.at(i).IntPtr) return false; break;
      case QDPJitArgs::Size_t: if (a.vecArgs.at(i).Size_t != b.vecArgs.at(i).Size_t) return false; break;
      default:
	assert( !"Unknown type" );
      }
    }
    return true;
  }

 
  bool operator!= (const QDPJitArgs& a, const QDPJitArgs& b)
  {
    return !(a == b);
  }



}
