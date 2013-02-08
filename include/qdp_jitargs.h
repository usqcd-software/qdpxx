#ifndef QDP_JIT_ARGS
#define QDP_JIT_ARGS

#include<vector>

namespace QDP {

  class QDPJitArgs {
  public:
    QDPJitArgs();
    ~QDPJitArgs();
    string getPtrName() const;
    string getCode(int i) const;
    UnionDevPtr* getDevPtr();
    int addPtr(void * devicePtr) const;
    int addInt(int i) const;
    int addBool(bool b) const;
    int addIntPtr( int * intPtr) const;
    int addSize_t(size_t i) const;

  private:
    mutable int  size;
    mutable UnionDevPtr * arrayArgs;
    mutable std::vector< int >         vecType;
    static  std::vector< std::string > vecUnionMember;
    mutable int myId;
  };

}

#endif
